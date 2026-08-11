import numpy as np
import polars as pl
from project_paths import paths
from scipy.signal import savgol_filter

from src.features import prepare_features

STATIONS_PATH = paths.processed_data / "lidar_stations.parquet"
OUT_PATH = paths.processed_data / "lidar_features.parquet"

PROMINENCE_GATE = (
    0.25  # median prominence (m) below which the crest pick is a shoulder / not raised
)
RESID_FLOOR = (
    0.05  # residual std (m) below which roughness features are DTM speckle, not signal
)
MIN_STATIONS = 5  # fewer valid stations than this -> longitudinal features unreliable
TREND_POLY = 2  # savgol polyorder for the along-length terrain trend

# tiers whose DTM crest geometry is trustworthy; others keep only offset + the flag
GEOMETRY_TIERS = {"engineered_raised", "natural_geomorphic"}


def resid_features(z: np.ndarray) -> dict:
    """detrended-crest residual stats for one asset's crest_z series (chainage-ordered).

    the spline tracks the asset's own along-length terrain (a bank running downhill
    is fine); residuals are what's left after removing it. roughness stats are only
    meaningful above the DTM noise floor - below it they measure speckle, so they NaN.
    """
    n = len(z)
    win = min(n - (n + 1) % 2, 11)  # odd window <= n
    win = max(win, 5)
    trend = savgol_filter(z, window_length=win, polyorder=min(TREND_POLY, win - 1))
    r = z - trend
    rng = float(r.max() - r.min())

    resid_std = float(r.std())
    max_dip = float(-r.min())  # deepest single depression below trend

    above_floor = resid_std >= RESID_FLOOR
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if (above_floor and n > 2) else None
    tv_range = (
        float(np.abs(np.diff(r)).sum() / rng) if (above_floor and rng > 0) else None
    )

    return {
        "resid_std": resid_std,
        "resid_max_dip": max_dip,
        "resid_ac1": ac1,
        "resid_tv_range": tv_range,
    }


def aggregate_asset(g: pl.DataFrame) -> dict:
    """one station-group (already sorted by chainage) -> one feature row."""
    tier = g["tier"][0]
    n_stations = g.height
    offset_median = float(g["crest_off"].abs().median())
    prom_median = float(g["prominence"].median())

    # always-emitted confidence/geometry-agnostic fields
    out = {
        "lidar__tier": tier,
        "lidar__n_stations": n_stations,
        "lidar__crest_offset_median": offset_median,
        "lidar__crest_prominence_median": prom_median,
    }

    # gate: geometry only trusted for the right tiers, a real raised pick, enough stations
    geometry_valid = (
        tier in GEOMETRY_TIERS
        and prom_median >= PROMINENCE_GATE
        and n_stations >= MIN_STATIONS
    )
    out["lidar__crest_found"] = geometry_valid

    geom_cols = [
        "lidar__crest_relheight_mean",
        "lidar__crest_relheight_min",
        "lidar__crest_relheight_std",
        "lidar__crest_width_mean",
        "lidar__crest_resid_std",
        "lidar__crest_resid_max_dip",
        "lidar__crest_resid_ac1",
        "lidar__crest_resid_tv_range",
    ]
    if not geometry_valid:
        out.update(
            {c: None for c in geom_cols}
        )  # NaN, never 0 - absence is not a measured zero
        return out

    relheight = g["relheight"].to_numpy()
    z = g["crest_z"].to_numpy()
    rf = resid_features(z)

    out.update(
        {
            "lidar__crest_relheight_mean": float(np.mean(relheight)),
            "lidar__crest_relheight_min": float(np.min(relheight)),
            "lidar__crest_relheight_std": float(np.std(relheight)),
            "lidar__crest_width_mean": float(g["footprint_width"].mean()),
            "lidar__crest_resid_std": rf["resid_std"],
            "lidar__crest_resid_max_dip": rf["resid_max_dip"],
            "lidar__crest_resid_ac1": rf["resid_ac1"],
            "lidar__crest_resid_tv_range": rf["resid_tv_range"],
        }
    )
    return out


def build_lidar_features(stations: pl.DataFrame, scope: list[int]) -> pl.DataFrame:
    """one row per scoped asset. assets with no stations (no window / all-invalid) get a
    null row so the feature set spans the full modelled scope, not just the covered subset."""
    stations = stations.sort("asset_id", "chainage")

    rows = [
        aggregate_asset(g) | {"asset_id": aid}
        for (aid,), g in stations.group_by("asset_id", maintain_order=True)
    ]
    features = pl.DataFrame(rows)

    # left-join onto the full scope so uncovered assets are present with null features
    # (crest_found False, everything else null) rather than silently absent
    scope_df = pl.DataFrame({"asset_id": scope})
    features = scope_df.join(features, on="asset_id", how="left").with_columns(
        pl.col("lidar__crest_found").fill_null(False),
        pl.col("lidar__n_stations").fill_null(0),
    )
    return features


def main():
    stations = pl.read_parquet(STATIONS_PATH)
    scope = prepare_features(pl.read_parquet(paths.unified_file))["asset_id"].to_list()

    # stations must be a subset of scope - extract loaded from the same prepare_features set
    stray = set(stations["asset_id"].to_list()) - set(scope)
    if stray:
        raise ValueError(
            f"{len(stray)} station assets outside prepare_features scope, e.g. {list(stray)[:5]}"
        )

    features = build_lidar_features(stations, scope)

    # ---- output gates ----
    assert features["asset_id"].n_unique() == features.height, "duplicate asset rows"
    assert features["asset_id"].sort().equals(pl.Series(sorted(scope))), (
        "scope mismatch"
    )
    # crest_found implies the geometry columns are populated; not-found implies they are null
    found = features.filter(pl.col("lidar__crest_found"))
    assert found["lidar__crest_relheight_mean"].is_not_null().all(), (
        "found asset with null geometry"
    )
    notfound = features.filter(~pl.col("lidar__crest_found"))
    assert notfound["lidar__crest_relheight_mean"].is_null().all(), (
        "not-found asset with populated geometry"
    )
    # roughness only where residual std clears the floor
    rough = features.filter(pl.col("lidar__crest_resid_ac1").is_not_null())
    assert (rough["lidar__crest_resid_std"] >= RESID_FLOOR).all(), (
        "roughness present below noise floor"
    )

    features.write_parquet(OUT_PATH)
    n_found = int(features["lidar__crest_found"].sum())
    print(
        f"wrote {features.height} assets x {features.width - 1} cols -> {OUT_PATH}\n"
        f"  crest_found: {n_found} ({n_found / features.height:.1%})  "
        f"roughness populated: {int(features['lidar__crest_resid_ac1'].is_not_null().sum())}"
    )


if __name__ == "__main__":
    main()
