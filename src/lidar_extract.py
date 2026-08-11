import polars as pl
import numpy as np
from math import hypot
import geopandas as gpd
import rasterio
from project_paths import paths
from shapely.ops import linemerge

from src.features import prepare_features

WINDOWS = paths.data / "lidar" / "dtm_windows"
OUT_PATH = paths.processed_data / "lidar_stations.parquet"

SPACING = 5.0  # A: along-line station spacing (m)
HALF_W = 25.0  # B: transect half-width (m)
CREST_BAND = 10.0  # C: crest search band around the line (± m)
SMOOTH_M = 3  # D: cross-section smoothing window (cells)
TOE_RISE_EPS = 0.02  # D: toe = first local min walking out from crest
MIN_STATIONS = 3  # below this a longitudinal series is meaningless -> asset dropped
OFFSETS = np.arange(-HALF_W, HALF_W + 1.0, 1.0)

TIERS = {
    "Embankment": "engineered_raised",
    "Engineered High Ground": "engineered_raised",
    "Barrier Beach": "engineered_raised",
    "Cliff": "natural_geomorphic",
    "Dunes": "natural_geomorphic",
    "Beach": "natural_geomorphic",
    "Wall": "wall_like",
    "Quay": "wall_like",
    "Promenade": "wall_like",
    "Demountable Defence": "wall_like",
    "Flood Gate": "point",
    "Weir": "point",
    "Spillway": "point",
    "Bridge Abutment": "point",
}

STATION_SCHEMA = pl.Schema(
    {
        "asset_id": pl.Int64,
        "tier": pl.String,
        "chainage": pl.Float64,
        "crest_off": pl.Float64,
        "crest_z": pl.Float64,
        "toe_l_z": pl.Float64,
        "toe_r_z": pl.Float64,  # kept separate for the deferred wet/dry side split
        "prominence": pl.Float64,
        "relheight": pl.Float64,
        "footprint_width": pl.Float64,
    }
)


def load_windowed_assets() -> gpd.GeoDataFrame:
    scope = prepare_features(pl.read_parquet(paths.unified_file))["asset_id"].to_list()
    a = gpd.read_file(
        paths.aims_data / "aims.gpkg", columns=["asset_id", "asset_sub_type"]
    )
    a["asset_id"] = a["asset_id"].astype("int64")
    a = a[a["asset_id"].isin(scope)].reset_index(drop=True)
    if a.crs is None or a.crs.to_epsg() != 27700:
        raise ValueError(f"expected epsg:27700, got {a.crs}")
    on_disk = {int(p.stem) for p in WINDOWS.glob("*.tif")}
    return a[a["asset_id"].isin(on_disk)].reset_index(drop=True)


def longest_line(geom):
    if geom.geom_type == "LineString":
        return geom
    merged = linemerge(geom)
    if merged.geom_type == "LineString":
        return merged
    if merged.geom_type == "MultiLineString":
        return max(merged.geoms, key=lambda g: g.length)
    return None


def stations(line):
    d = np.arange(0.0, line.length, SPACING)
    pts, perps = np.empty((len(d), 2)), np.empty((len(d), 2))
    for k, di in enumerate(d):
        a = line.interpolate(max(0.0, di - SPACING / 2))
        b = line.interpolate(min(line.length, di + SPACING / 2))
        tx, ty = b.x - a.x, b.y - a.y
        n = hypot(tx, ty) or 1.0
        perps[k] = (-ty / n, tx / n)
        p = line.interpolate(di)
        pts[k] = (p.x, p.y)
    return d, pts, perps


def sample_sections(src, pts, perps):
    coords = (pts[:, None, :] + OFFSETS[None, :, None] * perps[:, None, :]).reshape(
        -1, 2
    )
    vals = np.array([v[0] for v in src.sample(coords)], dtype="float64")
    if src.nodata is not None:
        vals[vals == src.nodata] = np.nan
    return vals.reshape(len(pts), len(OFFSETS))


def crest_and_toes(row):
    band = np.abs(OFFSETS) <= CREST_BAND
    finite = np.isfinite(row)
    if not (finite & band).any():
        return None
    sm = np.convolve(
        np.nan_to_num(row, nan=np.nanmin(row)),
        np.ones(SMOOTH_M) / SMOOTH_M,
        mode="same",
    )
    ci = int(np.argmax(np.where(band, sm, -np.inf)))
    if not finite[ci]:  # crest cell itself was nodata -> invalid station
        return None

    def toe(step):
        i = ci
        while 0 < i + step < len(sm):
            nxt = i + step
            if not finite[nxt] or sm[nxt] > sm[i] + TOE_RISE_EPS:
                break
            i = nxt
        return i

    li, ri = toe(-1), toe(+1)
    crest_z, toe_l, toe_r = float(row[ci]), float(row[li]), float(row[ri])
    return {
        "crest_off": float(OFFSETS[ci]),
        "crest_z": crest_z,
        "toe_l_z": toe_l,
        "toe_r_z": toe_r,
        "prominence": crest_z
        - max(toe_l, toe_r),  # left raw & unclamped: <=0 flags a non-peak
        "relheight": crest_z - (toe_l + toe_r) / 2.0,
        "footprint_width": float(OFFSETS[ri] - OFFSETS[li]),
    }


def extract_asset(asset_id, tier, geom) -> list[dict]:
    line = longest_line(geom)
    if line is None or line.length < MIN_STATIONS * SPACING:
        return []
    d, pts, perps = stations(line)
    with rasterio.open(WINDOWS / f"{asset_id}.tif") as src:
        sections = sample_sections(src, pts, perps)
    out = []
    for k in range(len(d)):
        r = crest_and_toes(sections[k])
        if r is not None:
            out.append(
                {"asset_id": int(asset_id), "tier": tier, "chainage": float(d[k]), **r}
            )
    return out if len(out) >= MIN_STATIONS else []


def main():
    assets = load_windowed_assets()
    records: list[dict] = []
    for row in assets.itertuples(index=False):
        records.extend(
            extract_asset(
                row.asset_id, TIERS.get(row.asset_sub_type, "other"), row.geometry
            )
        )

    df = pl.DataFrame(records, schema=STATION_SCHEMA)

    assert df["asset_id"].is_in(assets["asset_id"].to_list()).all()
    assert df["crest_z"].is_finite().all(), "non-finite crest elevation leaked"
    assert df["toe_l_z"].is_finite().all() and df["toe_r_z"].is_finite().all()
    assert (df["footprint_width"] >= 0).all()

    df.write_parquet(OUT_PATH)
    print(
        f"wrote {df.height} stations for {df['asset_id'].n_unique()} assets -> {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
