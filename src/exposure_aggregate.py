from datetime import date, timedelta
from typing import Literal

import polars as pl
from project_paths import paths

from src.features import prepare_features

RECORD_ORIGIN = date(1946, 1, 1)  # first date in recorded flood outlines data
SOURCES = ["fluvial", "coastal", "tidal", "surface_water"]

# substring parse of aims__protection_type -> asset source flags.
# 'Fluvial/Tidal' sets both; no value accidentally matches another.
PROTECTION_TYPE_PATTERNS = {
    "fluvial": "Fluvial",
    "coastal": "Coastal",
    "tidal": "Tidal",
    "surface_water": "Surface Water",
}

WINDOWS: list[timedelta | None] = [
    timedelta(days=365.25 * 2),
    timedelta(days=365.25 * 5),
    timedelta(days=365.25 * 15),
    None,  # lifetime
]

STATS = ["n_events", "freq", "cum_area", "max_event_area"]


def window_label(window: timedelta | None) -> str:
    return "lifetime" if window is None else f"{round(window.days / 365.25)}y"


def load_anchors() -> pl.DataFrame:

    unified = pl.read_parquet(paths.unified_file)
    scope = prepare_features(unified)["asset_id"]

    unified = unified.filter(pl.col("aims__asset_id").is_in(scope.to_list())).select(
        pl.col("aims__asset_id").alias("asset_id"),
        pl.col("eir__inspection_date").cast(pl.Date).alias("t_anchor"),
        pl.col("aims__asset_start_date").alias("asset_start_date"),
        *(
            pl.col("aims__protection_type")
            .str.contains(pattern, literal=True)
            .fill_null(False)
            .alias(f"asset_src_{source}")
            for source, pattern in PROTECTION_TYPE_PATTERNS.items()
        ),
        pl.col("aims__protection_type").is_null().alias("protection_type_missing"),
    )

    return unified


def exposure_features(
    events: pl.DataFrame,
    anchors: pl.DataFrame,
    window: timedelta | None,
    subset: Literal["all", "matched"],
) -> pl.DataFrame:
    """asset_id + exp__{subset}__{window}__{stat} columns, one row per anchor asset

    zero events maps to 0, null t_anchor or null protection type maps to null
    """

    label = window_label(window)
    prefix = f"exp__{subset}__{label}"

    ev = events.join(anchors, on="asset_id", how="inner")

    # temporal filter - whole event completed before the inspection and started inside the lookback
    # null dates fail and drop out
    ev = ev.filter(pl.col("end_date") < pl.col("t_anchor"))
    if window is not None:
        ev = ev.filter(
            pl.col("start_date") >= pl.col("t_anchor") - pl.duration(days=window.days)
        )

    if subset == "matched":
        ev = ev.filter(
            pl.any_horizontal(
                pl.col(f"src_{s}") & pl.col(f"asset_src_{s}") for s in SOURCES
            )
        )

    # event grain is rec_grp_id - one flood = one group of outlines
    # outlines with no group id count as single events so they dont get grouped under a "null" event
    per_event = (
        ev.with_columns(
            pl.coalesce("rec_grp_id", pl.format("outline:{}", "rec_out_id"))
        )
        .group_by("asset_id", "rec_grp_id")
        .agg(pl.col("intersected_area_m2").sum().alias("event_area"))
    )
    stats = per_event.group_by("asset_id").agg(
        pl.len().cast(pl.Int64).alias("n_events"),
        pl.col("event_area").sum().alias("cum_area"),
        pl.col("event_area").max().alias("max_event_area"),
    )

    # observable_years - how long this asset could have accrued recorded events within the window
    window_start = (
        [pl.col("t_anchor") - pl.duration(days=window.days)] if window else []
    )
    observable_floor = pl.max_horizontal(
        *window_start,
        "asset_start_date",
        pl.lit(RECORD_ORIGIN),
    )
    observable_years = (pl.col("t_anchor") - observable_floor).dt.total_days() / 365.25

    unassessable = pl.col("t_anchor").is_null() | (
        pl.lit(subset == "matched") & pl.col("protection_type_missing")
    )

    out = (
        anchors.join(stats, on="asset_id", how="left")
        .with_columns(
            pl.col("n_events", "cum_area", "max_event_area").fill_null(0),
            observable_years.alias("observable_years"),
        )
        .with_columns(
            # freq null (not inf/negative) when observable_years <= 0
            pl.when(pl.col("observable_years") > 0)
            .then(pl.col("n_events") / pl.col("observable_years"))
            .alias("freq"),
        )
        .with_columns(
            pl.when(unassessable).then(None).otherwise(pl.col(stat)).alias(stat)
            for stat in STATS
        )
    )

    named = out.select(
        "asset_id", *(pl.col(stat).alias(f"{prefix}__{stat}") for stat in STATS)
    )

    # emit once with the 'all' pass per window
    if subset == "all" and window is not None:
        named = named.with_columns(
            out.select(
                (pl.col("observable_years") < window.days / 365.25).alias(
                    f"exp__{label}__window_truncated"
                )
            )
        )

    return named


def build_exposure_features(
    events: pl.DataFrame, anchors: pl.DataFrame
) -> pl.DataFrame:
    """full {all, matched} x {2y, 5y, 15y, lifetime} grid, one row per anchor asset."""
    out = anchors.select("asset_id")
    for subset in ("all", "matched"):
        for window in WINDOWS:
            out = out.join(
                exposure_features(events, anchors, window, subset),
                on="asset_id",
                how="left",
            )
    return out


def main():
    events = pl.read_parquet(paths.processed_data / "exposure_events.parquet")
    anchors = load_anchors()

    # check data input before running

    for s in SOURCES:
        if not events[f"src_{s}"].any():
            raise ValueError(f"src_{s} all false, flag encoding could be broken")

    if not events.select("asset_id", "rec_out_id").is_unique().all():
        raise ValueError("duplicates on asset id and outline id present")

    if not events["asset_id"].is_in(anchors["asset_id"].to_list()).all():
        raise ValueError(
            "assets in the exposure events present not in the scope of other extracted features"
        )

    n_undated = events.filter(
        pl.col("start_date").is_null() | pl.col("end_date").is_null()
    ).height
    print(f"{n_undated} undated event rows (dropped)")

    features = build_exposure_features(events, anchors)

    # check outputs

    assert features["asset_id"].sort().equals(anchors["asset_id"].sort())

    stat_cols = [c for c in features.columns if not c.endswith("__window_truncated")]
    n_null_anchor = anchors["t_anchor"].null_count()
    n_unassessable_matched = anchors.filter(
        pl.col("t_anchor").is_null() | pl.col("protection_type_missing")
    ).height
    for col in stat_cols:
        if col == "asset_id":
            continue
        vals = features[col].drop_nulls()
        assert vals.is_finite().all(), f"{col}: non-finite values"
        assert (vals >= 0).all(), f"{col}: negative values"
        expected_nulls = (
            n_unassessable_matched if "__matched__" in col else n_null_anchor
        )
        assert features[col].null_count() >= expected_nulls, f"{col}: nulls missing"

    out_path = paths.processed_data / "exposure_features.parquet"
    features.write_parquet(out_path)
    print(
        f"wrote {features.height} assets x {features.width - 1} feature columns to {out_path}"
    )


if __name__ == "__main__":
    main()
