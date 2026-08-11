# the halcrow deterioration curves from SC060078, as instantiated nationally in nb08.
#
# aims carries neither construction material nor maintenance regime, so both are defaulted:
# material by sub type (earth/clay core for embankments, concrete/masonry for walls, metal for
# gates), and ("medium", 2) for every asset. those defaults are the whole reason the heuristic
# scores near-random nationally - an engineer assessing a single asset would know both.
#
# differs from nb08 in one respect: nb08's curve join silently missed four sub types
# ("demountable" never matched the curve key "demountable_defence", and beach/dunes/weir were
# given material "na" where the tables use "shingle/sand"/""/"all"), so those assets were left
# unscored. the keys are corrected here and the curve tables normalised to match.

import polars as pl

from src.features import AGE_MAX, AGE_MIN

# aims__asset_sub_type -> (halcrow_type, material, width, forced environment)
# high ground has no curve of its own and rides the embankment one, as in nb08.
# the four single-environment curve families force their environment, otherwise a coastal weir or
# a fluvial beach falls through the join with no curve at all.
SUB_TYPES = {
    "Embankment": ("embankment", "varying_core", "narrow", None),
    "Engineered High Ground": ("embankment", "varying_core", "narrow", None),
    "Natural High Ground": ("embankment", "varying_core", "narrow", None),
    "Wall": ("vertical_wall", "concrete_masonry", "na", None),
    "Flood Gate": ("flood_gate", "metal", "na", None),
    "Demountable Defence": ("demountable_defence", "metal", "na", "fluvial"),
    "Beach": ("beach", "shingle/sand", "na", "coastal"),
    "Barrier Beach": ("beach", "shingle/sand", "na", "coastal"),
    "Dunes": ("dunes", "na", "na", "coastal"),
    "Weir": ("weir", "all", "na", "fluvial"),
}

ENVIRONMENT = {
    "Fluvial": "fluvial",
    "Coastal": "coastal",
    "Tidal": "coastal",
    "Fluvial/Tidal": "coastal",
}

DEFAULT_RATE, DEFAULT_REGIME = "medium", 2

# (halcrow_type, environment, material, width) -> {(rate, regime): [t1..t5]}
# t_n is the age in years at which the asset reaches condition grade n.
CURVES = {
    # embankment, varying core (pg. 75 fluvial, pg. 77 coastal), narrow
    ("embankment", "fluvial", "varying_core", "narrow"): {
        ("slowest", 1): [0, 5, 10, 40, 60],
        ("slowest", 2): [0, 20, 40, 70, 110],
        ("slowest", 3): [0, 22, 44, 90, 130],
        ("medium", 1): [0, 3, 6, 25, 40],
        ("medium", 2): [0, 15, 30, 60, 80],
        ("medium", 3): [0, 16, 33, 70, 90],
        ("fastest", 1): [0, 1, 3, 5, 7],
        ("fastest", 2): [0, 2, 5, 7, 10],
        ("fastest", 3): [0, 3, 6, 8, 11],
    },
    ("embankment", "coastal", "varying_core", "narrow"): {
        ("slowest", 1): [0, 5, 10, 40, 60],
        ("slowest", 2): [0, 20, 40, 60, 80],
        ("slowest", 3): [0, 22, 45, 80, 110],
        ("medium", 1): [0, 3, 6, 22, 30],
        ("medium", 2): [0, 14, 28, 40, 50],
        ("medium", 3): [0, 15, 30, 45, 60],
        ("fastest", 1): [0, 1, 2, 4, 5],
        ("fastest", 2): [0, 2, 4, 6, 8],
        ("fastest", 3): [0, 3, 5, 8, 10],
    },
    # vertical wall, concrete/brick-masonry (tbl 2.3)
    ("vertical_wall", "fluvial", "concrete_masonry", "na"): {
        ("slowest", 1): [0, 20, 50, 70, 80],
        ("slowest", 2): [0, 25, 60, 100, 120],
        ("slowest", 3): [0, 30, 70, 130, 160],
        ("medium", 1): [0, 15, 35, 50, 60],
        ("medium", 2): [0, 20, 45, 70, 90],
        ("medium", 3): [0, 25, 55, 90, 120],
        ("fastest", 1): [0, 5, 20, 30, 40],
        ("fastest", 2): [0, 10, 30, 50, 60],
        ("fastest", 3): [0, 15, 40, 70, 80],
    },
    ("vertical_wall", "coastal", "concrete_masonry", "na"): {
        ("slowest", 1): [0, 15, 45, 60, 80],
        ("slowest", 2): [0, 20, 60, 80, 100],
        ("slowest", 3): [0, 25, 75, 100, 120],
        ("medium", 1): [0, 10, 30, 40, 50],
        ("medium", 2): [0, 15, 40, 55, 70],
        ("medium", 3): [0, 20, 50, 70, 90],
        ("fastest", 1): [0, 5, 15, 25, 30],
        ("fastest", 2): [0, 10, 20, 30, 40],
        ("fastest", 3): [0, 15, 25, 35, 50],
    },
    # vertical wall, timber (pg. 48, 49) - unreachable from aims, no sub type defaults to timber
    ("vertical_wall", "fluvial", "timber", "na"): {
        ("slowest", 1): [0, 7, 15, 18, 21],
        ("slowest", 2): [0, 15, 30, 35, 40],
        ("slowest", 3): [0, 23, 45, 52, 60],
        ("medium", 1): [0, 5, 10, 12, 15],
        ("medium", 2): [0, 10, 20, 25, 30],
        ("medium", 3): [0, 15, 30, 35, 42],
        ("fastest", 1): [0, 3, 5, 7, 10],
        ("fastest", 2): [0, 5, 10, 12, 15],
        ("fastest", 3): [0, 7, 15, 17, 20],
    },
    ("vertical_wall", "coastal", "timber", "na"): {
        ("slowest", 1): [0, 5, 13, 16, 20],
        ("slowest", 2): [0, 14, 28, 33, 38],
        ("slowest", 3): [0, 21, 42, 48, 55],
        ("medium", 1): [0, 4, 8, 10, 14],
        ("medium", 2): [0, 8, 18, 23, 28],
        ("medium", 3): [0, 13, 28, 33, 38],
        ("fastest", 1): [0, 2, 4, 6, 8],
        ("fastest", 2): [0, 4, 8, 10, 13],
        ("fastest", 3): [0, 5, 13, 15, 18],
    },
    # flood gates and barriers, metal (pg. 196, 197)
    ("flood_gate", "fluvial", "metal", "na"): {
        ("slowest", 1): [0, 15, 32, 41, 50],
        ("slowest", 2): [0, 20, 40, 50, 60],
        ("slowest", 3): [0, 25, 48, 59, 70],
        ("medium", 1): [0, 12, 25, 32, 38],
        ("medium", 2): [0, 18, 34, 42, 50],
        ("medium", 3): [0, 24, 43, 52, 62],
        ("fastest", 1): [0, 5, 12, 16, 20],
        ("fastest", 2): [0, 10, 22, 30, 35],
        ("fastest", 3): [0, 15, 32, 44, 50],
    },
    ("flood_gate", "coastal", "metal", "na"): {
        ("slowest", 1): [0, 13, 22, 26, 30],
        ("slowest", 2): [0, 18, 29, 35, 40],
        ("slowest", 3): [0, 23, 36, 44, 50],
        ("medium", 1): [0, 10, 14, 16, 18],
        ("medium", 2): [0, 15, 23, 27, 30],
        ("medium", 3): [0, 20, 32, 38, 42],
        ("fastest", 1): [0, 4, 7, 9, 10],
        ("fastest", 2): [0, 7, 11, 13, 15],
        ("fastest", 3): [0, 10, 15, 17, 20],
    },
    # demountable defence (tbl C.4 pg. 67, metal/wood)
    ("demountable_defence", "fluvial", "metal", "na"): {
        ("slowest", 1): [0, 2, 4, 5, 7],
        ("slowest", 2): [0, 10, 20, 60, 70],
        ("slowest", 3): [0, 15, 25, 70, 80],
        ("medium", 1): [0, 1, 3, 4, 5],
        ("medium", 2): [0, 5, 10, 45, 55],
        ("medium", 3): [0, 8, 15, 55, 65],
        ("fastest", 1): [0, 1, 2, 3, 4],
        ("fastest", 2): [0, 2, 5, 35, 45],
        ("fastest", 3): [0, 5, 10, 45, 55],
    },
    ("demountable_defence", "fluvial", "wood", "na"): {
        ("slowest", 1): [0, 2, 4, 5, 7],
        ("slowest", 2): [0, 5, 10, 30, 35],
        ("slowest", 3): [0, 8, 13, 35, 40],
        ("medium", 1): [0, 1, 3, 4, 5],
        ("medium", 2): [0, 3, 5, 23, 28],
        ("medium", 3): [0, 4, 8, 28, 33],
        ("fastest", 1): [0, 1, 2, 3, 4],
        ("fastest", 2): [0, 1, 3, 18, 23],
        ("fastest", 3): [0, 3, 5, 23, 28],
    },
    # beach (tbl C.7 pg. 130)
    ("beach", "coastal", "shingle/sand", "na"): {
        ("slowest", 1): [0, 15, 38, 75, 100],
        ("slowest", 2): [0, 27, 50, 150, 200],
        ("slowest", 3): [0, 27, 75, 200, 250],
        ("medium", 1): [0, 9, 13, 25, 35],
        ("medium", 2): [0, 16, 30, 50, 75],
        ("medium", 3): [0, 20, 55, 90, 120],
        ("fastest", 1): [0, 4, 7, 9, 13],
        ("fastest", 2): [0, 7, 10, 13, 20],
        ("fastest", 3): [0, 12, 20, 25, 40],
    },
    # dunes (tbl C.15 pg. 152)
    ("dunes", "coastal", "na", "na"): {
        ("slowest", 1): [0, 20, 40, 110, 150],
        ("slowest", 2): [0, 27, 60, 150, 200],
        ("slowest", 3): [0, 30, 80, 190, 250],
        ("medium", 1): [0, 10, 15, 30, 40],
        ("medium", 2): [0, 15, 35, 60, 80],
        ("medium", 3): [0, 20, 60, 100, 130],
        ("fastest", 1): [0, 5, 8, 10, 15],
        ("fastest", 2): [0, 7, 10, 13, 20],
        ("fastest", 3): [0, 12, 20, 25, 40],
    },
    # weir (tbl C.18 pg. 167)
    ("weir", "fluvial", "all", "na"): {
        ("slowest", 1): [0, 20, 30, 50, 70],
        ("slowest", 2): [0, 40, 70, 90, 110],
        ("slowest", 3): [0, 60, 110, 130, 150],
        ("medium", 1): [0, 15, 20, 40, 60],
        ("medium", 2): [0, 30, 50, 70, 90],
        ("medium", 3): [0, 45, 80, 100, 120],
        ("fastest", 1): [0, 10, 15, 30, 40],
        ("fastest", 2): [0, 20, 30, 50, 60],
        ("fastest", 3): [0, 30, 45, 70, 80],
    },
}


def curve_table() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "halcrow_type": halcrow_type,
                "environment": environment,
                "material": material,
                "width": width,
                "rate": rate,
                "regime": regime,
                **{f"t{i + 1}": float(t) for i, t in enumerate(thresholds)},
            }
            for (halcrow_type, environment, material, width), grid in CURVES.items()
            for (rate, regime), thresholds in grid.items()
        ]
    )


def _map_sub_type(position: int) -> pl.Expr:
    mapping = {k: v[position] for k, v in SUB_TYPES.items()}
    return pl.col("aims__asset_sub_type").replace_strict(
        mapping, default=None, return_dtype=pl.String
    )


def _fraction(low: str, high: str) -> pl.Expr:
    """position between two curve thresholds, 0 where the band has no width"""
    low_e, high_e = pl.col(low), pl.col(high)
    return (
        pl.when(high_e > low_e)
        .then((pl.col("age_years") - low_e) / (high_e - low_e))
        .otherwise(0.0)
    )


def score(
    df: pl.DataFrame, rate: str = DEFAULT_RATE, regime: int = DEFAULT_REGIME
) -> pl.DataFrame:
    """halcrow curve prediction per asset. pred_grade is null unless curve_status == 'scored'."""

    curves = curve_table().filter(
        pl.col("rate").eq(rate) & pl.col("regime").eq(regime)
    ).drop("rate", "regime")
    if curves.is_empty():
        raise ValueError(f"no curves for rate={rate!r}, regime={regime!r}")

    age_years = (
        pl.col("eir__inspection_date").cast(pl.Date) - pl.col("aims__asset_start_date")
    ).dt.total_days() / 365.25

    scoped = df.select(
        pl.col("aims__asset_id").alias("asset_id"),
        pl.col("aims__asset_sub_type").alias("sub_type"),
        _map_sub_type(0).alias("halcrow_type"),
        _map_sub_type(1).alias("material"),
        _map_sub_type(2).alias("width"),
        pl.coalesce(
            _map_sub_type(3),
            pl.col("aims__protection_type").replace_strict(
                ENVIRONMENT, default=None, return_dtype=pl.String
            ),
            pl.lit("fluvial"),
        ).alias("environment"),
        # same window as prepare_features, so heuristic age and model age_years cannot diverge
        pl.when(age_years.is_between(AGE_MIN, AGE_MAX))
        .then(age_years)
        .otherwise(None)
        .alias("age_years"),
    )

    pred_grade_cont = (
        pl.when(pl.col("age_years").ge(pl.col("t5")))
        .then(5.0)
        .when(pl.col("age_years").ge(pl.col("t4")))
        .then(4.0 + _fraction("t4", "t5"))
        .when(pl.col("age_years").ge(pl.col("t3")))
        .then(3.0 + _fraction("t3", "t4"))
        .when(pl.col("age_years").ge(pl.col("t2")))
        .then(2.0 + _fraction("t2", "t3"))
        .otherwise(1.0 + _fraction("t1", "t2"))
        .clip(1.0, 5.0)
    )

    return (
        scoped.join(
            curves,
            on=["halcrow_type", "environment", "material", "width"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.when(pl.col("halcrow_type").is_null())
            .then(pl.lit("unmapped_type"))
            .when(pl.col("t2").is_null())
            .then(pl.lit("no_curve"))
            .when(pl.col("age_years").is_null())
            .then(pl.lit("no_age"))
            .otherwise(pl.lit("scored"))
            .alias("curve_status")
        )
        .with_columns(
            pl.when(pl.col("curve_status").eq("scored"))
            .then(pred_grade_cont)
            .alias("pred_grade_cont")
        )
        .select(
            "asset_id",
            "sub_type",
            "curve_status",
            "age_years",
            "pred_grade_cont",
            pl.col("pred_grade_cont").floor().clip(1, 5).cast(pl.Int8).alias("pred_grade"),
        )
    )
