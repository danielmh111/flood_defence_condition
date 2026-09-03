# two compositions, carried out of the fine grid as the surviving candidates: climate_raw leads on
# the lift curve at the operational budget, lidar_full leads on whole-curve pr-auc, and the
# within-lidar ordering between them is at the noise floor. neither is a demonstrated winner.
#
# the column lists and their order are byte-identical to the fine grid's same-named arms. that is
# what lets the reference config hash to the same cell_id across the two experiments and reproduce
# its oof bit for bit - see models.REFERENCE_CONFIG.
#
# neither arm contains bedrock, which is what makes max_bins testable below 32 for the first time
# (bedrock_lex_rcs_binned has 21 levels and sklearn needs max_bins >= every categorical's
# cardinality). that exclusion is asserted here rather than assumed.

import numpy as np
import polars as pl

from src.features import create_arrays

IDENTITY_ENUMS = [
    "aims__asset_sub_type",
    "aims__protection_type",
    "aims__primary_purpose",
]
IDENTITY_NUMERIC = ["asset_length_log1p", "actual_dcl", "design_sop", "age_years"]

LIDAR_FULL = [
    "lidar__crest_prominence_median",
    "lidar__crest_resid_tv_range",
    "lidar__crest_resid_max_dip",
    "lidar__crest_resid_std",
]
LIDAR_NO_TV = [
    "lidar__crest_prominence_median",
    "lidar__crest_resid_max_dip",
    "lidar__crest_resid_std",
]
CLIMATE_RAW = ["climate__drydays__30y__rate", "climate__ftc__30y__rate"]

COMPOSITIONS = {
    "identity_lidar_climate_raw": [
        *IDENTITY_ENUMS,
        *IDENTITY_NUMERIC,
        *LIDAR_NO_TV,
        *CLIMATE_RAW,
    ],
    "identity_lidar_full": [*IDENTITY_ENUMS, *IDENTITY_NUMERIC, *LIDAR_FULL],
}

# only what these two arms need. the offset, z and bedrock blocks the fine grid joined are gone.
CLIMATE_BLOCK = CLIMATE_RAW
LIDAR_BLOCK = LIDAR_FULL

EXCLUDED_PREFIXES = ("bedrock_", "superficial_geo__", "geosure_")

DEAD_COLS = [
    "maintainer_is_ea",
    "age_estimated",
    "design_sop_missing",
    "actual_dcl_missing",
]


def attach_blocks(df_feats: pl.DataFrame, climate_parquet, lidar_parquet):
    """left-join the climate and lidar blocks. nulls are never filled - HGB handles them, and
    filling would invent geometry that was never measured."""
    asset_ids = df_feats["asset_id"]

    for parquet, cols in ((climate_parquet, CLIMATE_BLOCK), (lidar_parquet, LIDAR_BLOCK)):
        block = pl.read_parquet(parquet).select("asset_id", *cols)
        df_feats = df_feats.join(
            block, on="asset_id", how="left", validate="1:1", maintain_order="left"
        )

    if not df_feats["asset_id"].equals(asset_ids):
        raise ValueError("block join changed row count or asset_id order")

    coverage = {
        col: float(df_feats[col].is_not_null().mean())
        for col in (*CLIMATE_BLOCK, *LIDAR_BLOCK)
    }
    return df_feats, coverage


def check_compositions(df_joined: pl.DataFrame, max_bins_levels) -> dict[str, int]:
    """everything that can fail before the first fit, failing in the first second rather than hour
    two: columns resolve, no dead or geological column sneaks in, and the coarsest max_bins level
    clears every categorical's cardinality. returns those cardinalities for the run record."""
    for name, cols in COMPOSITIONS.items():
        missing = [c for c in cols if c not in df_joined.columns]
        if missing:
            raise ValueError(f"{name}: missing column(s) {missing}")
        dead = [c for c in DEAD_COLS if c in cols]
        if dead:
            raise ValueError(f"{name}: dead column(s) {dead} must not be in a composition")
        excluded = [c for c in cols if c.startswith(EXCLUDED_PREFIXES)]
        if excluded:
            raise ValueError(
                f"{name}: geological column(s) {excluded} - the sub-32 max_bins range is only "
                "reachable because neither arm carries one"
            )

    schema = df_joined.schema
    cardinality = {
        col: len(schema[col].categories)
        for col in set().union(*COMPOSITIONS.values())
        if schema[col] == pl.Enum
    }

    coarsest = min(max_bins_levels)
    over = {col: n for col, n in cardinality.items() if n > coarsest}
    if over:
        raise ValueError(
            f"max_bins {coarsest} is below categorical cardinality {over} - per spec 4.3 the level "
            "set falls back to [32, 64, 255] at 17-32 and [32, 255] above"
        )

    return cardinality


def build_matrix(df_joined: pl.DataFrame, name: str):
    """enums are sorted to the front here rather than in COMPOSITIONS, so a composition can be
    declared in whatever order reads best and create_arrays still yields a contiguous cat_idx."""
    cols = COMPOSITIONS[name]
    schema = df_joined.schema

    enums = [c for c in cols if schema[c] == pl.Enum]
    ordered = [*enums, *(c for c in cols if schema[c] != pl.Enum)]

    X, y, cat_idx, asset_ids = create_arrays(
        df_joined.select("asset_id", "condition_grade", *ordered)
    )

    if X.shape[1] != len(cols):
        raise ValueError(f"{name}: width {X.shape[1]} != expected {len(cols)}")
    if cat_idx != list(range(len(enums))):
        raise ValueError(
            f"{name}: cat_idx {cat_idx} is not the leading {len(enums)} columns - "
            "column order or dtypes have drifted"
        )

    return X, y, cat_idx, asset_ids, ordered


def build_all(df_joined: pl.DataFrame):
    """both compositions on one shared population, gated on identical asset_ids and y"""
    matrices, names, cat_idx_by_composition = {}, {}, {}
    y = asset_ids = None

    for name in COMPOSITIONS:
        X, y_i, cat_idx, ids_i, cols = build_matrix(df_joined, name)
        if y is None:
            y, asset_ids = y_i, ids_i
        elif not np.array_equal(ids_i, asset_ids) or not np.array_equal(y_i, y):
            raise ValueError(f"{name}: row order or target drifted between compositions")
        matrices[name], names[name], cat_idx_by_composition[name] = X, cols, cat_idx

    return matrices, names, cat_idx_by_composition, y, asset_ids
