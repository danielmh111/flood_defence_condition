# fourteen compositions, each a one-factor move from identity or identity_lidar_no_tv (spec 4.2).
# this is column-granular where the predecessor experiment was block-granular, so a composition is
# an explicit ordered column list rather than a set of extras layered onto an implied identity.
#
# lidar_no_tv is the pre-committed lidar representative in every combination arm: tv_range ranked
# last at grade 4 in the predecessor's shap and covers 24.6% against resid_std's 49.75%. reducing
# the lidar list also raises coverage, so arms 2->3->4->5 confound width with support - not
# fixable, separated at analysis time with a coverage-stratified diagnostic.
#
# the four dead columns are still rejected everywhere: all had mean_abs_shap of exactly 0.0,
# because HGB splits on NaN natively and an explicit missingness indicator is redundant.

import numpy as np
import polars as pl

from src.features import create_arrays

IDENTITY = [
    "aims__asset_sub_type",
    "aims__protection_type",
    "aims__primary_purpose",
    "asset_length_log1p",
    "actual_dcl",
    "design_sop",
    "age_years",
]

BEDROCK = ["bedrock_lex_rcs_binned"]

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
LIDAR_CORE = ["lidar__crest_prominence_median", "lidar__crest_resid_std"]
LIDAR_OFFSET = ["lidar__crest_offset_median"]

CLIMATE_FULL = [
    "climate__drydays__30y__rate",
    "climate__drydays__30y__z",
    "climate__ftc__30y__rate",
    "climate__ftc__30y__z",
]
CLIMATE_RAW = ["climate__drydays__30y__rate", "climate__ftc__30y__rate"]
CLIMATE_Z = ["climate__drydays__30y__z", "climate__ftc__30y__z"]

COMPOSITIONS = {
    "identity": IDENTITY,
    "identity_lidar_full": [*IDENTITY, *LIDAR_FULL],
    "identity_lidar_no_tv": [*IDENTITY, *LIDAR_NO_TV],
    "identity_lidar_core": [*IDENTITY, *LIDAR_CORE],
    "identity_resid_std": [*IDENTITY, "lidar__crest_resid_std"],
    "identity_climate_full": [*IDENTITY, *CLIMATE_FULL],
    "identity_climate_raw": [*IDENTITY, *CLIMATE_RAW],
    "identity_climate_z": [*IDENTITY, *CLIMATE_Z],
    "identity_bedrock": [*IDENTITY, *BEDROCK],
    "identity_lidar_climate_raw": [*IDENTITY, *LIDAR_NO_TV, *CLIMATE_RAW],
    "identity_lidar_climate_z": [*IDENTITY, *LIDAR_NO_TV, *CLIMATE_Z],
    "identity_lidar_bedrock": [*IDENTITY, *LIDAR_NO_TV, *BEDROCK],
    "identity_lidar_climate_bedrock": [*IDENTITY, *LIDAR_NO_TV, *CLIMATE_RAW, *BEDROCK],
    "identity_lidar_offset": [*IDENTITY, *LIDAR_NO_TV, *LIDAR_OFFSET],
}

# every block column any composition can ask for. crest_offset_median is new here and its
# coverage is unknown, which materially affects how identity_lidar_offset reads (spec 3).
CLIMATE_BLOCK = CLIMATE_FULL
LIDAR_BLOCK = [*LIDAR_FULL, *LIDAR_OFFSET]

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


def check_compositions(df_joined: pl.DataFrame, max_bins: int) -> None:
    """every arm's columns resolve and every enum fits in max_bins, checked before the first fit
    so a typo fails in the first second rather than hour four."""
    for name, cols in COMPOSITIONS.items():
        missing = [c for c in cols if c not in df_joined.columns]
        if missing:
            raise ValueError(f"{name}: missing column(s) {missing}")
        dead = [c for c in DEAD_COLS if c in cols]
        if dead:
            raise ValueError(f"{name}: dead column(s) {dead} must not be in a composition")

    schema = df_joined.schema
    cardinality = {
        col: len(schema[col].categories)
        for col in set().union(*COMPOSITIONS.values())
        if schema[col] == pl.Enum
    }
    over = {col: n for col, n in cardinality.items() if n > max_bins}
    if over:
        raise ValueError(f"max_bins {max_bins} is below categorical cardinality {over}")


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
    """every composition on one shared population, gated on identical asset_ids and y"""
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
