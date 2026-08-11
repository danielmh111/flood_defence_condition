# the five named feature sets. `base` from oof_grid is deliberately not reused as a name - there
# it included geology, so the token would mean two different things across the two experiments.
#
# the four dead columns from nb16 are dropped everywhere: all had mean_abs_shap of exactly 0.0,
# because HGB splits on NaN natively and an explicit missingness indicator is redundant with the
# split it already encodes.

import numpy as np
import polars as pl

from src.features import GEOSURE_COLS, create_arrays

IDENTITY_ENUMS = [
    "aims__asset_sub_type",
    "aims__protection_type",
    "aims__primary_purpose",
]

IDENTITY_NUMERIC = [
    "asset_length_log1p",
    "actual_dcl",
    "design_sop",
    "age_years",
]

GEOLOGY_ENUMS = ["bedrock_lex_rcs_binned"]

# one moisture axis and one temperature axis, each raw and de-meaned (nb17). the raw/z pairing
# tests whether a metric helps only through its spatial mean or retains anything after de-meaning.
CLIMATE = [
    "climate__drydays__30y__rate",
    "climate__drydays__30y__z",
    "climate__ftc__30y__rate",
    "climate__ftc__30y__z",
]

# nb18's closing set, carrying the two "maybe" columns because interactions were untested
LIDAR = [
    "lidar__crest_prominence_median",
    "lidar__crest_resid_tv_range",
    "lidar__crest_resid_max_dip",
    "lidar__crest_resid_std",
]

# name -> (extra enums, extra numerics, expected width)
FEATURE_SETS = {
    "identity": ([], [], 7),
    "identity_geo": (GEOLOGY_ENUMS, GEOSURE_COLS, 14),
    "identity_climate": ([], CLIMATE, 11),
    "identity_lidar": ([], LIDAR, 11),
    "identity_all": (GEOLOGY_ENUMS, [*GEOSURE_COLS, *CLIMATE, *LIDAR], 22),
}

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

    for parquet, cols in ((climate_parquet, CLIMATE), (lidar_parquet, LIDAR)):
        block = pl.read_parquet(parquet).select("asset_id", *cols)
        df_feats = df_feats.join(
            block, on="asset_id", how="left", validate="1:1", maintain_order="left"
        )

    if not df_feats["asset_id"].equals(asset_ids):
        raise ValueError("block join changed row count or asset_id order")

    # lidar support is structurally confounded with tier (nb18), so the analysis needs these
    coverage = {
        col: float(df_feats[col].is_not_null().mean()) for col in (*CLIMATE, *LIDAR)
    }
    return df_feats, coverage


def build_matrix(df_joined: pl.DataFrame, name: str):
    extra_enums, extra_numeric, expected_width = FEATURE_SETS[name]

    # enums lead so create_arrays derives a contiguous cat_idx from position 0
    cols = [*IDENTITY_ENUMS, *extra_enums, *IDENTITY_NUMERIC, *extra_numeric]

    missing = [c for c in cols if c not in df_joined.columns]
    if missing:
        raise ValueError(f"{name}: missing column(s) {missing}")
    dead = [c for c in DEAD_COLS if c in cols]
    if dead:
        raise ValueError(f"{name}: dead column(s) {dead} must not be in a feature set")

    X, y, cat_idx, asset_ids = create_arrays(
        df_joined.select("asset_id", "condition_grade", *cols)
    )

    if X.shape[1] != expected_width:
        raise ValueError(f"{name}: width {X.shape[1]} != expected {expected_width}")

    # cat_idx is per feature set, not constant - bedrock_lex_rcs_binned is only in the two geology
    # sets, so the other three have three categoricals. oof_grid's hardcoded [0,1,2,3] would
    # mistype a numeric column here.
    n_enums = len(IDENTITY_ENUMS) + len(extra_enums)
    if cat_idx != list(range(n_enums)):
        raise ValueError(
            f"{name}: cat_idx {cat_idx} is not the leading {n_enums} columns - "
            "column order or dtypes have drifted"
        )

    return X, y, cat_idx, asset_ids, cols


def build_all(df_joined: pl.DataFrame):
    """every feature set on one shared population, gated on identical asset_ids and y"""
    matrices, names, cat_idx_by_set = {}, {}, {}
    y = asset_ids = None

    for name in FEATURE_SETS:
        X, y_i, cat_idx, ids_i, cols = build_matrix(df_joined, name)
        if y is None:
            y, asset_ids = y_i, ids_i
        elif not np.array_equal(ids_i, asset_ids) or not np.array_equal(y_i, y):
            raise ValueError(f"{name}: row order or target drifted between feature sets")
        matrices[name], names[name], cat_idx_by_set[name] = X, cols, cat_idx

    return matrices, names, cat_idx_by_set, y, asset_ids
