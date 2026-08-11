import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import numpy as np
import polars as pl
from project_paths import paths
from sklearn.model_selection import StratifiedGroupKFold

from experiments.oof_grid.models import FEATURE_SETS, MODELS, build_grid
from src.cross_validate import (
    GRADES,
    fold_ids_from_splitter,
    produce_spatial_blocks,
    run_oof,
)
from src.features import GEOSURE_COLS, LEAKY_COLS, create_arrays, prepare_features
from src.oof_shap import make_tree_shap_fn

SEED = 123
N_BLOCKS = 50
N_SPLITS = 10

EXP_COLS = ["exp__all__2y__cum_area", "exp__all__2y__max_event_area"]
TRUNC_COL = "exp__2y__window_truncated"
PROTECTION_TYPE_NULL_COUNT = 285  # known number, found in nb14 and baseline.py

FULL_COVERAGE_COLS = [
    "aims__asset_sub_type",
    "aims__primary_purpose",
    *GEOSURE_COLS,
    "maintainer_is_ea",
    "asset_length_log1p",
]

PARQUET = paths.processed_data / "unified_aims_eir_bgs.parquet"
GPKG = paths.aims_data / "aims.gpkg"
EXPOSURE_PARQUET = paths.processed_data / "exposure_features.parquet"
OUT = paths.experiments / "oof_grid"


def feature_names(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("asset_id", "condition_grade")]


def load_coords(asset_ids: np.ndarray, gpkg_path: Path) -> np.ndarray:
    gdf = gpd.read_file(gpkg_path)
    cent = gdf.geometry.centroid
    coords = pl.DataFrame(
        {
            "asset_id": gdf["asset_id"].astype("int64").to_numpy(),
            "x": cent.x.to_numpy(),
            "y": cent.y.to_numpy(),
        }
    )
    joined = pl.DataFrame({"asset_id": asset_ids}).join(
        coords, on="asset_id", how="left", validate="1:1", maintain_order="left"
    )
    if joined["x"].null_count() > 0:
        raise ValueError("asset(s) with no centroid in aims.gpkg")
    if not np.array_equal(joined["asset_id"].to_numpy(), asset_ids):
        raise ValueError("coords join did not preserve asset_id row order")
    return joined.select("x", "y").to_numpy()


def load_set(parquet: Path) -> tuple:
    df_feats = prepare_features(pl.read_parquet(parquet))

    leaky_hit = set(LEAKY_COLS) & set(df_feats.columns)
    if leaky_hit:
        raise ValueError(f"G1 failed: leaky columns present: {sorted(leaky_hit)}")

    if df_feats["asset_id"].n_unique() != df_feats.height:
        raise ValueError("asset_id is not unique on the set")
    if not df_feats["condition_grade"].is_between(1, 5).all():
        raise ValueError("condition_grade outside 1..5")
    if not df_feats["asset_length_log1p"].is_finite().all():
        raise ValueError("asset_length_log1p has non-finite values")

    full_coverage_nulls = int(
        df_feats.select(FULL_COVERAGE_COLS).null_count().to_numpy().sum()
    )
    if full_coverage_nulls:
        raise ValueError(
            f"G6 failed: {full_coverage_nulls} nulls across nominally full-coverage columns "
            f"{FULL_COVERAGE_COLS}"
        )
    protection_type_nulls = int(df_feats["aims__protection_type"].null_count())
    if protection_type_nulls != PROTECTION_TYPE_NULL_COUNT:
        raise ValueError(
            f"G6 failed: aims__protection_type null count drifted "
            f"({protection_type_nulls} != known {PROTECTION_TYPE_NULL_COUNT}) - "
            "investigate before trusting this run (see plan L2)"
        )

    X_base, y, cat_idx, asset_ids = create_arrays(df_feats)
    names = feature_names(df_feats)
    if len(names) != X_base.shape[1]:
        raise ValueError(
            f"feature_names length {len(names)} != X_base width {X_base.shape[1]}"
        )

    bedrock_categories = df_feats.schema["bedrock_lex_rcs_binned"].categories.to_list()

    return (
        df_feats,
        X_base,
        y,
        cat_idx,
        asset_ids,
        names,
        protection_type_nulls,
        bedrock_categories,
    )


def main(
    parquet: Path = PARQUET,
    gpkg: Path = GPKG,
    exposure_parquet: Path = EXPOSURE_PARQUET,
    n_blocks: int = N_BLOCKS,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    out_dir: Path = OUT,
    smoke: bool = False,
) -> None:
    if smoke:
        n_splits = min(n_splits, 2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load set of assets
    (
        df_feats,
        X_base,
        y,
        cat_idx,
        asset_ids,
        names_base,
        protection_type_nulls,
        bedrock_categories,
    ) = load_set(parquet)
    n_assets = int(X_base.shape[0])

    # get centroid coords of each asset
    coords = load_coords(asset_ids, gpkg)

    # split the folds that the experiment will run. they need to be defined before the loop and fed in so that metrics and results from within each fold can be tracked

    blocks = produce_spatial_blocks(coords, n_blocks=n_blocks, seed=seed)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_id = fold_ids_from_splitter(splitter, X_base, y, groups=blocks)

    folds_df = pl.DataFrame(
        {
            "asset_id": asset_ids.astype(np.int64),
            "block_id": blocks.astype(np.int16),
            "fold_id": fold_id,
        }
    )
    folds_df.write_parquet(out_dir / "folds.parquet")

    # exposure feature set - left join, keep nulls, append exposure columns

    exposure = pl.read_parquet(exposure_parquet).select(
        "asset_id", *EXP_COLS, TRUNC_COL
    )
    joined = pl.DataFrame({"asset_id": asset_ids.astype(np.int64)}).join(
        exposure, on="asset_id", how="left", validate="1:1", maintain_order="left"
    )
    if joined.height != X_base.shape[0]:
        raise ValueError(
            f"G4 failed: exposure left-join changed row count "
            f"({joined.height} != {X_base.shape[0]})"
        )
    if not np.array_equal(joined["asset_id"].to_numpy(), asset_ids):
        raise ValueError("exposure join did not preserve asset_id row order")

    exp_matrix = joined.select(pl.col(c).cast(pl.Float64) for c in EXP_COLS).to_numpy()
    X_base_exp = np.column_stack([X_base, exp_matrix])
    names_base_exp = [*names_base, *EXP_COLS]

    if not np.array_equal(X_base_exp[:, : X_base.shape[1]], X_base, equal_nan=True):
        raise ValueError(
            "H3 failed: X_base_exp does not extend X_base unchanged - cat_idx invalid"
        )

    n_known_nulls = int(np.isnan(X_base_exp[:, -2]).sum())  # exp__all__2y__cum_area
    if n_known_nulls != 1:
        raise ValueError(
            f"exp__all__2y__cum_area null count drifted ({n_known_nulls} != known 1) - "
            "investigate before trusting this run"
        )

    exp_coverage_rate = float(1.0 - np.isnan(exp_matrix).mean())
    trunc_rate = float(joined[TRUNC_COL].fill_null(False).mean())  # type: ignore

    # define experiment grid and loop
    feature_sets = {
        "base": (X_base, names_base),
        "base_exp": (X_base_exp, names_base_exp),
    }
    grid = build_grid(cat_idx, seed)

    oof_frames: list[pl.DataFrame] = []
    fh_inversion_rates: dict[str, float] = {}

    for spec in grid:
        X, names = feature_sets[spec.feature_set]
        shap_fn = None
        if spec.wants_shap:
            shap_fn = make_tree_shap_fn(names, max_rows_per_fold=50 if smoke else None)

        oof_df, shap_df = run_oof(
            spec.make_model,
            X,
            y,
            fold_id,
            asset_ids,
            fit_fn=spec.fit_fn,
            predict_fn=spec.predict_fn,
            shap_fn=shap_fn,
        )

        # run_oof already enforces G2 (every asset predicted exactly once) and, via
        # proba_to_grade_matrix, the row-sum half of G5.

        check = oof_df.join(
            folds_df.select("asset_id", "fold_id"),
            on="asset_id",
            how="left",
            suffix="_check",
            validate="1:1",
        )
        if not check.select(
            (pl.col("fold_id") == pl.col("fold_id_check")).all()
        ).item():
            raise ValueError(f"{spec.cell_id} fold_id map disagrees with folds.parquet")

        if not oof_df["y_true"].is_between(1, 5).all():
            raise ValueError(f"{spec.cell_id} has y_true outside 1 to 5")

        tagged = oof_df.with_columns(
            pl.lit(spec.model_key).cast(MODELS).alias("model"),
            pl.lit(spec.feature_set).cast(FEATURE_SETS).alias("feature_set"),
        ).select(
            "asset_id",
            "fold_id",
            "feature_set",
            "model",
            "y_true",
            *[f"proba_{g}" for g in GRADES],
            "y_pred_argmax",
        )
        oof_frames.append(tagged)

        if spec.model_key == "fh_hgb":
            inverted = tagged.select(
                pl.any_horizontal(pl.col(f"proba_{g}") == 0.0 for g in GRADES).alias(
                    "inv"
                )
            )["inv"]
            fh_inversion_rates[spec.cell_id] = float(inverted.mean())  # type: ignore

        if shap_df is not None:
            shap_df.write_parquet(out_dir / f"oof_shap__{spec.cell_id}.parquet")

        print(
            f"{spec.cell_id}: {oof_df.height} OOF rows"
            + (" (+ SHAP)" if shap_df is not None else "")
        )

    # concat + write
    oof_all = pl.concat(oof_frames, how="vertical")
    oof_all.write_parquet(out_dir / "oof_predictions.parquet")

    # diagnostics + summary and description
    manifest = {
        "run_id": str(uuid4()),
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": "oof_grid",
        "smoke": bool(smoke),
        "seed": int(seed),
        "n_blocks": int(n_blocks),
        "n_splits": int(n_splits),
        "splitter_class": "StratifiedGroupKFold",
        "n_assets": n_assets,
        "grid": [
            {
                "cell_id": spec.cell_id,
                "model": spec.model_key,
                "feature_set": spec.feature_set,
                "params": spec.params,
                "wants_shap": spec.wants_shap,
            }
            for spec in grid
        ],
        "artefact_paths": {
            "folds": str(out_dir / "folds.parquet"),
            "oof_predictions": str(out_dir / "oof_predictions.parquet"),
            "oof_shap": [
                str(out_dir / f"oof_shap__{spec.cell_id}.parquet")
                for spec in grid
                if spec.wants_shap
            ],
        },
        "diagnostics": {
            "fh_inversion_rate": fh_inversion_rates,
            "exp_2y_truncation_rate": trunc_rate,
            "exp_coverage_rate": exp_coverage_rate,
            "protection_type_null_count": protection_type_nulls,
            "bedrock_categories": bedrock_categories,
        },
    }

    with (out_dir / "results.jsonl").open("a") as f:
        f.write(json.dumps(manifest) + "\n")

    print(
        f"\nwrote {oof_all.height} OOF rows ({len(grid)} cells x {n_assets} assets) to {out_dir}"
    )
    print(f"fh_inversion_rate: {fh_inversion_rates}")
    print(
        f"exp_2y_truncation_rate: {trunc_rate:.4f}, exp_coverage_rate: {exp_coverage_rate:.4f}"
    )


if __name__ == "__main__":
    main()
