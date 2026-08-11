# coarse grid: 5 feature sets x 50 configs + 3 reference cells, flat. every cell is one
# independent 10-fold spatially blocked OOF pass through run_oof. folds are built exactly as in
# oof_grid (KMeans blocks seed 123, StratifiedGroupKFold seed 123) so results pair with it.
#
# artefacts, one file per cell keyed by cell_id:
#   folds.parquet                shared across every cell
#   oof/{cell_id}.parquet        OOF probabilities
#   shap_agg/{cell_id}.parquet   mean_abs / mean shap per feature x grade x fold
#   shap_full/{cell_id}.parquet  per-row wide shap, reference config only
#   results.jsonl                run header, one line per completed cell, footer
#
# resume: a cell whose oof parquet exists is skipped. cancel with ctrl-c, edit the grid, rerun.
#
# the grid, fold count, block count and seed are fixed by the experiment design (spec 5 and 7),
# not runtime parameters - a run that could be pointed at a different grid would not be the
# experiment. the only mode is `smoke`, which is reachable as main(smoke=True).
#
# run from the repo root:
#   uv run python -m experiments.coarse_grid.runner

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import numpy as np
import polars as pl
from loguru import logger
from project_paths import paths
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from experiments.coarse_grid.features import (
    CLIMATE,
    LIDAR,
    attach_blocks,
    build_all,
)
from experiments.coarse_grid.models import N_CONFIGS, SEED, build_cells, sample_configs
from src.cross_validate import (
    GRADES,
    fold_ids_from_splitter,
    produce_spatial_blocks,
    run_oof,
)
from src.features import LEAKY_COLS, prepare_features
from src.oof_shap import make_tree_shap_fn

N_BLOCKS = 50
N_SPLITS = 10
PROTECTION_TYPE_NULL_COUNT = 285  # known number, found in nb14 and baseline.py

PARQUET = paths.processed_data / "unified_aims_eir_bgs.parquet"
GPKG = paths.aims_data / "aims.gpkg"
CLIMATE_PARQUET = paths.processed_data / "climate_features.parquet"
LIDAR_PARQUET = paths.processed_data / "lidar_features.parquet"
HEURISTIC_PARQUET = paths.processed_data / "heuristic_national.parquet"
OUT = paths.experiments / "coarse_grid"
SMOKE_OUT = paths.experiments / "coarse_grid_smoke"


def append_result(out_dir: Path, record: dict) -> None:
    with (out_dir / "results.jsonl").open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_frame(parquet: Path) -> pl.DataFrame:
    df_feats = prepare_features(pl.read_parquet(parquet))

    leaky = set(LEAKY_COLS) & set(df_feats.columns)
    if leaky:
        raise ValueError(f"leaky columns present: {sorted(leaky)}")
    if df_feats["asset_id"].n_unique() != df_feats.height:
        raise ValueError("asset_id is not unique")
    if not df_feats["condition_grade"].is_between(1, 5).all():
        raise ValueError("condition_grade outside 1..5")
    if not df_feats["asset_length_log1p"].is_finite().all():
        raise ValueError("asset_length_log1p has non-finite values")

    nulls = int(df_feats["aims__protection_type"].null_count())
    if nulls != PROTECTION_TYPE_NULL_COUNT:
        raise ValueError(
            f"aims__protection_type null count drifted ({nulls} != known "
            f"{PROTECTION_TYPE_NULL_COUNT}) - investigate before trusting this run"
        )
    return df_feats


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
    if joined["x"].null_count():
        raise ValueError("asset(s) with no centroid in aims.gpkg")
    return joined.select("x", "y").to_numpy()


def load_heuristic(asset_ids: np.ndarray, parquet: Path):
    """(n, 1) column of precomputed grades, NaN where the asset is not mappable onto a curve"""
    heuristic = pl.read_parquet(parquet).select("asset_id", "pred_grade")
    aligned = pl.DataFrame({"asset_id": asset_ids.astype(np.int64)}).join(
        heuristic, on="asset_id", how="left", validate="1:1", maintain_order="left"
    )
    X = aligned.select(pl.col("pred_grade").cast(pl.Float64)).to_numpy()
    return X, float(1.0 - np.isnan(X).mean())


def aggregate_shap(shap_df: pl.DataFrame, cell_id: str) -> pl.DataFrame:
    """mean_abs_shap / mean_shap per feature x grade x fold - ~900 rows, and what nb16's
    importance table and bump plot actually used. the per-row form is only needed for beeswarms."""
    shap_cols = [c for c in shap_df.columns if c.startswith("shap__")]
    return (
        shap_df.select("fold_id", *shap_cols)
        .unpivot(index="fold_id", on=shap_cols, variable_name="col", value_name="shap")
        .with_columns(
            pl.col("col").str.extract(r"^shap__(.+)__grade\d+$", 1).alias("feature"),
            pl.col("col").str.extract(r"__grade(\d+)$", 1).cast(pl.Int8).alias("grade"),
        )
        .group_by("fold_id", "feature", "grade")
        .agg(
            pl.col("shap").abs().mean().alias("mean_abs_shap"),
            pl.col("shap").mean().alias("mean_shap"),
        )
        .with_columns(pl.lit(cell_id).alias("cell_id"))
        .sort("fold_id", "grade", "mean_abs_shap", descending=[False, False, True])
    )


def build_folds(asset_ids, X, y, coords, out_dir: Path, n_splits: int):
    blocks = produce_spatial_blocks(coords, n_blocks=N_BLOCKS, seed=SEED)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_id = fold_ids_from_splitter(splitter, X, y, groups=blocks)

    folds = pl.DataFrame(
        {
            "asset_id": asset_ids.astype(np.int64),
            "block_id": blocks.astype(np.int16),
            "fold_id": fold_id,
        }
    )

    path = out_dir / "folds.parquet"
    if path.exists():
        # a resumed run must reuse the same folds or the completed cells stop being paired
        if not pl.read_parquet(path).equals(folds):
            raise ValueError(
                "folds.parquet on disk does not match the folds just built - the asset set, "
                "seed or fold count has changed, so completed cells are not comparable. move "
                "the output directory aside and start a clean run."
            )
        logger.info("folds match on disk, resuming")
    else:
        folds.write_parquet(path)

    return fold_id


def main(smoke: bool = False) -> None:
    if "max_features" not in HistGradientBoostingClassifier().get_params():
        raise ValueError("scikit-learn is too old - HGB has no max_features (needs >= 1.4)")

    # smoke writes to its own directory so its 2-fold folds.parquet can never collide with the
    # real run's, which would trip the fold gate
    out_dir = SMOKE_OUT if smoke else OUT
    n_configs = 3 if smoke else N_CONFIGS
    n_splits = 2 if smoke else N_SPLITS
    shap_rows = 50 if smoke else None

    for sub in ("oof", "shap_agg", "shap_full"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    df_joined, coverage = attach_blocks(load_frame(PARQUET), CLIMATE_PARQUET, LIDAR_PARQUET)
    matrices, names_by_set, cat_idx_by_set, y, asset_ids = build_all(df_joined)

    logger.info(f"loaded {len(y)} assets")
    for name, cat_idx in cat_idx_by_set.items():
        logger.info(f"  {name}: {matrices[name].shape[1]} features, cat_idx={cat_idx}")
    for col in (*CLIMATE, *LIDAR):
        logger.info(f"  coverage {col}: {coverage[col]:.1%}")

    fold_id = build_folds(
        asset_ids, matrices["identity"], y, load_coords(asset_ids, GPKG), out_dir, n_splits
    )

    heuristic_X, heuristic_coverage = None, None
    if HEURISTIC_PARQUET.exists():
        heuristic_X, heuristic_coverage = load_heuristic(asset_ids, HEURISTIC_PARQUET)
        logger.info(f"heuristic_national: {heuristic_coverage:.1%} of assets mappable")
    else:
        logger.info(f"heuristic_national skipped - {HEURISTIC_PARQUET} not found")

    configs = sample_configs(n=n_configs)
    cells = build_cells(cat_idx_by_set, configs, include_heuristic=heuristic_X is not None)

    done = {p.stem for p in (out_dir / "oof").glob("*.parquet")}
    todo = [c for c in cells if c.cell_id not in done]
    logger.info(f"{len(cells)} cells total, {len(done)} already complete, {len(todo)} to run")

    run_id = str(uuid4())
    append_result(
        out_dir,
        {
            "kind": "run_start",
            "run_id": run_id,
            "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": "coarse_grid",
            "smoke": smoke,
            "seed": SEED,
            "n_blocks": N_BLOCKS,
            "n_splits": n_splits,
            "n_assets": int(len(y)),
            "n_configs": n_configs,
            "shap_rows_per_fold": shap_rows,
            "n_cells_total": len(cells),
            "n_cells_todo": len(todo),
            "configs": configs,
            "feature_sets": names_by_set,
            "block_coverage": coverage,
            "heuristic_coverage": heuristic_coverage,
            "protection_type_null_count": PROTECTION_TYPE_NULL_COUNT,
        },
    )

    elapsed_each = []
    for i, spec in enumerate(todo, start=1):
        if spec.model == "heuristic_national":
            X, names = heuristic_X, ["heuristic_pred_grade"]
        elif spec.feature_set is None:
            X, names = matrices["identity"], names_by_set["identity"]  # dummies ignore X
        else:
            X, names = matrices[spec.feature_set], names_by_set[spec.feature_set]

        shap_fn = (
            make_tree_shap_fn(names, max_rows_per_fold=shap_rows)
            if spec.model == "hgb"
            else None
        )

        t0 = time.perf_counter()
        oof_df, shap_df = run_oof(
            spec.make_model, X, y, fold_id, asset_ids, shap_fn=shap_fn
        )
        elapsed = time.perf_counter() - t0
        elapsed_each.append(elapsed)

        tagged = oof_df.with_columns(
            pl.lit(spec.cell_id).alias("cell_id"),
            pl.lit(spec.model).alias("model"),
            pl.lit(spec.feature_set, dtype=pl.String).alias("feature_set"),
        ).select(
            "asset_id",
            "fold_id",
            "cell_id",
            "model",
            "feature_set",
            "y_true",
            *[f"proba_{g}" for g in GRADES],
            "y_pred_argmax",
        )

        # shap first, oof last - the oof file existing IS the completion flag, so writing it
        # first would let a crash mid-shap mark a cell complete with no shap artefact
        if shap_df is not None:
            aggregate_shap(shap_df, spec.cell_id).write_parquet(
                out_dir / "shap_agg" / f"{spec.cell_id}.parquet"
            )
            if spec.wants_full_shap:
                shap_df.write_parquet(out_dir / "shap_full" / f"{spec.cell_id}.parquet")
        tagged.write_parquet(out_dir / "oof" / f"{spec.cell_id}.parquet")

        append_result(
            out_dir,
            {
                "kind": "cell",
                "run_id": run_id,
                "cell_id": spec.cell_id,
                "model": spec.model,
                "feature_set": spec.feature_set,
                "params": spec.params,
                "wants_full_shap": spec.wants_full_shap,
                "seconds": round(elapsed, 2),
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        mean_s = float(np.mean(elapsed_each))
        eta = timedelta(seconds=int(mean_s * (len(todo) - i)))
        logger.info(
            f"{i}/{len(todo)}  {spec.feature_set or spec.model}/{spec.cell_id}  "
            f"{elapsed:6.1f}s  (mean {mean_s:5.1f}s, eta {eta})"
        )

    total = float(np.sum(elapsed_each)) if elapsed_each else 0.0
    append_result(
        out_dir,
        {
            "kind": "run_end",
            "run_id": run_id,
            "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_cells_run": len(todo),
            "total_seconds": round(total, 1),
        },
    )
    logger.info(f"done: {len(todo)} cells in {timedelta(seconds=int(total))}")


if __name__ == "__main__":
    main()
