# final grid: 2 compositions x 1,440 configs + 3 reference cells, flat. every cell is one
# independent 10-fold spatially blocked OOF pass through run_oof. folds are built exactly as in the
# fine grid (KMeans blocks seed 123, StratifiedGroupKFold seed 123) so cells pair across experiments.
#
# this module also holds the machinery the two sibling entry points reuse - prepare(), init_worker(),
# run_cell() and run_pass() - so atomicity, resume and the run record are defined once:
#   runner.py       the main grid
#   frank_hall.py   the frank-hall paired comparison at selected configs (spec 6)
#   final_shap.py   uncapped per-row shap at the selected config (spec 7)
#
# artefacts, one file per cell keyed by cell_id:
#   folds.parquet                shared across every cell and every pass
#   oof/{cell_id}.parquet        OOF probabilities
#   shap_agg/{cell_id}.parquet   mean_abs / mean shap per feature x grade x fold
#   shap_full/{cell_id}.parquet  per-row wide shap, final pass only
#   meta/{cell_id}.json          per-cell record, written by the worker
#   runs.jsonl                   ONE line per invocation, written by the parent at the end
#
# resume: a cell whose completion parquet exists is skipped. every write is atomic (tmp +
# os.replace), so a killed process cannot leave a truncated parquet that the next resume reads as
# complete.
#
# parallelism is over cells, not within a fit. small trees saturate OpenMP badly, so each worker is
# pinned to one thread and 12 run at once. the fine grid measured ~11.8x net that way.
#
# run from the repo root:
#   uv run python -m experiments.final_grid.runner
#   uv run python -m experiments.final_grid.runner --smoke

import os

# OpenMP reads these when its runtime initialises, so they must be set before numpy or sklearn is
# imported - setting them afterwards is a no-op. under spawn every worker re-executes this module,
# so the assignment lands in each process. the sibling entry points import this module first for
# the same reason.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import numpy as np
import polars as pl
from loguru import logger
from project_paths import paths
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from experiments.final_grid.features import (
    CLIMATE_BLOCK,
    COMPOSITIONS,
    LIDAR_BLOCK,
    attach_blocks,
    build_all,
    check_compositions,
)
from experiments.final_grid.models import (
    CAPACITY,
    FIXED_PARAMS,
    L2,
    LEARNING_RATE,
    MAX_BINS,
    MAX_LEAF_NODES,
    MIN_SAMPLES_LEAF,
    N_CONFIGS,
    REFERENCE_CONFIG,
    SEED,
    CellSpec,
    build_cells,
    build_configs,
    check_unique,
    frank_hall_diagnostics,
    make_model,
    reference_cells,
)
from src.cross_validate import (
    GRADES,
    default_fit,
    fold_ids_from_splitter,
    produce_spatial_blocks,
    run_oof,
)
from src.features import LEAKY_COLS, prepare_features
from src.oof_shap import make_tree_shap_fn

N_WORKERS = 20  # 12 physical cores, one thread each
N_BLOCKS = 50
N_SPLITS = 10
BUDGET_SECONDS = int(6.5 * 3600)
PROTECTION_TYPE_NULL_COUNT = 285  # known number, found in nb14 and baseline.py

PARQUET = paths.processed_data / "unified_aims_eir_bgs.parquet"
GPKG = paths.aims_data / "aims.gpkg"
CLIMATE_PARQUET = paths.processed_data / "climate_features.parquet"
LIDAR_PARQUET = paths.processed_data / "lidar_features.parquet"
HEURISTIC_PARQUET = paths.processed_data / "heuristic_national.parquet"
OUT = paths.experiments / "final_grid"
SMOKE_OUT = paths.experiments / "final_grid_smoke"

PROBA_COLS = [f"proba_{g}" for g in GRADES]

SMOKE_WORKERS = 4
SMOKE_SPLITS = 2

# the most expensive corner is mandatory: per-cell cost spans ~200x here, and a smoke sampling
# only cheap cells would underestimate the projection by an order of magnitude. the reference config
# is in for the opposite reason - it is the only cell carrying shap, so it is what measures the
# shap ratio.
SMOKE_CONFIGS = [
    dict(REFERENCE_CONFIG),
    {
        "capacity": 1.6,
        "learning_rate": 0.05,
        "max_leaf_nodes": 4,
        "min_samples_leaf": 20,
        "max_bins": 16,
        "l2_regularization": 0.0,
    },
    {
        "capacity": 4.0,
        "learning_rate": 0.0125,
        "max_leaf_nodes": 16,
        "min_samples_leaf": 10,
        "max_bins": 32,
        "l2_regularization": 1.0,
    },
    {
        "capacity": 6.0,
        "learning_rate": 0.00625,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 5,
        "max_bins": 255,
        "l2_regularization": 0.0,
    },
]

_STATE: dict = {}


def write_atomic(path: Path, write) -> None:
    """write to a sibling .tmp then os.replace. a killed process can then never leave a truncated
    file that the resume glob reads as complete - which matters more with 12 processes in flight."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def git_sha() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return proc.stdout.strip()


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
    """mean_abs_shap / mean_shap per feature x grade x fold - ~600 rows, and what the importance
    tables actually read. the per-row form is only needed for beeswarms."""
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


def build_folds(asset_ids, X, y, coords, out_dir: Path, n_splits: int) -> None:
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
        # a resumed run, or a later pass, must reuse the same folds or the completed cells stop
        # being paired - with each other and with the fine grid
        if not pl.read_parquet(path).equals(folds):
            raise ValueError(
                "folds.parquet on disk does not match the folds just built - the asset set, seed or "
                "fold count has changed, so completed cells are not comparable. move the output "
                "directory aside and start a clean run."
            )
        logger.info("folds match on disk, resuming")
    else:
        write_atomic(path, folds.write_parquet)


def prepare(out_dir: Path, n_splits: int) -> dict:
    """everything the parent must do before a worker starts: load, gate, build folds. shared by all
    three entry points so the gates cannot drift between passes."""
    for sub in ("oof", "shap_agg", "shap_full", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    df_joined, coverage = attach_blocks(
        load_frame(PARQUET), CLIMATE_PARQUET, LIDAR_PARQUET
    )
    cardinality = check_compositions(df_joined, MAX_BINS)
    matrices, names, cat_idx_by_composition, y, asset_ids = build_all(df_joined)

    logger.info(f"loaded {len(y)} assets")
    for name in COMPOSITIONS:
        logger.info(f"  {name}: {matrices[name].shape[1]} features")
    for col, n in sorted(cardinality.items()):
        logger.info(f"  cardinality {col}: {n} (coarsest max_bins {min(MAX_BINS)})")
    for col in (*CLIMATE_BLOCK, *LIDAR_BLOCK):
        logger.info(f"  coverage {col}: {coverage[col]:.1%}")

    build_folds(
        asset_ids,
        matrices[next(iter(COMPOSITIONS))],
        y,
        load_coords(asset_ids, GPKG),
        out_dir,
        n_splits,
    )

    heuristic_coverage = None
    if HEURISTIC_PARQUET.exists():
        _, heuristic_coverage = load_heuristic(asset_ids, HEURISTIC_PARQUET)
        logger.info(f"heuristic_national: {heuristic_coverage:.1%} of assets mappable")
    else:
        logger.info(f"heuristic_national skipped - {HEURISTIC_PARQUET} not found")

    return {
        "cat_idx_by_composition": cat_idx_by_composition,
        "names": names,
        "n_assets": int(len(y)),
        "block_coverage": coverage,
        "categorical_cardinalities": cardinality,
        "heuristic_coverage": heuristic_coverage,
    }


def init_worker(out_dir: Path, run_id: str) -> None:
    """once per process, not once per task. under spawn, shipping the matrices with each task would
    re-pickle them once per cell; rebuilding them here costs ~30 MB per worker."""
    df_joined, _ = attach_blocks(load_frame(PARQUET), CLIMATE_PARQUET, LIDAR_PARQUET)
    matrices, names, _, y, asset_ids = build_all(df_joined)

    folds = pl.read_parquet(out_dir / "folds.parquet")
    if not np.array_equal(folds["asset_id"].to_numpy(), asset_ids.astype(np.int64)):
        raise ValueError("worker built a different asset order than folds.parquet")

    heuristic_X = (
        load_heuristic(asset_ids, HEURISTIC_PARQUET)[0]
        if HEURISTIC_PARQUET.exists()
        else None
    )

    _STATE.update(
        out_dir=out_dir,
        run_id=run_id,
        matrices=matrices,
        names=names,
        y=y,
        asset_ids=asset_ids,
        fold_id=folds["fold_id"].to_numpy(),
        heuristic_X=heuristic_X,
    )


def _timed(seconds: dict, key: str, fn):
    def wrapped(*args):
        t0 = time.perf_counter()
        out = fn(*args)
        seconds[key] += time.perf_counter() - t0
        return out

    return wrapped


def run_cell(spec: CellSpec) -> dict:
    """one cell, start to finish, inside a worker. returns the meta record it wrote."""
    if spec.model == "heuristic_national":
        X, names = _STATE["heuristic_X"], ["heuristic_pred_grade"]
    elif spec.composition is None:
        # dummies ignore X, so any matrix will do
        X, names = _STATE["matrices"][next(iter(COMPOSITIONS))], None
    else:
        X, names = (
            _STATE["matrices"][spec.composition],
            _STATE["names"][spec.composition],
        )

    seconds = {"fit": 0.0, "shap": 0.0}
    if spec.model == "frank_hall":
        aux_fn = frank_hall_diagnostics
    elif spec.shap is not None:
        # no row cap anywhere: the fine grid explained all 8,820 of its cells and produced the
        # cross-config bump plots already, so this experiment explains four cells in total and can
        # afford every row of every fold
        aux_fn = _timed(
            seconds, "shap", make_tree_shap_fn(names, max_rows_per_fold=None, seed=SEED)
        )
    else:
        aux_fn = None

    t0 = time.perf_counter()
    with threadpool_limits(1):
        oof_df, aux_df = run_oof(
            lambda: make_model(spec),
            X,
            _STATE["y"],
            _STATE["fold_id"],
            _STATE["asset_ids"],
            fit_fn=_timed(seconds, "fit", default_fit),
            shap_fn=aux_fn,
        )
    total = time.perf_counter() - t0

    tagged = oof_df.with_columns(
        pl.col("asset_id").cast(pl.Int32),
        pl.col(PROBA_COLS).cast(pl.Float32),
        pl.lit(spec.cell_id).alias("cell_id"),
        pl.lit(spec.model).alias("model"),
        pl.lit(spec.composition, dtype=pl.String).alias("composition"),
    ).select(
        "asset_id",
        "fold_id",
        "cell_id",
        "model",
        "composition",
        "y_true",
        *PROBA_COLS,
        "y_pred_argmax",
    )

    meta = {
        "cell_id": spec.cell_id,
        "run_id": _STATE["run_id"],
        "model": spec.model,
        "composition": spec.composition,
        "params": spec.params,
        "capacity": spec.capacity,
        "shap": spec.shap,
        "n_shap_rows": 0,
        "seconds_fit": round(seconds["fit"], 3),
        "seconds_shap": round(seconds["shap"], 3),
        "seconds_total": round(total, 3),
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_dir = _STATE["out_dir"]
    if spec.model == "frank_hall":
        # the aux frame is the per-fold inversion diagnostics, not shap - it goes into meta rather
        # than to a parquet, and frank-hall cells write no shap artefact at all
        meta.update(aux_df.sort("fold_id").to_dict(as_series=False))
    elif aux_df is not None:
        meta["n_shap_rows"] = aux_df.height
        write_atomic(
            out_dir / "shap_agg" / f"{spec.cell_id}.parquet",
            aggregate_shap(aux_df, spec.cell_id).write_parquet,
        )
        if spec.shap == "full":
            write_atomic(
                out_dir / "shap_full" / f"{spec.cell_id}.parquet", aux_df.write_parquet
            )

    # shap first, oof last - the oof file existing IS the completion flag for the main grid, so
    # writing it first would let a crash mid-shap mark a cell complete with no shap artefact
    write_atomic(
        out_dir / "meta" / f"{spec.cell_id}.json",
        lambda p: p.write_text(json.dumps(meta, default=str)),
    )
    write_atomic(out_dir / "oof" / f"{spec.cell_id}.parquet", tagged.write_parquet)

    return meta


def run_pass(
    cells: list[CellSpec],
    out_dir: Path,
    n_workers: int,
    pass_name: str,
    record: dict,
    done_dir: str = "oof",
) -> list[dict]:
    """the pool, the progress log and the single runs.jsonl append, shared by all three entry
    points. `done_dir` is the artefact whose presence means a cell is finished - oof for the grid,
    but shap_full for the final pass, whose cells already have an oof from the grid."""
    done = {p.stem for p in (out_dir / done_dir).glob("*.parquet")}
    todo = [c for c in cells if c.cell_id not in done]
    # longest first: cost spans ~200x here, and without this the run ends with 11 idle workers
    # waiting on one long cell
    todo.sort(key=lambda c: -c.est_cost)
    logger.info(
        f"{len(cells)} cells total, {len(cells) - len(todo)} already complete, {len(todo)} to run"
    )

    run_id = str(uuid4())
    completed: list[dict] = []
    t_wall = time.perf_counter()

    with ProcessPoolExecutor(
        n_workers, initializer=init_worker, initargs=(out_dir, run_id)
    ) as pool:
        futures = [pool.submit(run_cell, spec) for spec in todo]
        try:
            for i, future in enumerate(as_completed(futures), start=1):
                meta = future.result()
                completed.append(meta)

                mean_s = float(np.mean([c["seconds_total"] for c in completed]))
                eta = timedelta(seconds=int(mean_s * (len(todo) - i) / n_workers))
                logger.info(
                    f"{i}/{len(todo)}  {meta['composition'] or meta['model']}/{meta['cell_id']}  "
                    f"{meta['seconds_total']:6.1f}s  (mean {mean_s:5.1f}s, eta {eta})"
                )
        except KeyboardInterrupt:
            logger.warning(
                "interrupted - cancelling queued cells, cells in flight will finish"
            )
            pool.shutdown(cancel_futures=True)

    wall = time.perf_counter() - t_wall

    with (out_dir / "runs.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": "final_grid",
                    "pass": pass_name,
                    "git_sha": git_sha(),
                    "seed": SEED,
                    "n_blocks": N_BLOCKS,
                    "n_workers": n_workers,
                    "omp_num_threads": int(os.environ["OMP_NUM_THREADS"]),
                    "n_cells_total": len(cells),
                    "n_cells_run": len(completed),
                    "n_cells_skipped": len(cells) - len(todo),
                    "protection_type_null_count": PROTECTION_TYPE_NULL_COUNT,
                    "oof_proba_dtype": "float32",
                    **record,
                    "total_seconds": round(
                        sum(c["seconds_total"] for c in completed), 1
                    ),
                    "total_seconds_fit": round(
                        sum(c["seconds_fit"] for c in completed), 1
                    ),
                    "total_seconds_shap": round(
                        sum(c["seconds_shap"] for c in completed), 1
                    ),
                    "wall_seconds": round(wall, 1),
                },
                default=str,
            )
            + "\n"
        )

    logger.info(f"done: {len(completed)} cells in {timedelta(seconds=int(wall))} wall")
    return completed


def report_projection(completed: list[dict], n_workers: int) -> None:
    """flat mean seconds per cell, scaled to the real run's fold count, cell count and worker count.

    the bias runs the other way from the fine grid's: the dearest corner is 1-in-4 among the smoke
    configs against 1-in-720 in the real grid, so this over-estimates. that is the safe direction,
    but read it as an order of magnitude rather than a promise."""
    total = sum(c["seconds_total"] for c in completed)
    shap = sum(c["seconds_shap"] for c in completed)
    ratio = shap / total if total else 0.0

    n_cells = N_CONFIGS * len(COMPOSITIONS) + 3
    mean_s = total / len(completed) * (N_SPLITS / SMOKE_SPLITS)
    projected = mean_s * n_cells / N_WORKERS

    logger.info(
        f"shap is {ratio:.1%} of runtime ({shap:.0f}s of {total:.0f}s), on 1 cell of 4"
    )
    logger.info(
        f"projected full run {timedelta(seconds=int(projected))} at {N_WORKERS} workers "
        f"({n_cells} cells, mean {mean_s:.1f}s/cell serial) - over-estimated, see docstring"
    )

    if projected > BUDGET_SECONDS:
        logger.warning(
            f"projection exceeds the {BUDGET_SECONDS / 3600:.1f}h budget - cut order is "
            "1) drop capacity 6.0, 2) drop min_samples_leaf 10, 3) drop max_bins 32. do not cut "
            "learning_rate 0.00625. apply by editing the constant; this runner applies nothing"
        )


def main(smoke: bool = False) -> None:
    # smoke writes to its own directory so its 2-fold folds.parquet can never collide with the real
    # run's, which would trip the fold gate
    out_dir = SMOKE_OUT if smoke else OUT
    n_splits = SMOKE_SPLITS if smoke else N_SPLITS
    n_workers = SMOKE_WORKERS if smoke else N_WORKERS

    context = prepare(out_dir, n_splits)

    configs = build_configs()
    if smoke:
        missing = [c for c in SMOKE_CONFIGS if c not in configs]
        if missing:
            raise ValueError(f"smoke config(s) not in the grid: {missing}")
        configs = SMOKE_CONFIGS
    logger.info(f"{len(configs)} configs")

    cells = check_unique(
        [
            *build_cells(
                context["cat_idx_by_composition"],
                configs,
                shap_for=REFERENCE_CONFIG,
                shap_tier="agg",
            ),
            *reference_cells(
                include_heuristic=context["heuristic_coverage"] is not None
            ),
        ]
    )

    completed = run_pass(
        cells,
        out_dir,
        n_workers,
        pass_name="grid",
        record={
            "smoke": smoke,
            "n_splits": n_splits,
            "n_assets": context["n_assets"],
            "n_configs": len(configs),
            "grid": {
                "capacity": CAPACITY,
                "learning_rate": LEARNING_RATE,
                "max_leaf_nodes": MAX_LEAF_NODES,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "max_bins": MAX_BINS,
                "l2_regularization": L2,
            },
            "fixed_params": FIXED_PARAMS,
            "reference_config": REFERENCE_CONFIG,
            "max_bins_levels": MAX_BINS,
            "compositions": context["names"],
            "categorical_cardinalities": context["categorical_cardinalities"],
            "block_coverage": context["block_coverage"],
            "heuristic_coverage": context["heuristic_coverage"],
        },
    )

    if smoke and completed:
        report_projection(completed, n_workers)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
