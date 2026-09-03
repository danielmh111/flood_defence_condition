# loading side only. no metrics live here - every quantity notebook 21 needs is derivable from
# oof/*.parquet and folds.parquet, and the decision rules are still being revised.
#
# 2,883 cells is ~67M rows and ~1.2 GB, so `reduce_cells` stays the primary path: it streams one
# cell at a time and holds only what the caller's reduce fn returns. `load_oof` is for an explicit
# handful, where the row-level validation gates are still affordable.

import json
from pathlib import Path
from typing import Callable

import polars as pl
from loguru import logger

from experiments.final_grid.models import GRID_AXES
from src.cross_validate import GRADES

PROBA_COLS = [f"proba_{g}" for g in GRADES]
FRANK_HALL_COLS = ["inversion_rate", "n_inverted", "n_predicted"]


def load_run(out_dir: Path) -> dict:
    """the most recent runs.jsonl line. lines are appended, never rewritten, so earlier ones are
    the record of previous invocations - the grid pass, the frank-hall pass, the final shap pass."""
    lines = (Path(out_dir) / "runs.jsonl").read_text().splitlines()
    return json.loads(lines[-1])


def load_runs(out_dir: Path) -> pl.DataFrame:
    lines = (Path(out_dir) / "runs.jsonl").read_text().splitlines()
    return pl.DataFrame([json.loads(line) for line in lines], infer_schema_length=None)


def load_folds(out_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(Path(out_dir) / "folds.parquet")


def load_cells(out_dir: Path) -> pl.DataFrame:
    """one row per completed cell, with capacity and the tuned axes flattened out of `params`. the
    frank-hall diagnostics come through as per-fold lists, null on every hgb cell."""
    rows = []
    for path in sorted((Path(out_dir) / "meta").glob("*.json")):
        record = json.loads(path.read_text())
        rows.append(
            {
                "cell_id": record["cell_id"],
                "run_id": record["run_id"],
                "model": record["model"],
                "composition": record["composition"],
                "capacity": record["capacity"],
                **{axis: record["params"].get(axis) for axis in GRID_AXES if axis != "capacity"},
                "max_iter": record["params"].get("max_iter"),
                "shap": record["shap"],
                "n_shap_rows": record["n_shap_rows"],
                **{col: record.get(col) for col in FRANK_HALL_COLS},
                "seconds_fit": record["seconds_fit"],
                "seconds_shap": record["seconds_shap"],
                "seconds_total": record["seconds_total"],
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _cell_paths(out_dir: Path, cell_ids: list[str] | None) -> list[Path]:
    out_dir = Path(out_dir)
    if cell_ids is None:
        return sorted((out_dir / "oof").glob("*.parquet"))
    return [out_dir / "oof" / f"{c}.parquet" for c in cell_ids]


def _read_oof(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).with_columns(
        (pl.col("proba_4") + pl.col("proba_5")).cast(pl.Float64).alias("score_ge4"),
        (pl.col("y_true") >= 4).cast(pl.Int8).alias("y_ge4"),
    )


def load_oof(out_dir: Path, cell_ids: list[str]) -> pl.DataFrame:
    """concat an explicit handful of cells, validated. deliberately takes no glob-all default -
    the full set is ~67M rows and belongs in `reduce_cells`."""
    oof = pl.concat([_read_oof(p) for p in _cell_paths(out_dir, cell_ids)], how="vertical")

    row_sum = oof.select(pl.sum_horizontal(PROBA_COLS).cast(pl.Float64).alias("s"))["s"]
    n_bad = int((row_sum - 1.0).abs().gt(1e-5).sum())
    if n_bad:
        raise ValueError(f"{n_bad} rows have proba_* summing off 1.0 by more than 1e-5")
    if not oof["y_true"].is_between(1, 5).all():
        raise ValueError("y_true outside 1..5")

    counts = oof.group_by("cell_id").len()
    if counts["len"].n_unique() != 1:
        raise ValueError(f"cells do not all have the same row count:\n{counts}")

    return oof


def reduce_cells(
    out_dir: Path,
    fn: Callable[[pl.DataFrame], pl.DataFrame],
    cell_ids: list[str] | None = None,
    log_every: int = 250,
) -> pl.DataFrame:
    """apply `fn` to one cell's OOF frame at a time and concat only its output. `fn` receives the
    frame with score_ge4 / y_ge4 already derived and should return a small per-cell summary."""
    paths = _cell_paths(out_dir, cell_ids)
    if not paths:
        raise ValueError(f"no OOF parquets under {Path(out_dir) / 'oof'}")

    frames = []
    for i, path in enumerate(paths, start=1):
        frames.append(fn(_read_oof(path)))
        if log_every and i % log_every == 0:
            logger.info(f"{i}/{len(paths)} cells reduced")

    return pl.concat(frames, how="vertical")


def load_shap_agg(out_dir: Path, cell_ids: list[str] | None = None) -> pl.DataFrame:
    """mean_abs_shap / mean_shap per feature x grade x fold, reference config only on the main grid"""
    out_dir = Path(out_dir)
    paths = (
        sorted((out_dir / "shap_agg").glob("*.parquet"))
        if cell_ids is None
        else [out_dir / "shap_agg" / f"{c}.parquet" for c in cell_ids]
    )
    return pl.concat([pl.read_parquet(p) for p in paths], how="vertical")


def load_shap_full(out_dir: Path, cell_id: str) -> pl.DataFrame:
    """per-row wide shap, written by the final pass only - one file per composition"""
    return pl.read_parquet(Path(out_dir) / "shap_full" / f"{cell_id}.parquet")
