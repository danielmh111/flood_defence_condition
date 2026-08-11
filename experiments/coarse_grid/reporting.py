# loading side only. the per-cell artefact layout needs its own loader, but everything
# downstream of it - metrics_per_fold, metrics_summary, paired_deltas, threshold_sweep,
# per_grade_metrics - is reused unchanged from experiments/oof_grid/reporting.py, which already
# keys on cell_id and takes the same columns.

import json
from pathlib import Path

import polars as pl

from experiments.coarse_grid.models import GRID
from src.cross_validate import GRADES

PROBA_COLS = [f"proba_{g}" for g in GRADES]


def _records(out_dir: Path, kind: str) -> list[dict]:
    lines = (Path(out_dir) / "results.jsonl").read_text().splitlines()
    return [r for r in (json.loads(line) for line in lines) if r.get("kind") == kind]


def load_manifest(out_dir: Path) -> dict:
    """the most recent run_start header"""
    return _records(out_dir, "run_start")[-1]


def load_cells(out_dir: Path) -> pl.DataFrame:
    """one row per completed cell, with the tuned axes flattened out of `params` and a config_id
    giving the cell's position in the sampled grid (0 is the reference config)."""
    configs = {
        tuple(config[axis] for axis in GRID): i
        for i, config in enumerate(load_manifest(out_dir)["configs"])
    }

    rows = [
        {
            "cell_id": r["cell_id"],
            "model": r["model"],
            "feature_set": r["feature_set"],
            "config_id": configs.get(tuple(r["params"].get(axis) for axis in GRID)),
            **{axis: r["params"].get(axis) for axis in GRID},
            "seconds": r["seconds"],
        }
        for r in _records(out_dir, "cell")
    ]
    return pl.DataFrame(rows).unique(subset="cell_id", keep="last", maintain_order=True)


def load_oof(out_dir: Path, cell_ids: list[str] | None = None) -> pl.DataFrame:
    """concat the per-cell OOF parquets and derive score_ge4 / y_ge4"""
    out_dir = Path(out_dir)
    paths = (
        sorted((out_dir / "oof").glob("*.parquet"))
        if cell_ids is None
        else [out_dir / "oof" / f"{c}.parquet" for c in cell_ids]
    )
    if not paths:
        raise ValueError(f"no OOF parquets under {out_dir / 'oof'}")

    oof = pl.concat([pl.read_parquet(p) for p in paths], how="vertical")

    row_sum = oof.select(pl.sum_horizontal(PROBA_COLS).alias("s"))["s"]
    n_bad = int((row_sum - 1.0).abs().gt(1e-6).sum())
    if n_bad:
        raise ValueError(f"{n_bad} rows have proba_* summing off 1.0 by more than 1e-6")
    if not oof["y_true"].is_between(1, 5).all():
        raise ValueError("y_true outside 1..5")

    counts = oof.group_by("cell_id").len()
    if counts["len"].n_unique() != 1:
        raise ValueError(f"cells do not all have the same row count:\n{counts}")

    return oof.with_columns(
        (pl.col("proba_4") + pl.col("proba_5")).alias("score_ge4"),
        (pl.col("y_true") >= 4).cast(pl.Int8).alias("y_ge4"),
    )
