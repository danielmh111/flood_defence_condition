"""downstream metric surface for the OOF grid (spec §9). everything here reads
oof_predictions.parquet (via `load_oof`) and nothing else - no re-fitting, no re-scoring
against Predictions objects. `src/metrics.py`'s scorers are (y, Predictions)-shaped and
belong to the legacy fold-scalar `cross_validate` path; the two surfaces are deliberately
kept separate.
"""

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, cohen_kappa_score, mean_absolute_error

from src.cross_validate import GRADES

PROBA_COLS = [f"proba_{g}" for g in GRADES]
CELL_KEYS = ("model", "feature_set")
_GRADE_FRAME = pl.DataFrame({"grade": list(GRADES)}, schema={"grade": pl.Int8})


def load_oof(path: str | Path) -> pl.DataFrame:
    """load + validate against spec §7.1, and derive cell_id, score_ge4, y_ge4.

    cell_id = model + "__" + feature_set is computed once here so everything downstream
    can key on a single string; the compound (model, feature_set) key would make the
    vs-floor contrast in `paired_deltas` ambiguous (it deliberately crosses feature sets).
    """
    oof = pl.read_parquet(path)

    required = {
        "asset_id",
        "fold_id",
        "feature_set",
        "model",
        "y_true",
        *PROBA_COLS,
        "y_pred_argmax",
    }
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"oof frame missing columns: {sorted(missing)}")

    n_bad_grade = oof.filter(~pl.col("y_true").is_in(list(GRADES))).height
    if n_bad_grade:
        raise ValueError(f"{n_bad_grade} rows have y_true outside {list(GRADES)}")

    row_sum = oof.select(pl.sum_horizontal(PROBA_COLS).alias("s")).to_series()
    n_bad_sum = int((row_sum - 1.0).abs().gt(1e-6).sum())
    if n_bad_sum:
        raise ValueError(f"{n_bad_sum} rows have proba_* summing off 1.0 by more than 1e-6")

    n_dupe = oof.height - oof.select("asset_id", "model", "feature_set").unique().height
    if n_dupe:
        raise ValueError(f"{n_dupe} duplicate (asset_id, model, feature_set) rows")

    counts = oof.group_by("model", "feature_set").len()
    if counts["len"].n_unique() != 1:
        raise ValueError(
            f"cells do not all have the same row count:\n{counts.sort('model', 'feature_set')}"
        )

    return oof.with_columns(
        (pl.col("model").cast(pl.String) + "__" + pl.col("feature_set").cast(pl.String)).alias(
            "cell_id"
        ),
        (pl.col("proba_4") + pl.col("proba_5")).alias("score_ge4"),
        (pl.col("y_true") >= 4).cast(pl.Int8).alias("y_ge4"),
    )


def _nan_to_null(df: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c) for c in cols
    )


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, score_ge4: np.ndarray) -> dict:
    y_ge4 = (y_true >= 4).astype(np.int64)
    qwk = float(cohen_kappa_score(y_true, y_pred, labels=list(GRADES), weights="quadratic"))
    pr_auc = (
        float(average_precision_score(y_ge4, score_ge4)) if y_ge4.sum() > 0 else float("nan")
    )
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "n": int(len(y_true)),
        "n_pos_ge4": int(y_ge4.sum()),
        "qwk": qwk,
        "pr_auc_ge4": pr_auc,
        "mae": mae,
    }


def metrics_per_fold(oof: pl.DataFrame) -> pl.DataFrame:
    """qwk / pr_auc_ge4 / mae, one row per (cell, fold). `labels=[1..5]` on qwk is not
    optional - without it a fold whose argmax never emits grade 1 or 5 gets a different
    kappa denominator. mae is report-only (spec §9.3), never a selection criterion.
    """
    rows = []
    for (cell_id, model, feature_set, fold_id), g in oof.group_by(
        ["cell_id", "model", "feature_set", "fold_id"], maintain_order=True
    ):
        m = _fold_metrics(
            g["y_true"].to_numpy(), g["y_pred_argmax"].to_numpy(), g["score_ge4"].to_numpy()
        )
        rows.append(
            {"cell_id": cell_id, "model": model, "feature_set": feature_set, "fold_id": fold_id, **m}
        )
    out = pl.DataFrame(rows).sort("cell_id", "fold_id")
    return _nan_to_null(out, ["qwk", "pr_auc_ge4", "mae"])


def metrics_pooled(oof: pl.DataFrame) -> pl.DataFrame:
    """qwk / pr_auc_ge4 / mae over all OOF rows, one row per cell."""
    rows = []
    for (cell_id, model, feature_set), g in oof.group_by(
        ["cell_id", "model", "feature_set"], maintain_order=True
    ):
        m = _fold_metrics(
            g["y_true"].to_numpy(), g["y_pred_argmax"].to_numpy(), g["score_ge4"].to_numpy()
        )
        m.pop("n_pos_ge4")
        rows.append({"cell_id": cell_id, "model": model, "feature_set": feature_set, **m})
    out = pl.DataFrame(rows).sort("cell_id")
    return _nan_to_null(out, ["qwk", "pr_auc_ge4", "mae"])


def metrics_summary(per_fold: pl.DataFrame) -> pl.DataFrame:
    """long-form mean/std/n_folds over `metrics_per_fold`, mirroring metrics.report_means
    (nanmean/nanstd - undefined folds are already null via `_nan_to_null`, ddof=0)."""
    id_cols = ["cell_id", "model", "feature_set"]
    metric_cols = ["qwk", "pr_auc_ge4", "mae"]

    long = per_fold.unpivot(
        index=[*id_cols, "fold_id"], on=metric_cols, variable_name="metric", value_name="value"
    )
    return (
        long.group_by([*id_cols, "metric"], maintain_order=True)
        .agg(
            pl.col("value").drop_nulls().mean().alias("mean"),
            pl.col("value").drop_nulls().std(ddof=0).alias("std"),
            pl.col("value").drop_nulls().len().cast(pl.Int32).alias("n_folds"),
        )
        .sort("cell_id", "metric")
    )


def confusion(oof: pl.DataFrame, by_fold: bool = False) -> pl.DataFrame:
    """5x5 confusion, long form, zero-filled so all 25 (y_true, y_pred_argmax) cells exist
    per group - the total-coverage grid is what makes `per_grade_metrics` a safe join.
    """
    keys = ["cell_id", "model", "feature_set"] + (["fold_id"] if by_fold else [])

    counted = oof.group_by([*keys, "y_true", "y_pred_argmax"]).agg(pl.len().cast(pl.Int32).alias("n"))

    distinct_cells = oof.select(keys).unique()
    grid = (
        distinct_cells.join(_GRADE_FRAME.rename({"grade": "y_true"}), how="cross")
        .join(_GRADE_FRAME.rename({"grade": "y_pred_argmax"}), how="cross")
    )

    return (
        grid.join(counted, on=[*keys, "y_true", "y_pred_argmax"], how="left")
        .with_columns(pl.col("n").fill_null(0).cast(pl.Int32))
        .sort([*keys, "y_true", "y_pred_argmax"])
    )


def confusion_wide(conf: pl.DataFrame) -> pl.DataFrame:
    """pivot the long confusion frame to y_true (rows) x y_pred_argmax (columns) per cell."""
    index_cols = [c for c in conf.columns if c not in ("y_pred_argmax", "n")]
    return conf.pivot(on="y_pred_argmax", index=index_cols, values="n").sort(index_cols)


def per_grade_metrics(
    oof: pl.DataFrame, by_fold: bool = False, zero_division: float | None = None
) -> pl.DataFrame:
    """precision/recall/F1 per grade, one-vs-rest, derived entirely from `confusion` so the
    two surfaces can never disagree. null (or `zero_division`, if given) on a zero
    denominator - replaces the metrics.py per-grade recall/precision, which return NaN.
    """
    keys = ["cell_id", "model", "feature_set"] + (["fold_id"] if by_fold else [])
    conf = confusion(oof, by_fold=by_fold)

    tp = conf.filter(pl.col("y_true") == pl.col("y_pred_argmax")).select(
        *keys, pl.col("y_true").alias("grade"), pl.col("n").alias("tp")
    )
    support = conf.group_by(*keys, pl.col("y_true").alias("grade")).agg(
        pl.col("n").sum().alias("support")
    )
    n_pred = conf.group_by(*keys, pl.col("y_pred_argmax").alias("grade")).agg(
        pl.col("n").sum().alias("n_pred")
    )

    out = support.join(n_pred, on=[*keys, "grade"], how="left").join(
        tp, on=[*keys, "grade"], how="left"
    )

    def _rate(num: str, den: str) -> pl.Expr:
        expr = pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den))
        if zero_division is not None:
            expr = expr.otherwise(pl.lit(float(zero_division)))
        return expr

    out = out.with_columns(_rate("tp", "n_pred").alias("precision"), _rate("tp", "support").alias("recall"))

    f1 = pl.when((pl.col("precision") + pl.col("recall")) > 0).then(
        2 * pl.col("precision") * pl.col("recall") / (pl.col("precision") + pl.col("recall"))
    )
    if zero_division is not None:
        f1 = f1.otherwise(pl.lit(float(zero_division)))
    out = out.with_columns(f1.alias("f1"))

    return out.select(
        *keys,
        pl.col("grade").cast(pl.Int8),
        pl.col("support").cast(pl.Int32),
        pl.col("n_pred").cast(pl.Int32),
        pl.col("tp").cast(pl.Int32),
        "precision",
        "recall",
        "f1",
    ).sort([*keys, "grade"])


def default_contrasts(cells: Sequence[str]) -> list[tuple[str, str, str]]:
    """the four families of spec §9.4, restricted to cells actually present in `cells`.

    exposure: base_exp - base within each of hgb_default/hgb_qwk_es/fh_hgb.
    ordinal: fh_hgb - hgb_default within each feature set.
    objective: hgb_qwk_es - hgb_default within each feature set.
    vs_floor: every non-dummy cell - dummy_stratified__base.
    """
    cell_set = set(cells)
    contrasts: list[tuple[str, str, str]] = []

    for model in ("hgb_default", "hgb_qwk_es", "fh_hgb"):
        a, b = f"{model}__base_exp", f"{model}__base"
        if a in cell_set and b in cell_set:
            contrasts.append(("exposure", a, b))

    for feature_set in ("base", "base_exp"):
        a, b = f"fh_hgb__{feature_set}", f"hgb_default__{feature_set}"
        if a in cell_set and b in cell_set:
            contrasts.append(("ordinal", a, b))

    for feature_set in ("base", "base_exp"):
        a, b = f"hgb_qwk_es__{feature_set}", f"hgb_default__{feature_set}"
        if a in cell_set and b in cell_set:
            contrasts.append(("objective", a, b))

    floor = "dummy_stratified__base"
    if floor in cell_set:
        for cell in sorted(cell_set):
            if cell != floor and not cell.startswith("dummy"):
                contrasts.append(("vs_floor", cell, floor))

    return contrasts


def paired_deltas(
    per_fold: pl.DataFrame,
    contrasts: Sequence[tuple[str, str, str]],
    metrics: Sequence[str] = ("qwk", "pr_auc_ge4", "mae"),
    ddof: int = 1,
) -> pl.DataFrame:
    """within-fold deltas d_k = m_A(fold k) - m_B(fold k) for each (family, cell_a, cell_b)
    contrast, over folds where both are defined. keyed by cell_id only (never model/
    feature_set) - the vs_floor contrast deliberately crosses feature sets, pairing e.g.
    hgb_default__base_exp against dummy_stratified__base. `ddof=1` (inferential in spirit,
    per spec §9.4); note this differs from metrics.report_means's np.nanstd (ddof=0).
    """
    if not contrasts:
        return pl.DataFrame(
            schema={
                "family": pl.String,
                "cell_a": pl.String,
                "cell_b": pl.String,
                "metric": pl.String,
                "mean_d": pl.Float64,
                "std_d": pl.Float64,
                "n_folds_paired": pl.Int32,
                "frac_d_gt0": pl.Float64,
            }
        )

    long = per_fold.unpivot(
        index=["cell_id", "fold_id"], on=list(metrics), variable_name="metric", value_name="value"
    )
    contrasts_df = pl.DataFrame(
        list(contrasts), schema=["family", "cell_a", "cell_b"], orient="row"
    )

    a = long.rename({"cell_id": "cell_a", "value": "value_a"})
    b = long.rename({"cell_id": "cell_b", "value": "value_b"})

    paired = (
        contrasts_df.join(a, on="cell_a", how="left")
        .join(b, on=["cell_b", "fold_id", "metric"], how="inner")
        .with_columns((pl.col("value_a") - pl.col("value_b")).alias("d"))
        .drop_nulls("d")
    )

    return (
        paired.group_by(["family", "cell_a", "cell_b", "metric"], maintain_order=True)
        .agg(
            pl.col("d").mean().alias("mean_d"),
            pl.col("d").std(ddof=ddof).alias("std_d"),
            pl.col("d").len().cast(pl.Int32).alias("n_folds_paired"),
            (pl.col("d") > 0).mean().alias("frac_d_gt0"),
        )
        .sort("family", "cell_a", "cell_b", "metric")
    )


def threshold_sweep(oof: pl.DataFrame, thresholds: np.ndarray | None = None) -> pl.DataFrame:
    """grade->=4 PR sweep on score_ge4, per cell.

    event-based by default (not a fixed grid: 8 cells x 23,304 rows x 1,001 thresholds
    would be ~187M rows). sorts descending by score_ge4 within cell, accumulates tp/k,
    dedupes tied scores - this is what sklearn's precision_recall_curve computes, O(n log n),
    <= n rows per cell, and yields every achievable operating point. pass `thresholds` for
    a fixed grid instead, where cross-cell plotting parity is wanted.

    two things that look like bugs but are not: `dummy_*` cells emit a one-hot
    predict_proba (not the prior), so their score_ge4 takes ~1-2 distinct values and their
    curve has ~2-3 points; a high FH inversion rate is itself a finding (see frank_hall.py).
    """
    if thresholds is not None:
        return _threshold_grid_sweep(oof, thresholds)

    frames = []
    for cell_id, g in oof.group_by("cell_id", maintain_order=True):
        cell_id = cell_id[0] if isinstance(cell_id, tuple) else cell_id
        g = g.sort("score_ge4", descending=True)

        y_ge4 = g["y_ge4"].to_numpy().astype(np.int64)
        score = g["score_ge4"].to_numpy()
        n = len(y_ge4)

        tp = np.cumsum(y_ge4)
        k = np.arange(1, n + 1)
        p_total = int(tp[-1]) if n else 0

        precision = tp / k
        recall = tp / p_total if p_total > 0 else np.zeros(n)

        frame = pl.DataFrame(
            {
                "cell_id": [cell_id] * n,
                "model": g["model"].to_list(),
                "feature_set": g["feature_set"].to_list(),
                "threshold": score,
                "n_pred_pos": k.astype(np.int32),
                "tp": tp.astype(np.int32),
                "fp": (k - tp).astype(np.int32),
                "fn": (p_total - tp).astype(np.int32),
                "precision": precision,
                "recall": recall,
            }
        ).unique(subset=["cell_id", "threshold"], keep="last", maintain_order=True)

        frames.append(frame)

    out = pl.concat(frames, how="vertical")
    out = out.with_columns(
        pl.when((pl.col("precision") + pl.col("recall")) > 0)
        .then(2 * pl.col("precision") * pl.col("recall") / (pl.col("precision") + pl.col("recall")))
        .otherwise(0.0)
        .alias("f1")
    )
    return out.sort(["cell_id", "threshold"], descending=[False, True])


def _threshold_grid_sweep(oof: pl.DataFrame, thresholds: np.ndarray) -> pl.DataFrame:
    """fixed-grid variant, for cross-cell plotting parity. O(cells x rows x thresholds)."""
    keys = ["cell_id", "model", "feature_set"]
    rows = []
    for (cell_id, model, feature_set), g in oof.group_by(keys, maintain_order=True):
        y_ge4 = g["y_ge4"].to_numpy().astype(np.int64)
        score = g["score_ge4"].to_numpy()
        p_total = int(y_ge4.sum())

        for t in thresholds:
            pred_pos = score >= t
            tp = int((pred_pos & (y_ge4 == 1)).sum())
            fp = int((pred_pos & (y_ge4 == 0)).sum())
            fn = p_total - tp
            k = tp + fp
            precision = tp / k if k > 0 else 0.0
            recall = tp / p_total if p_total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            rows.append(
                {
                    "cell_id": cell_id,
                    "model": model,
                    "feature_set": feature_set,
                    "threshold": float(t),
                    "n_pred_pos": k,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pl.DataFrame(rows).sort(["cell_id", "threshold"], descending=[False, True])


def recommend_operating_point(
    sweep: pl.DataFrame,
    rule: Literal["max_f1", "min_recall"] = "max_f1",
    target_recall: float = 0.5,
) -> pl.DataFrame:
    """one recommended threshold per cell. `max_f1` picks the sweep row maximising F1.
    `min_recall` picks the highest threshold (== highest precision) whose recall still
    meets `target_recall`. the *policy* is an analysis choice; the artefact is `sweep`.
    """
    keys = ["cell_id", "model", "feature_set"]

    if rule == "max_f1":
        chosen = (
            sweep.sort(["cell_id", "f1"], descending=[False, True])
            .group_by("cell_id", maintain_order=True)
            .first()
        )
    elif rule == "min_recall":
        chosen = (
            sweep.filter(pl.col("recall") >= target_recall)
            .sort(["cell_id", "threshold"], descending=[False, True])
            .group_by("cell_id", maintain_order=True)
            .first()
        )
    else:
        raise ValueError(f"unknown rule {rule!r}")

    return chosen.with_columns(pl.lit(rule).alias("rule")).select(
        *keys, "rule", "threshold", "precision", "recall", "f1"
    )
