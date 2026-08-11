from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, mean_absolute_error


@dataclass
class Predictions:
    label: np.ndarray
    proba: np.ndarray | None = None
    classes: np.ndarray | None = None
    score: np.ndarray | None = None


# need default functions for the args of the main function, lets the user specify a different method than .fit if using a non sklear api
def default_fit(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    model.fit(X, y)
    return model


def default_predict(model: Any, X: np.ndarray) -> Predictions:
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return Predictions(
        label=model.predict(X),
        proba=proba,
        classes=getattr(model, "classes_", None),
    )


DEFAULT_SCORERS = {
    "mae": lambda y, p: mean_absolute_error(y, p.label),
    "macro_f1": lambda y, p: f1_score(y, p.label, average="macro"),
}


def cross_validate(
    make_model: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    cv_splitter,
    groups: np.ndarray | None = None,
    scorers: dict | None = None,
    fit_fn: Callable[[Any, np.ndarray, np.ndarray], Any] = default_fit,
    predict_fn: Callable[[Any, np.ndarray], Any] = default_predict,
    compute_importance=None,
) -> dict:
    if scorers is None:
        scorers = DEFAULT_SCORERS

    fold_scores = {name: [] for name in scorers}
    importance_per_fold = []

    split_args = (X, y, groups) if groups is not None else (X, y)

    for train_idx, test_idx in cv_splitter.split(*split_args):
        model = fit_fn(make_model(), X[train_idx], y[train_idx])
        preds = predict_fn(model, X[test_idx])
        y_test = y[test_idx]

        for name, scorer in scorers.items():
            fold_scores[name].append(scorer(y_test, preds))

        if compute_importance is not None:
            importance_per_fold.append(compute_importance(model))

    results = {}
    for name, scores in fold_scores.items():
        results[f"{name}_mean"] = float(np.mean(scores))
        results[f"{name}_std"] = float(np.std(scores))
        results[f"{name}_per_fold"] = scores

    if importance_per_fold:
        results["importance_per_fold"] = importance_per_fold

    return results


def produce_spatial_blocks(
    coords: np.ndarray, n_blocks: int, seed: int = 42
) -> np.ndarray:
    return KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit_predict(coords)


# --- OOF harness (additive; nothing above this line is touched) ---

GRADES: tuple[int, ...] = (1, 2, 3, 4, 5)


def _grade_column_index(
    classes: np.ndarray, grades: Sequence[int] = GRADES
) -> dict[int, int]:
    """grade -> column index into a `classes`-ordered proba array. classes need not be sorted."""
    classes = np.asarray(classes)
    if classes.ndim != 1:
        raise ValueError(f"classes must be 1-D, got ndim={classes.ndim}")
    if not np.array_equal(classes, classes.astype(np.int64)):
        raise ValueError(f"classes must be integral grade labels, got {classes}")
    classes_int = classes.astype(np.int64)
    if np.unique(classes_int).size != classes_int.size:
        raise ValueError(f"duplicate classes: {classes_int.tolist()}")
    if not set(classes_int.tolist()) <= set(grades):
        raise ValueError(f"classes {classes_int.tolist()} outside grades {list(grades)}")
    return {int(c): j for j, c in enumerate(classes_int)}


def proba_to_grade_matrix(
    proba: np.ndarray,
    classes: np.ndarray,
    grades: Sequence[int] = GRADES,
    strict: bool = True,
) -> np.ndarray:
    """(n, len(classes)) proba ordered by `classes` -> (n, len(grades)) ordered by `grades`.

    a fitted model's classes_ need not span every grade or be sorted; mapping is by label,
    not position. absent grades are zero-filled. `strict=True` (default) raises instead of
    zero-filling, since a silently zero-filled grade corrupts any score built from it (e.g.
    pr_auc_ge4) without tripping a row-sum check.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D, got ndim={proba.ndim}")
    if classes is None:
        raise ValueError("classes is required to map proba columns to grades")

    col_of = _grade_column_index(classes, grades)
    if proba.shape[1] != len(col_of):
        raise ValueError(
            f"proba has {proba.shape[1]} columns but classes has {len(col_of)} entries"
        )
    if strict and set(col_of) != set(grades):
        raise ValueError(
            f"classes {sorted(col_of)} do not cover all grades {list(grades)} (strict=True)"
        )

    out = np.zeros((proba.shape[0], len(grades)), dtype=np.float64)
    for i, g in enumerate(grades):
        if g in col_of:
            out[:, i] = proba[:, col_of[g]]

    row_sums = out.sum(axis=1)
    max_err = np.abs(row_sums - 1.0).max() if row_sums.size else 0.0
    if max_err > 1e-6:
        worst = int(np.abs(row_sums - 1.0).argmax())
        raise ValueError(
            f"proba rows must sum to 1 (+/-1e-6); row {worst} sums to {row_sums[worst]!r}"
        )

    return out


def fold_ids_from_splitter(
    cv_splitter, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
) -> np.ndarray:
    """materialise each row's TEST fold id, once, so every cell in a grid shares fold labels."""
    n = len(y)
    fold_id = np.full(n, -1, dtype=np.int64)
    split_args = (X, y, groups) if groups is not None else (X, y)

    for k, (_, test_idx) in enumerate(cv_splitter.split(*split_args)):
        fold_id[test_idx] = k

    n_unassigned = int((fold_id < 0).sum())
    if n_unassigned:
        raise ValueError(f"{n_unassigned} rows were never assigned to a test fold")

    return fold_id.astype(np.int8)


ShapFn = Callable[[Any, np.ndarray, np.ndarray, int], pl.DataFrame]


def run_oof(
    make_model: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    asset_ids: np.ndarray,
    fit_fn: Callable[[Any, np.ndarray, np.ndarray], Any] = default_fit,
    predict_fn: Callable[[Any, np.ndarray], Predictions] = default_predict,
    shap_fn: ShapFn | None = None,
    grades: Sequence[int] = GRADES,
    strict_classes: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """out-of-fold predictions: fit on fold_id != k, predict on == k, for every k.

    every row is predicted exactly once (raises otherwise). no scoring happens here -
    the returned frame is the single artefact every downstream metric re-reads.
    """
    n = len(y)
    if len(fold_id) != n or len(asset_ids) != n:
        raise ValueError(
            f"length mismatch: y={n}, fold_id={len(fold_id)}, asset_ids={len(asset_ids)}"
        )

    n_grades = len(grades)
    proba_oof = np.full((n, n_grades), np.nan, dtype=np.float64)
    filled = np.zeros(n, dtype=bool)
    shap_frames: list[pl.DataFrame] = []

    for k in np.unique(fold_id):
        train, test = fold_id != k, fold_id == k

        model = fit_fn(make_model(), X[train], y[train])
        preds = predict_fn(model, X[test])

        if preds.proba is None or preds.classes is None:
            raise ValueError(
                f"fold {k}: predict_fn must return proba and classes for OOF emission"
            )

        proba_oof[test] = proba_to_grade_matrix(
            preds.proba, preds.classes, grades, strict_classes
        )

        if filled[test].any():
            raise ValueError(f"fold {k}: some rows already predicted by an earlier fold")
        filled[test] = True

        if shap_fn is not None:
            shap_frames.append(shap_fn(model, X[test], asset_ids[test], int(k)))

    n_unfilled = int((~filled).sum())
    if n_unfilled:
        raise ValueError(f"{n_unfilled} rows were never predicted (gap in fold_id)")

    grades_arr = np.asarray(grades)
    y_pred_argmax = grades_arr[proba_oof.argmax(axis=1)]

    data: dict[str, np.ndarray] = {
        "asset_id": np.asarray(asset_ids).astype(np.int64),
        "fold_id": np.asarray(fold_id).astype(np.int8),
        "y_true": np.asarray(y).astype(np.int8),
    }
    for i, g in enumerate(grades):
        data[f"proba_{g}"] = proba_oof[:, i]
    data["y_pred_argmax"] = y_pred_argmax.astype(np.int8)

    oof_df = pl.DataFrame(data)
    shap_df = pl.concat(shap_frames, how="vertical") if shap_frames else None

    return oof_df, shap_df
