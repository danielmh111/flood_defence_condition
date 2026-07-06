import numpy as np
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    recall_score,
    precision_score,
)
from functools import partial

from src.cross_validate import Predictions

GRADES = [1, 2, 3, 4, 5]


# def cohen_kappa(y, p):
#     # (p_o - p_e) / (1 - p_e)


def quad_weighted_cohen_kappa(y, predictions: Predictions):
    return float(
        cohen_kappa_score(y, predictions.label, labels=GRADES, weights="quadratic")
    )


def _recall_scorer(y, predictions: Predictions, grade):
    y = np.asarray(y)

    if not np.any(y == grade):
        return float("nan")  # if class absent in this test fold then score is undefined

    return float(
        recall_score(
            y, predictions.label, labels=[grade], average="macro", zero_division=0
        )
    )


def recall_for_grade(grade):
    return partial(_recall_scorer, grade=grade)


def _precision_scorer(y, predictions: Predictions, grade):
    y = np.asarray(y)

    if not np.any(y == grade):
        return float("nan")  # if class absent in this test fold then score is undefined

    return float(
        precision_score(
            y, predictions.label, labels=[grade], average="macro", zero_division=0
        )
    )


def precision_for_grade(grade):
    return partial(_precision_scorer, grade=grade)


def pr_auc_ge4(y, predictions: Predictions):
    """
    average precision for detecting grade >= 4.
    predicting these grades is more important than predicting the median (3) or the assets in good condition
    """

    y = np.asarray(y)
    y_bin = (y >= 4).astype(int)

    if y_bin.sum() == 0 or predictions.proba is None or predictions.classes is None:
        return float("nan")

    classes = np.asarray(predictions.classes)
    cols = [i for i, c in enumerate(classes) if c >= 4]

    if not cols:
        return float("nan")

    score = np.asarray(predictions.proba)[:, cols].sum(axis=1)

    return float(average_precision_score(y_bin, score))


def build_scorers() -> dict:
    scorers = {
        "mae": lambda y, predictions: mean_absolute_error(y, predictions.label),
        "macro_f1": lambda y, predictions: f1_score(
            y, predictions.label, average="macro"
        ),
        "qwk": quad_weighted_cohen_kappa,
        "pr_auc_ge4": pr_auc_ge4,
        **{f"recall_for_grade_{grade}": recall_for_grade(grade) for grade in GRADES},
        **{
            f"precision_for_grade_{grade}": precision_for_grade(grade)
            for grade in GRADES
        },
    }

    return scorers


def report_means(results: dict) -> dict:
    """reaggregation of cross_validate() output, cleaning nans"""
    out = {}
    for key, val in results.items():
        if key.endswith("_per_fold"):
            name = key[: -len("_per_fold")]
            arr = np.asarray(val, dtype=float)
            out[f"{name}_mean"] = float(np.nanmean(arr)) if arr.size else float("nan")
            out[f"{name}_std"] = float(np.nanstd(arr)) if arr.size else float("nan")
            out[f"{name}_n_folds"] = int(np.sum(~np.isnan(arr)))
    return out
