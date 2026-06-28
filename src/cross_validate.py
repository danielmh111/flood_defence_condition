from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
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
