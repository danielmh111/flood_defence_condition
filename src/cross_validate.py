import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_recall_curve,
    auc,
)
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from functools import partial


DEFAULT_SCORERS = {
    "mae": mean_absolute_error,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "spearman": lambda y_true, y_pred: spearmanr(y_true, y_pred).statistic,  # type: ignore
    "f1_score": partial(f1_score, average="macro"),
}


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model,
    cv_splitter,
    groups: np.ndarray | None = None,
    extend_func=None,
    compute_importance=None,
    scorers: dict | None = None,
) -> dict:
    if scorers is None:
        scorers = DEFAULT_SCORERS

    fold_scores = {name: [] for name in scorers}
    importance_per_fold = []

    split_args = (X, y, groups) if groups is not None else (X, y)

    for train_idx, test_idx in cv_splitter.split(*split_args):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if extend_func is not None:
            X_train, X_test = extend_func(X_train, X_test, train_idx, test_idx)

        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)

        for name, scorer in scorers.items():
            fold_scores[name].append(scorer(y_test, y_pred))
            # todo some scorers take y_prob instead of y_pred. I need to find a way of routing the correct argument to each function, even if the scorers dict is user defined and passed in at runtime

        if compute_importance is not None:
            importance_per_fold.append(compute_importance(m))

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
