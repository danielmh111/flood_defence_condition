# the sampled hyperparameter grid plus the three unfitted reference predictors.
#
# flat design: every (feature_set, config) pair is an independent cell scored by one 10-fold
# spatially blocked pass, no inner loop. feature sets are compared at MATCHED configs (paired
# within-fold deltas), never by comparing per-set maxima, which would select on fold noise.
#
# learning_rate is fixed at 0.1 and early stopping is off. nb16's early stopping validated on a
# random 10% of each training fold, spatially interleaved with the training data, so its iteration
# count (147, with the 400 cap never binding) was chosen under a leaky criterion. max_iter is
# tuned explicitly instead, centred near the 75 that halving the step size implies but extending
# low, since the honest spatial optimum is likely below it.

import hashlib
import itertools
import json
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from src.cross_validate import GRADES

SEED = 123
N_CONFIGS = 50  # of 720 combinations

FIXED_PARAMS = {
    "learning_rate": 0.1,
    "early_stopping": False,
    "class_weight": None,
}

GRID = {
    "max_iter": [25, 50, 100, 200, 400],
    "max_leaf_nodes": [8, 16, 31, 64],
    "min_samples_leaf": [5, 20, 50, 100],
    "l2_regularization": [0.0, 0.1, 1.0],
    "max_features": [0.5, 0.7, 1.0],
}

# forced into the sample and present in every feature set, so cross-set comparison at a fixed
# hyperparameter setting is available and full per-row SHAP is like-for-like
REFERENCE_CONFIG = {
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "max_features": 0.7,
}


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    model: str
    feature_set: str | None
    params: dict[str, Any]
    make_model: Callable[[], Any]
    wants_full_shap: bool = False


def cell_id(feature_set: str | None, model: str, params: dict) -> str:
    """deterministic short id, so reruns are idempotent and an edited config produces a new cell
    rather than silently reusing stale output"""
    payload = json.dumps(
        {"feature_set": feature_set, "model": model, "params": params},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def sample_configs(n: int = N_CONFIGS, seed: int = SEED) -> list[dict[str, Any]]:
    """n distinct configs, reference config first. randomised rather than factorial: at 720
    combinations a coarse factorial of the same size would cover the space far worse."""
    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    if n > len(combos):
        raise ValueError(f"asked for {n} configs but the grid has only {len(combos)}")

    rest = [c for c in combos if c != REFERENCE_CONFIG]
    order = np.random.default_rng(seed).permutation(len(rest))
    return [dict(REFERENCE_CONFIG), *(rest[i] for i in order[: n - 1])]


class HeuristicNational:
    """the nb08 halcrow instantiation as an unfitted predictor. X is a single column carrying the
    precomputed grade (NaN where the asset is not mappable onto a curve), fit is a no-op, and
    predict_proba one-hots - so it costs one predict per fold and is scored on the same folds and
    metrics as every fitted cell. unmappable assets fall back to grade 3, which is what a
    practitioner without the inputs would do."""

    def __init__(self, fallback_grade: int = 3):
        self.classes_ = np.asarray(GRADES)
        self.fallback_grade = fallback_grade

    def fit(self, X, y):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError(f"HeuristicNational expects an (n, 1) X, got {X.shape}")

        pred = np.rint(np.nan_to_num(X[:, 0], nan=float(self.fallback_grade))).astype(
            np.int64
        )
        outside = sorted(set(pred.tolist()) - set(GRADES))
        if outside:
            raise ValueError(f"heuristic emitted grade(s) outside {GRADES}: {outside}")

        proba = np.zeros((len(pred), len(self.classes_)))
        proba[np.arange(len(pred)), self.classes_.searchsorted(pred)] = 1.0
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_cells(
    cat_idx_by_set: dict[str, list[int]],
    configs: list[dict[str, Any]],
    seed: int = SEED,
    include_heuristic: bool = False,
) -> list[CellSpec]:
    """fitted cells = feature sets x configs. the reference predictors take no hyperparameters, so
    they sit outside the grid: one cell each, not one per config."""
    cells = []

    for feature_set, cat_idx in cat_idx_by_set.items():
        for config in configs:
            params = {
                **FIXED_PARAMS,
                **config,
                "categorical_features": list(cat_idx),
                "random_state": seed,
            }
            cells.append(
                CellSpec(
                    cell_id=cell_id(feature_set, "hgb", params),
                    model="hgb",
                    feature_set=feature_set,
                    params=params,
                    make_model=partial(HistGradientBoostingClassifier, **params),
                    wants_full_shap=config == REFERENCE_CONFIG,
                )
            )

    references = [
        ("dummy_median", {"strategy": "constant", "constant": 3}, DummyClassifier),
        (
            "dummy_stratified",
            {"strategy": "stratified", "random_state": seed},
            DummyClassifier,
        ),
    ]
    if include_heuristic:
        references.append(
            ("heuristic_national", {"fallback_grade": 3}, HeuristicNational)
        )

    for model, params, cls in references:
        cells.append(
            CellSpec(
                cell_id=cell_id(None, model, params),
                model=model,
                feature_set=None,
                params=params,
                make_model=partial(cls, **params),
            )
        )

    ids = [c.cell_id for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("cell_id collision - two cells hash to the same id")

    return cells
