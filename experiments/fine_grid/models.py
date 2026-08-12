# the 630-config full factorial plus the three unfitted reference predictors.
#
# capacity = learning_rate * max_iter is the gridded axis and max_iter is derived from it, so the
# same effective capacity is matched exactly across learning rates. a rectangular lr x max_iter
# cross would spend most of its cells on duplicate capacities at very different costs; the ragged
# form turns learning_rate into an answerable question - at matched capacity, does the shrinkage
# route matter?
#
# max_depth is paired with max_leaf_nodes rather than crossed: max_depth 3 permits 8 leaves, so it
# is a no-op at 2, 4 or 8 leaves. the seven pairs are the non-redundant set - per leaf count either
# None (leaf-wise, lopsided growth permitted) or log2(max_leaf_nodes) (balanced growth forced).
#
# max_features is fixed at 1.0, not gridded. it is a fraction, so at a matched config it subsamples
# a different absolute number of columns per composition (0.7 is 5 of 7 for identity, 9 of 13 for
# the widest arm) - which breaks the premise that compositions are treated identically. it was also
# flat in the predecessor experiment.

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from src.cross_validate import GRADES

SEED = 123

CAPACITY = [0.4, 0.8, 1.6, 2.5, 5.0]
LEARNING_RATE = [0.025, 0.05, 0.1]
LEAF_DEPTH = [(2, None), (4, None), (4, 2), (8, None), (8, 3), (16, None), (16, 4)]
L2 = [0.0, 1.0]
MIN_SAMPLES_LEAF = [1, 5, 20]

N_CONFIGS = len(CAPACITY) * len(LEARNING_RATE) * len(LEAF_DEPTH) * len(L2) * len(MIN_SAMPLES_LEAF)

# max_bins is written explicitly at its default so it enters the dict cell_id hashes. if it is ever
# promoted to a grid axis, every 255-cell hashes identically to what is already on disk and resume
# skips it, instead of rehashing all 8,820 cells.
FIXED_PARAMS = {
    "max_features": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "class_weight": None,
    "random_state": SEED,
}

# the predecessor's best corner (lr 0.1, max_iter 25) plus this experiment's new defaults, with
# three capacity levels below it. forced to sort first and flagged for full per-row shap on every
# arm - its presence in all fourteen is what makes cross-composition shap comparison like-for-like.
REFERENCE_CONFIG = {
    "capacity": 2.5,
    "learning_rate": 0.1,
    "max_leaf_nodes": 8,
    "max_depth": None,
    "l2_regularization": 0.0,
    "min_samples_leaf": 20,
}

GRID_AXES = tuple(REFERENCE_CONFIG)


@dataclass(frozen=True)
class CellSpec:
    """picklable and dumb - it crosses a spawn boundary, so it carries no callable and no matrix.
    the worker constructs the estimator from `params` itself."""

    cell_id: str
    model: str
    composition: str | None
    params: dict[str, Any]
    capacity: float | None = None
    wants_full_shap: bool = False


def cell_id(composition: str | None, model: str, params: dict) -> str:
    """deterministic short id, so reruns are idempotent and an edited config produces a new cell
    rather than silently reusing stale output. capacity is a design annotation, not a constructor
    argument, and is deliberately absent from `params` - within a learning rate capacity -> max_iter
    is injective and across learning rates the lr key differs, so this stays collision-free."""
    payload = json.dumps(
        {"composition": composition, "model": model, "params": params},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def derive_max_iter(capacity: float, learning_rate: float) -> int:
    """assert exactness rather than round quietly, so a future edit to either axis fails loud"""
    exact = capacity / learning_rate
    if abs(exact - round(exact)) > 1e-9:
        raise ValueError(f"capacity {capacity} / lr {learning_rate} = {exact} is not an integer")
    max_iter = int(round(exact))
    if max_iter < 1:
        raise ValueError(f"capacity {capacity} / lr {learning_rate} gives max_iter {max_iter}")
    return max_iter


def build_configs() -> list[dict[str, Any]]:
    """full factorial, reference config first. no sampling, no seed."""
    configs = [
        {
            "capacity": capacity,
            "learning_rate": lr,
            "max_leaf_nodes": leaves,
            "max_depth": depth,
            "l2_regularization": l2,
            "min_samples_leaf": min_samples,
        }
        for capacity, lr, (leaves, depth), l2, min_samples in itertools.product(
            CAPACITY, LEARNING_RATE, LEAF_DEPTH, L2, MIN_SAMPLES_LEAF
        )
    ]
    if len(configs) != N_CONFIGS:
        raise ValueError(f"built {len(configs)} configs, expected {N_CONFIGS}")
    if REFERENCE_CONFIG not in configs:
        raise ValueError("reference config is not a member of the grid")

    for config in configs:
        derive_max_iter(config["capacity"], config["learning_rate"])

    return [dict(REFERENCE_CONFIG), *(c for c in configs if c != REFERENCE_CONFIG)]


def sklearn_params(config: dict[str, Any], cat_idx: list[int]) -> dict[str, Any]:
    tuned = {k: v for k, v in config.items() if k != "capacity"}
    return {
        **FIXED_PARAMS,
        **tuned,
        "max_iter": derive_max_iter(config["capacity"], config["learning_rate"]),
        "categorical_features": list(cat_idx),
    }


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

        pred = np.rint(np.nan_to_num(X[:, 0], nan=float(self.fallback_grade))).astype(np.int64)
        outside = sorted(set(pred.tolist()) - set(GRADES))
        if outside:
            raise ValueError(f"heuristic emitted grade(s) outside {GRADES}: {outside}")

        proba = np.zeros((len(pred), len(self.classes_)))
        proba[np.arange(len(pred)), self.classes_.searchsorted(pred)] = 1.0
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


MODELS = {
    "hgb": HistGradientBoostingClassifier,
    "dummy_median": DummyClassifier,
    "dummy_stratified": DummyClassifier,
    "heuristic_national": HeuristicNational,
}


def make_model(spec: CellSpec):
    return MODELS[spec.model](**spec.params)


def build_cells(
    cat_idx_by_composition: dict[str, list[int]],
    configs: list[dict[str, Any]],
    include_heuristic: bool = False,
) -> list[CellSpec]:
    """fitted cells = compositions x configs. the reference predictors take no hyperparameters, so
    they sit outside the grid: one cell each, not one per config."""
    cells = []

    for composition, cat_idx in cat_idx_by_composition.items():
        for config in configs:
            params = sklearn_params(config, cat_idx)
            cells.append(
                CellSpec(
                    cell_id=cell_id(composition, "hgb", params),
                    model="hgb",
                    composition=composition,
                    params=params,
                    capacity=config["capacity"],
                    wants_full_shap=config == REFERENCE_CONFIG,
                )
            )

    references = [
        ("dummy_median", {"strategy": "constant", "constant": 3}),
        ("dummy_stratified", {"strategy": "stratified", "random_state": SEED}),
    ]
    if include_heuristic:
        references.append(("heuristic_national", {"fallback_grade": 3}))

    for model, params in references:
        cells.append(
            CellSpec(cell_id=cell_id(None, model, params), model=model, composition=None, params=params)
        )

    ids = [c.cell_id for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("cell_id collision - two cells hash to the same id")

    return cells
