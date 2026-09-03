# the 1,440-config full factorial, the three unfitted reference predictors, and the frank-hall
# variant the second pass fits at a handful of selected configs.
#
# capacity = learning_rate * max_iter is the gridded axis and max_iter is derived from it, so the
# same effective capacity is matched exactly across learning rates. that ragged form is what made
# the fine grid's learning-rate finding readable - at matched capacity, lower lr won monotonically
# to the bottom of the range - and learning rate is the axis this experiment exists to resolve.
#
# max_depth is fixed at None rather than paired with max_leaf_nodes as it was in the fine grid.
# leaf-wise growth beat the balanced-growth constraint at every leaf count there (+0.0014 at 4
# leaves, +0.0023 at 8, +0.0012 at 16 pr-auc), so the pairing is reported from that experiment
# rather than re-tested.
#
# l2_regularization stays gridded. the fine grid's marginal spanned 0.001378 with 1.0 ahead in
# every slice, over the pre-committed ~0.001 threshold for reinstating it.
#
# max_bins is the new axis. at ~21k training rows per fold, 16/32/255 bins is ~1,300/650/82 rows
# per bin, so the range genuinely spans binning-as-regularisation. three levels rather than two
# because three axes in a row have now returned boundary optima and two can only detect an effect,
# not locate an interior one.
#
# max_features is fixed at 1.0, not gridded. it is a fraction, so at a matched config it subsamples
# a different absolute number of columns per composition (11 vs 12 wide here), which breaks the
# premise that the two arms are treated identically.

import hashlib
import itertools
import json
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from src.cross_validate import GRADES
from src.frank_hall import FrankHallClassifier

SEED = 123

CAPACITY = [1.6, 2.5, 4.0, 5.0, 6.0]
LEARNING_RATE = [0.00625, 0.0125, 0.025, 0.05]
MAX_LEAF_NODES = [4, 8, 16, 31]
MIN_SAMPLES_LEAF = [5, 10, 20]
MAX_BINS = [16, 32, 255]
L2 = [0.0, 1.0]

N_CONFIGS = (
    len(CAPACITY)
    * len(LEARNING_RATE)
    * len(MAX_LEAF_NODES)
    * len(MIN_SAMPLES_LEAF)
    * len(MAX_BINS)
    * len(L2)
)

FIXED_PARAMS = {
    "max_depth": None,
    "max_features": 1.0,
    "early_stopping": False,
    "class_weight": None,
    "random_state": SEED,
}

# this exact config exists in the fine grid too, at both of these compositions - capacity 2.5 over
# lr 0.025 is max_iter 100, and every other key matches. the sklearn param dict therefore hashes to
# the same cell_id in both experiments, and since the fits are deterministic and single-threaded,
# the oof should reproduce bit for bit. that makes it a free regression test on the whole pipeline
# rather than merely a shared anchor.
REFERENCE_CONFIG = {
    "capacity": 2.5,
    "learning_rate": 0.025,
    "max_leaf_nodes": 8,
    "min_samples_leaf": 20,
    "max_bins": 255,
    "l2_regularization": 0.0,
}

GRID_AXES = tuple(REFERENCE_CONFIG)


@dataclass(frozen=True)
class CellSpec:
    """picklable and dumb - it crosses a spawn boundary, so it carries no callable and no matrix.
    the worker constructs the estimator from `params` itself.

    `shap` is a tier rather than a flag because the main grid wants shap_agg on the reference config
    without the per-row artefact, and only the final pass wants both. `est_cost` is what the parent
    sorts the submission queue by."""

    cell_id: str
    model: str
    composition: str | None
    params: dict[str, Any]
    capacity: float | None = None
    shap: str | None = None  # None | "agg" | "full"
    est_cost: float = 0.0


def cell_id(composition: str | None, model: str, params: dict) -> str:
    """deterministic short id, so reruns are idempotent and an edited config produces a new cell
    rather than silently reusing stale output. capacity is a design annotation, not a constructor
    argument, and is deliberately absent from `params` - within a learning rate capacity -> max_iter
    is injective and across learning rates the lr key differs, so this stays collision-free. model
    is in the payload, which is what keeps a frank-hall cell off its hgb twin at identical params."""
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
            "min_samples_leaf": min_samples,
            "max_bins": bins,
            "l2_regularization": l2,
        }
        for capacity, lr, leaves, min_samples, bins, l2 in itertools.product(
            CAPACITY, LEARNING_RATE, MAX_LEAF_NODES, MIN_SAMPLES_LEAF, MAX_BINS, L2
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

# four binary models per cell, so roughly 4x the cost of the hgb cell at the same params
FRANK_HALL_COST_FACTOR = 4


def make_model(spec: CellSpec):
    """called in the worker. frank-hall is built here rather than shipped as a partial, because a
    partial over an sklearn class must not cross the spawn boundary."""
    if spec.model == "frank_hall":
        return FrankHallClassifier(
            base_factory=partial(HistGradientBoostingClassifier, **spec.params)
        )
    return MODELS[spec.model](**spec.params)


def frank_hall_diagnostics(model, X_test, asset_ids, fold: int) -> pl.DataFrame:
    """run_oof calls its shap_fn after predict_fn, so by here the classifier's inversion counters
    describe exactly this fold's test rows. borrowing that slot harvests them per fold without
    touching src/cross_validate.py, and frank-hall gets no shap anyway - src/oof_shap.py assumes one
    model emitting a 3-D multiclass array, and four binary models need four explainers.

    a high inversion rate means the four cut models disagree about monotonicity and the clamped
    reconstruction is doing real work. it is the diagnostic that says whether the decomposition is
    behaving."""
    return pl.DataFrame(
        {
            "fold_id": [fold],
            "inversion_rate": [float(model.inversion_rate_)],
            "n_inverted": [int(model.n_inverted_)],
            "n_predicted": [int(model.n_predicted_)],
        }
    )


def build_cells(
    cat_idx_by_composition: dict[str, list[int]],
    configs: list[dict[str, Any]],
    model: str = "hgb",
    shap_for: dict[str, Any] | None = None,
    shap_tier: str = "agg",
) -> list[CellSpec]:
    """compositions x configs. `shap_for` names the one config that gets explained - the fine grid
    explained all 8,820 cells and produced the bump plots and rank-stability analysis already, so
    here it is two cells on the main grid and two on the final pass."""
    factor = FRANK_HALL_COST_FACTOR if model == "frank_hall" else 1
    cells = []

    for composition, cat_idx in cat_idx_by_composition.items():
        for config in configs:
            params = sklearn_params(config, cat_idx)
            cells.append(
                CellSpec(
                    cell_id=cell_id(composition, model, params),
                    model=model,
                    composition=composition,
                    params=params,
                    capacity=config["capacity"],
                    shap=shap_tier if config == shap_for else None,
                    est_cost=factor * params["max_iter"] * params["max_leaf_nodes"],
                )
            )

    return cells


def reference_cells(include_heuristic: bool) -> list[CellSpec]:
    """the unfitted predictors take no hyperparameters, so they sit outside the grid: one cell each,
    not one per config."""
    references = [
        ("dummy_median", {"strategy": "constant", "constant": 3}),
        ("dummy_stratified", {"strategy": "stratified", "random_state": SEED}),
    ]
    if include_heuristic:
        references.append(("heuristic_national", {"fallback_grade": 3}))

    return [
        CellSpec(cell_id=cell_id(None, model, params), model=model, composition=None, params=params)
        for model, params in references
    ]


def check_unique(cells: list[CellSpec]) -> list[CellSpec]:
    ids = [c.cell_id for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("cell_id collision - two cells hash to the same id")
    return cells
