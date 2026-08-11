from dataclasses import dataclass
from typing import Any, Callable, Sequence

import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer

from src.cross_validate import default_fit, default_predict
from src.frank_hall import CUTS, FrankHallClassifier

FEATURE_SETS = pl.Enum(["base", "base_exp"])
MODELS = pl.Enum(
    ["hgb_default", "hgb_qwk_es", "fh_hgb", "dummy_median", "dummy_stratified"]
)

QWK_SCORER = make_scorer(cohen_kappa_score, weights="quadratic")


@dataclass(frozen=True)
class CellSpec:
    cell_id: str  # f"{model_key}__{feature_set}"
    model_key: str
    feature_set: str
    make_model: Callable[[], Any]
    params: dict[str, Any]
    fit_fn: Callable = default_fit
    predict_fn: Callable = default_predict
    wants_shap: bool = False


def make_hgb(
    cat_idx: Sequence[int], seed: int, scoring: Any = None
) -> HistGradientBoostingClassifier:
    """single definition for the shared HGB config.

    every hgb_* and fh_hgb cell goes through this.
    `scoring=None` (default) leaves the HGB default ("loss") in place
    hgb_qwk_es overrides it.
    """

    kwargs: dict[str, Any] = dict(
        categorical_features=list(cat_idx),
        learning_rate=0.05,
        max_iter=400,
        early_stopping=True,
        random_state=seed,
        class_weight=None,
    )
    if scoring is not None:
        kwargs["scoring"] = scoring
    return HistGradientBoostingClassifier(**kwargs)


def make_fh(cat_idx: Sequence[int], seed: int) -> FrankHallClassifier:
    """Frank-Hall over four default early-stopping HGB models"""
    return FrankHallClassifier(base_factory=lambda: make_hgb(cat_idx, seed))


def build_grid(cat_idx: Sequence[int], seed: int) -> list[CellSpec]:
    """the 8 cells of the grid. dummies ignore X, so they run on `base` only."""
    cat_idx = list(cat_idx)

    def hgb_params(scoring_label: str = "loss") -> dict[str, Any]:
        return {
            "categorical_features": cat_idx,
            "learning_rate": 0.05,
            "max_iter": 400,
            "early_stopping": True,
            "random_state": seed,
            "class_weight": None,
            "scoring": scoring_label,
        }

    return [
        CellSpec(
            cell_id="hgb_default__base",
            model_key="hgb_default",
            feature_set="base",
            make_model=lambda: make_hgb(cat_idx, seed),
            params=hgb_params(),
            wants_shap=True,
        ),
        CellSpec(
            cell_id="hgb_default__base_exp",
            model_key="hgb_default",
            feature_set="base_exp",
            make_model=lambda: make_hgb(cat_idx, seed),
            params=hgb_params(),
            wants_shap=True,
        ),
        CellSpec(
            cell_id="hgb_qwk_es__base",
            model_key="hgb_qwk_es",
            feature_set="base",
            make_model=lambda: make_hgb(cat_idx, seed, scoring=QWK_SCORER),
            params=hgb_params("quadratic_weighted_kappa"),
            wants_shap=True,
        ),
        CellSpec(
            cell_id="hgb_qwk_es__base_exp",
            model_key="hgb_qwk_es",
            feature_set="base_exp",
            make_model=lambda: make_hgb(cat_idx, seed, scoring=QWK_SCORER),
            params=hgb_params("quadratic_weighted_kappa"),
            wants_shap=True,
        ),
        CellSpec(
            cell_id="fh_hgb__base",
            model_key="fh_hgb",
            feature_set="base",
            make_model=lambda: make_fh(cat_idx, seed),
            params={"sub_model": hgb_params(), "cuts": list(CUTS)},
            wants_shap=False,
        ),
        CellSpec(
            cell_id="fh_hgb__base_exp",
            model_key="fh_hgb",
            feature_set="base_exp",
            make_model=lambda: make_fh(cat_idx, seed),
            params={"sub_model": hgb_params(), "cuts": list(CUTS)},
            wants_shap=False,
        ),
        CellSpec(
            cell_id="dummy_median__base",
            model_key="dummy_median",
            feature_set="base",
            make_model=lambda: DummyClassifier(strategy="constant", constant=3),
            params={"strategy": "constant", "constant": 3},
            wants_shap=False,
        ),
        CellSpec(
            cell_id="dummy_stratified__base",
            model_key="dummy_stratified",
            feature_set="base",
            make_model=lambda: DummyClassifier(
                strategy="stratified", random_state=seed
            ),
            params={"strategy": "stratified", "random_state": seed},
            wants_shap=False,
        ),
    ]
