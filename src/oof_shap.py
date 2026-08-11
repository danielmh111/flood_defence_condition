from typing import Sequence

import numpy as np
import polars as pl
import shap

from src.cross_validate import GRADES, ShapFn, _grade_column_index


def tree_shap_frame(
    model,
    X_test: np.ndarray,
    asset_ids: np.ndarray,
    fold: int,
    *,
    feature_names: Sequence[str],
    grades: Sequence[int] = GRADES,
    check_additivity: bool = True,
) -> pl.DataFrame:
    """TreeExplainer SHAP for this fold's test rows, flattened to
    shap__{feature}__grade{g} + base_value__grade{g} wide columns.

    class axis is mapped by model.classes_, not position, and any grade absent from
    classes_ is zero-filled so every fold's frame has an identical schema - required
    for the vertical concat across a cell's 10 folds.
    """
    n = X_test.shape[0]
    n_features = len(feature_names)

    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("model has no classes_; TreeExplainer needs a fitted classifier")
    col_of = _grade_column_index(classes, grades)

    explanation = shap.TreeExplainer(model)(X_test, check_additivity=check_additivity)
    values = np.asarray(explanation.values)

    if values.ndim != 3:
        raise ValueError(
            f"expected 3-D shap values (n, n_features, n_classes), got ndim={values.ndim}"
        )
    if values.shape[0] != n:
        raise ValueError(f"shap values row count {values.shape[0]} != X_test row count {n}")
    if values.shape[1] != n_features:
        raise ValueError(
            f"shap values feature dim {values.shape[1]} != len(feature_names) {n_features}"
        )

    n_cls = values.shape[2]
    bv = np.asarray(explanation.base_values, dtype=np.float64)
    if bv.ndim == 1:
        if bv.shape != (n_cls,):
            raise ValueError(f"1-D base_values shape {bv.shape} != (n_classes,) ({n_cls},)")
        bv = np.broadcast_to(bv, (n, n_cls))
    elif bv.ndim == 2:
        if bv.shape == (1, n_cls):
            bv = np.broadcast_to(bv, (n, n_cls))
        elif bv.shape != (n, n_cls):
            raise ValueError(f"2-D base_values shape {bv.shape} != (n, n_classes) ({n}, {n_cls})")
    else:
        raise ValueError(f"unexpected base_values ndim {bv.ndim}")
    bv = np.ascontiguousarray(bv)  # np.broadcast_to returns a read-only view

    if len(asset_ids) != n:
        raise ValueError(f"asset_ids length {len(asset_ids)} != X_test row count {n}")

    data: dict[str, np.ndarray] = {
        "asset_id": np.asarray(asset_ids).astype(np.int64),
        "fold_id": np.full(n, fold, dtype=np.int8),
    }
    zeros = np.zeros(n, dtype=np.float64)
    for f, name in enumerate(feature_names):
        for g in grades:
            j = col_of.get(g)
            data[f"shap__{name}__grade{g}"] = values[:, f, j] if j is not None else zeros
    for g in grades:
        j = col_of.get(g)
        data[f"base_value__grade{g}"] = bv[:, j] if j is not None else zeros

    return pl.DataFrame(data)


def make_tree_shap_fn(
    feature_names: Sequence[str],
    grades: Sequence[int] = GRADES,
    max_rows_per_fold: int | None = None,
) -> ShapFn:
    """binds feature_names (and an optional smoke-test row cap) into run_oof's 4-positional
    (model, X_test, asset_ids, fold) -> pl.DataFrame shap_fn signature.
    """
    names = list(feature_names)

    def shap_fn(model, X_test: np.ndarray, asset_ids: np.ndarray, fold: int) -> pl.DataFrame:
        if max_rows_per_fold is not None:
            X_test = X_test[:max_rows_per_fold]
            asset_ids = asset_ids[:max_rows_per_fold]
        return tree_shap_frame(
            model, X_test, asset_ids, fold, feature_names=names, grades=grades
        )

    return shap_fn
