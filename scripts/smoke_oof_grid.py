# hand-run smoke/inspect script for the oof_grid experiment (no pytest in this project -
# see plan L4). asserts here are fine: this is a call-site script, not library code (L3).

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.cross_validate import GRADES, proba_to_grade_matrix
from src.frank_hall import FrankHallClassifier, clamp_renormalise, reconstruct
from src.oof_shap import tree_shap_frame


def check_proba_to_grade_matrix():
    out = proba_to_grade_matrix(np.array([[0.1, 0.2, 0.7]]), np.array([3, 1, 2]), grades=(1, 2, 3))
    assert np.allclose(out, [[0.2, 0.7, 0.1]]), "unsorted classes_ mapped incorrectly"

    out = proba_to_grade_matrix(
        np.array([[0.6, 0.4]]), np.array([1, 3]), grades=(1, 2, 3), strict=False
    )
    assert np.allclose(out, [[0.6, 0.0, 0.4]]) and np.isclose(out.sum(), 1.0), (
        "lenient zero-fill of a missing class failed"
    )

    try:
        proba_to_grade_matrix(np.array([[0.6, 0.4]]), np.array([1, 3]), grades=(1, 2, 3), strict=True)
        raise AssertionError("expected ValueError for a missing class under strict=True")
    except ValueError:
        pass

    out = proba_to_grade_matrix(np.array([[0, 0, 1, 0, 0]], dtype=float), np.array(GRADES))
    assert np.allclose(out, [[0, 0, 1, 0, 0]]), "dummy one-hot proba mishandled"

    try:
        proba_to_grade_matrix(np.array([[0.1, 0.1, 0.1]]), np.array([1, 2, 3]), grades=(1, 2, 3))
        raise AssertionError("expected ValueError for a bad row sum")
    except ValueError:
        pass

    print("proba_to_grade_matrix: ok")


def check_frank_hall():
    cum = np.array([[0.8, 0.5, 0.2, 0.05]])
    raw = reconstruct(cum)
    assert np.allclose(raw, [[0.2, 0.3, 0.3, 0.15, 0.05]]), "monotone reconstruction wrong"
    assert np.isclose(raw[0, 3] + raw[0, 4], cum[0, 2]), "telescoping identity failed (monotone)"

    proba, inverted = clamp_renormalise(raw)
    assert not inverted[0] and np.isclose(proba.sum(), 1.0) and np.allclose(proba, raw), (
        "clamp should be a no-op on an already-monotone vector"
    )

    cum_inv = np.array([[0.8, 0.3, 0.5, 0.05]])  # P(>2) < P(>3): an inversion
    raw_inv = reconstruct(cum_inv)
    assert raw_inv[0, 2] < 0, "expected a negative pre-clamp entry"
    assert np.isclose(raw_inv[0, 3] + raw_inv[0, 4], cum_inv[0, 2]), (
        "telescoping identity failed (inverted case)"
    )

    proba_inv, inverted_inv = clamp_renormalise(raw_inv)
    assert inverted_inv[0] and np.isclose(proba_inv.sum(), 1.0) and (proba_inv >= 0).all(), (
        "clamp/renorm failed on the inverted vector"
    )

    y_degenerate = np.array([1] * 10)
    X_degenerate = np.random.rand(10, 2)
    fh = FrankHallClassifier(base_factory=lambda: HistGradientBoostingClassifier())
    try:
        fh.fit(X_degenerate, y_degenerate)
        raise AssertionError("expected ValueError for a degenerate cut")
    except ValueError:
        pass

    rng = np.random.default_rng(0)
    n = 400
    X = rng.random((n, 4))
    y = np.clip((X[:, 0] * 5 + rng.standard_normal(n) * 0.7).astype(int) + 1, 1, 5)
    fh = FrankHallClassifier(base_factory=lambda: HistGradientBoostingClassifier(max_iter=50, random_state=1))
    fh.fit(X, y)
    proba = fh.predict_proba(X)
    assert proba.shape == (n, 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert 0.0 <= fh.inversion_rate_ <= 1.0

    print(f"frank_hall: ok (inversion_rate_={fh.inversion_rate_:.3f})")


def check_oof_shap():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.random((n, 4))
    y = np.clip((X[:, 0] * 5 + rng.standard_normal(n) * 0.7).astype(int) + 1, 1, 5)
    model = HistGradientBoostingClassifier(max_iter=50, random_state=1).fit(X, y)

    names = ["f0", "f1", "f2", "f3"]
    df = tree_shap_frame(model, X[:50], np.arange(50), fold=0, feature_names=names)
    expected_cols = 2 + len(names) * 5 + 5
    assert df.shape == (50, expected_cols), f"expected shape (50, {expected_cols}), got {df.shape}"
    assert df.columns[:4] == ["asset_id", "fold_id", "shap__f0__grade1", "shap__f0__grade2"], (
        "column order should be asset_id, fold_id, then feature-outer/grade-inner"
    )

    print(f"oof_shap: ok ({df.shape[1]} columns for {len(names)} features + asset_id/fold_id)")


def main():
    check_proba_to_grade_matrix()
    check_frank_hall()
    check_oof_shap()

    print("\nrunning end-to-end smoke run (2 folds, SHAP capped at 50 rows/fold)...")
    from experiments.oof_grid.runner import main as run_oof_grid

    run_oof_grid(smoke=True)
    print("\nsmoke run complete")


if __name__ == "__main__":
    main()
