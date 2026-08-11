from typing import Any, Callable, Sequence

import numpy as np

from src.cross_validate import GRADES

CUTS: tuple[int, ...] = (1, 2, 3, 4)


def reconstruct(cumulative: np.ndarray) -> np.ndarray:
    """
    the four boundaries probablities ( P(y>1), ..., P(y>4) ) from independent models combined
    to calculate class probabilities for all 5 classes
    """

    cumulative = np.asarray(cumulative, dtype=np.float64)
    if cumulative.ndim != 2 or cumulative.shape[1] != 4:
        raise ValueError(f"cumulative must be (n, 4), got shape {cumulative.shape}")

    raw = np.empty((cumulative.shape[0], 5), dtype=np.float64)
    raw[:, 0] = 1.0 - cumulative[:, 0]
    raw[:, 1:4] = cumulative[:, :3] - cumulative[:, 1:]
    raw[:, 4] = cumulative[:, 3]
    return raw


def clamp_renormalise(
    raw: np.ndarray, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """negative entries clamped to 0, rows renormalised to sum to 1.

    the pre clamp row sum is exactly 1 by construction, and clamping only ever raises entries.
    so the post clamp sum is always >= 1. This is normalised so that the total probability of being any grade is exactly 1.
    """

    raw = np.asarray(raw, dtype=np.float64)
    inverted = (raw < -tol).any(axis=1)

    clamped = np.clip(raw, 0.0, None)
    row_sums = clamped.sum(axis=1)

    if (row_sums < 1.0 - 1e-9).any():
        worst = int(row_sums.argmin())
        raise ValueError(
            f"clamped row sum below 1 at row {worst} ({row_sums[worst]!r}) - this should be impossible"
        )

    proba = clamped / row_sums[:, None]
    return proba, inverted


class FrankHallClassifier:
    """
    Frank-Hall decomposition
    four independent binary models,
    each model k predicting P(y > k) - the boundaries between classes, not the class value itself.
    Reconstructs the 5 class distribution using the differences between these e.g. P(y=3) = P(y>3) - P(y>2).

    The models are independant so the four boundary probabilities are not guarenteed to increase, and so its possible for P(y=3) = P(y>3) - P(y>2) to end up being negative.
    This is obviously wrong, so any negative entries are clipped to 0, then probabilities are renormalised.


    Exposes `predict_proba`/`classes_` so it drops into `default_fit`/`default_predict` (and
    therefore `run_oof`) unchanged.
    """

    def __init__(
        self,
        base_factory: Callable[[], Any],
        grades: Sequence[int] = GRADES,
        tol: float = 1e-12,
    ) -> None:
        if len(grades) != 5:
            raise ValueError(
                f"Frank Hall is defined for exactly 5 grades, got {list(grades)}"
            )

        self.base_factory = base_factory
        self.grades = tuple(grades)
        self.tol = tol

        self.classes_ = np.asarray(self.grades)
        self.models_: list[Any] = []
        self.pos_col_: list[int] = []

        # diagnostics from the most recent predict_proba call
        self.inversion_rate_: float = float("nan")
        self.n_inverted_: int = 0
        self.n_predicted_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FrankHallClassifier":
        y = np.asarray(y)
        self.models_ = []
        self.pos_col_ = []

        for k in CUTS:
            yk = (y > k).astype(np.int8)
            n_pos = int(yk.sum())
            if n_pos == 0 or n_pos == yk.size:
                raise ValueError(
                    f"cut k={k} is degenerate: {n_pos}/{yk.size} positives (y > {k})"
                )

            model = self.base_factory()
            model.fit(X, yk)

            pos_matches = np.flatnonzero(model.classes_ == 1)
            if pos_matches.size != 1:
                raise ValueError(
                    f"cut k={k}: expected binary classes_ containing exactly one `1`, got {model.classes_}"
                )

            self.models_.append(model)
            self.pos_col_.append(int(pos_matches[0]))

        return self

    def cumulative_proba(self, X: np.ndarray) -> np.ndarray:
        """
        raw P(y > k) for k in cuts
        column 2 (k=3) is the clf3 cross-check for the >=4 score
        """

        if not self.models_:
            raise ValueError("FrankHallClassifier is not fitted")

        cols = [
            model.predict_proba(X)[:, pos_col]
            for model, pos_col in zip(self.models_, self.pos_col_)
        ]
        return np.column_stack(cols)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cumulative = self.cumulative_proba(X)
        raw = reconstruct(cumulative)
        proba, inverted = clamp_renormalise(raw, tol=self.tol)

        self.n_predicted_ = int(proba.shape[0])
        self.n_inverted_ = int(inverted.sum())
        self.inversion_rate_ = (
            self.n_inverted_ / self.n_predicted_ if self.n_predicted_ else float("nan")
        )

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]
