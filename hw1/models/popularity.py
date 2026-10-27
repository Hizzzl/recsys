"""Fixed popularity baselines used for post-hoc experiment diagnostics."""

from typing import TYPE_CHECKING

import numpy as np

from ._als import check_float, validate_training_data, validate_users

if TYPE_CHECKING:
    import pandas as pd


class Popularity:
    """Count positive reviews, optionally decaying their weight with age.

    An observed rating of at least four contributes one, or
    ``2 ** (-age_days / half_life_days)`` when a half-life is supplied.
    Scores are shared across users; the common evaluator masks each user's
    known items. Neither baseline has parameters selected on held-out scores.
    """

    def __init__(self, half_life_days: float | None = None):
        self.half_life_days = (
            None if half_life_days is None else
            check_float("half_life_days", half_life_days, strictly_positive=True)
        )

    def fit(self, train: "pd.DataFrame", n_users: int, n_items: int,
            cutoff_ms: int) -> "Popularity":
        _, items, ratings, timestamps = validate_training_data(
            train, n_users, n_items, cutoff_ms)
        positive = ratings >= 4
        weights = np.ones(int(positive.sum()), dtype=np.float64)
        if self.half_life_days is not None:
            age_days = (float(cutoff_ms) - timestamps[positive]) / 86_400_000.0
            weights = np.exp2(-age_days / self.half_life_days)
        self.item_scores_ = np.bincount(
            items[positive], weights=weights, minlength=n_items).astype(np.float64)
        self.n_users_, self.n_items_ = n_users, n_items
        self.cutoff_ms_ = cutoff_ms
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        """Return writable dense scores without changing or masking the model."""
        if not hasattr(self, "item_scores_"):
            raise RuntimeError("fit must be called before score")
        users = validate_users(users, self.n_users_)
        return np.broadcast_to(
            self.item_scores_, (len(users), self.n_items_)).copy()
