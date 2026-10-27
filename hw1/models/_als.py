"""Shared validation and a from-scratch weighted alternating least-squares solver.

There is one record per observed user-item pair. Explicit zero targets remain
stored in the sparse matrix: an observed zero has confidence 1 + alpha, while
an unobserved zero has confidence 1.
"""

from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

if TYPE_CHECKING:
    import pandas as pd


def check_integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def check_float(name: str, value: float, minimum: float = 0.0,
                strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if value < minimum or (strictly_positive and value <= minimum):
        comparator = ">" if strictly_positive else ">="
        raise ValueError(f"{name} must be {comparator} {minimum}")
    return float(value)


def validate_training_data(
    train: "pd.DataFrame", n_users: int, n_items: int, cutoff_ms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate dimensions, finite ratings/timestamps and unique observed pairs.

    IDs must already be encoded into [0, n_users) and [0, n_items). All
    observations must have existed by cutoff_ms. Cold rows/columns are allowed.
    """
    check_integer("n_users", n_users)
    check_integer("n_items", n_items)
    cutoff = check_float("cutoff_ms", cutoff_ms)
    required = {"user_idx", "item_idx", "rating", "timestamp"}
    if not hasattr(train, "columns") or not required.issubset(train.columns):
        raise ValueError(f"train must have columns {sorted(required)}")
    if len(train) == 0:
        raise ValueError("train must contain at least one observed interaction")
    users = np.asarray(train["user_idx"])
    items = np.asarray(train["item_idx"])
    for name, values, size in (("user_idx", users, n_users), ("item_idx", items, n_items)):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must contain integer IDs")
        if np.any(values < 0) or np.any(values >= size):
            raise ValueError(f"{name} contains an ID outside [0, {size})")
    ratings = np.asarray(train["rating"], dtype=np.float64)
    timestamps = np.asarray(train["timestamp"], dtype=np.float64)
    if not np.all(np.isfinite(ratings)) or np.any((ratings < 1) | (ratings > 5)):
        raise ValueError("rating must be finite and in [1, 5]")
    if not np.all(np.isfinite(timestamps)) or np.any(timestamps > cutoff):
        raise ValueError("timestamps must be finite and no later than cutoff_ms")
    # Structured pairs avoid integer-key overflow and do not depend on sort order.
    pairs = np.rec.fromarrays([users, items], names="user,item")
    if np.unique(pairs).size != len(train):
        raise ValueError("train must contain unique user-item pairs; deduplicate first")
    return (users.astype(np.int64, copy=False), items.astype(np.int64, copy=False),
            ratings, timestamps)


def validate_users(users: np.ndarray, n_users: int) -> np.ndarray:
    users = np.asarray(users)
    if users.ndim != 1 or not np.issubdtype(users.dtype, np.integer):
        raise ValueError("users must be a one-dimensional integer array")
    if np.any(users < 0) or np.any(users >= n_users):
        raise ValueError(f"users contains an ID outside [0, {n_users})")
    return users.astype(np.int64, copy=False)


def positive_matrix(users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
                    n_users: int, n_items: int) -> sparse.csr_matrix:
    positive = ratings >= 4
    return sparse.csr_matrix(
        (np.ones(positive.sum(), dtype=np.float32), (users[positive], items[positive])),
        shape=(n_users, n_items), dtype=np.float32,
    )


def least_squares_update(
    targets: sparse.csr_matrix,
    fixed_factors: np.ndarray,
    regularization: float,
    alpha: float,
) -> np.ndarray:
    """Exactly solve one ALS half-step; targets stores every observed pair.

    For each row u, A = Y.T Y + alpha * Y_obs.T Y_obs + lambda I and
    b = (1 + alpha) * Y_obs.T p_obs. The unobserved target is zero.
    Double precision is used for normal equations; returned factors are float32.
    """
    fixed = np.asarray(fixed_factors, dtype=np.float64)
    rank = fixed.shape[1]
    base = fixed.T @ fixed
    base.flat[::rank + 1] += regularization
    updated = np.zeros((targets.shape[0], rank), dtype=np.float32)
    for row in range(targets.shape[0]):
        start, end = targets.indptr[row:row + 2]
        if start == end:
            continue
        factors = fixed[targets.indices[start:end]]
        matrix = base + alpha * (factors.T @ factors)
        rhs = (1.0 + alpha) * (factors.T @ targets.data[start:end])
        decomposition = cho_factor(matrix, lower=True, overwrite_a=True, check_finite=False)
        updated[row] = cho_solve(decomposition, rhs, check_finite=False)
    return updated


class WeightedALS:
    """Internal common solver; subclasses define the observed target."""

    def __init__(self, factors: int = 24, regularization: float = 1.0,
                 alpha: float = 10.0, iterations: int = 8, seed: int = 42):
        self.factors = check_integer("factors", factors)
        self.regularization = check_float("regularization", regularization, strictly_positive=True)
        self.alpha = check_float("alpha", alpha)
        self.iterations = check_integer("iterations", iterations)
        self.seed = check_integer("seed", seed, minimum=0)

    def _observed_target(self, ratings: np.ndarray, timestamps: np.ndarray,
                         cutoff_ms: int) -> np.ndarray:
        raise NotImplementedError

    def fit(self, train: "pd.DataFrame", n_users: int, n_items: int,
            cutoff_ms: int) -> "WeightedALS":
        users, items, ratings, timestamps = validate_training_data(
            train, n_users, n_items, cutoff_ms)
        values = self._observed_target(ratings, timestamps, cutoff_ms)
        targets = sparse.csr_matrix((values, (users, items)), shape=(n_users, n_items))
        targets.sort_indices()
        # Do not eliminate zero values: their observed confidence must be retained.
        item_targets = targets.T.tocsr()
        rng = np.random.default_rng(self.seed)
        item_factors = rng.normal(0.0, 0.05, (n_items, self.factors)).astype(np.float32)
        for _ in range(self.iterations):
            user_factors = least_squares_update(
                targets, item_factors, self.regularization, self.alpha)
            item_factors = least_squares_update(
                item_targets, user_factors, self.regularization, self.alpha)
        self.user_factors_ = user_factors
        self.item_factors_ = item_factors
        self.n_users_, self.n_items_ = n_users, n_items
        self.cutoff_ms_ = cutoff_ms
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        """Return unmasked scores; the common evaluator removes known items."""
        if not hasattr(self, "user_factors_"):
            raise RuntimeError("fit must be called before score")
        users = validate_users(users, self.n_users_)
        return self.user_factors_[users] @ self.item_factors_.T
