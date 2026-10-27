"""EASE from its constrained closed-form solution, without a model library."""

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._als import check_float, positive_matrix, validate_training_data, validate_users

if TYPE_CHECKING:
    import pandas as pd


class EASE:
    r"""Minimize ||X-XB||_F² + lambda ||B||_F² subject to diag(B)=0.

    X_ui = 1[rating >= 4]. For P = (X.T X + lambda I)^(-1),
    B_ij = -P_ij / P_jj for i != j; B_jj = 0. Recommendations use X_u B.
    The inverse is computed via Cholesky solves in float64.
    """

    def __init__(self, regularization: float = 100.0):
        self.regularization = check_float("regularization", regularization, strictly_positive=True)

    def fit(self, train: "pd.DataFrame", n_users: int, n_items: int,
            cutoff_ms: int) -> "EASE":
        users, items, ratings, _ = validate_training_data(train, n_users, n_items, cutoff_ms)
        self.interactions_ = positive_matrix(users, items, ratings, n_users, n_items)
        x = self.interactions_.astype(np.float64)
        gram = (x.T @ x).toarray()
        gram.flat[::n_items + 1] += self.regularization
        decomposition = cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
        precision = cho_solve(decomposition, np.eye(n_items), check_finite=False)
        coefficients = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(coefficients, 0.0)
        self.coefficients_ = coefficients.astype(np.float32)
        self.n_users_, self.n_items_ = n_users, n_items
        self.cutoff_ms_ = cutoff_ms
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        """Return unmasked scores; the common evaluator removes known items."""
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("fit must be called before score")
        users = validate_users(users, self.n_users_)
        return np.asarray(self.interactions_[users] @ self.coefficients_, dtype=np.float32)
