"""Item-based collaborative filtering with a from-scratch cosine neighbourhood."""

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from ._als import (check_float, check_integer, positive_matrix,
                   validate_training_data, validate_users)

if TYPE_CHECKING:
    import pandas as pd


class ItemKNN:
    r"""Sum item similarities over a user's positively rated history.

    Let X_ui = 1[rating >= 4], n_ij = (X.T X)_ij and n_i = sum_u X_ui.
    s_ij = n_ij / sqrt(n_i n_j) * n_ij / (n_ij + shrinkage), with s_ii=0.
    Keep the largest `neighbors` outgoing similarities for each source item;
    ties use ascending item ID. Predictions are X_u S. Negative/neutral ratings
    are not positive neighbours; all known ratings are masked by the evaluator.
    """

    def __init__(self, neighbors: int = 50, shrinkage: float = 10.0):
        self.neighbors = check_integer("neighbors", neighbors)
        self.shrinkage = check_float("shrinkage", shrinkage)

    def fit(self, train: "pd.DataFrame", n_users: int, n_items: int,
            cutoff_ms: int) -> "ItemKNN":
        users, items, ratings, _ = validate_training_data(train, n_users, n_items, cutoff_ms)
        self.interactions_ = positive_matrix(users, items, ratings, n_users, n_items)
        cooccurrence = (self.interactions_.T @ self.interactions_).tocsr()
        counts = np.asarray(self.interactions_.sum(axis=0)).ravel().astype(np.float64)
        norms = np.sqrt(counts)
        values, indices = [], []
        indptr = [0]
        for source in range(n_items):
            start, end = cooccurrence.indptr[source:source + 2]
            candidate_items = cooccurrence.indices[start:end]
            shared = cooccurrence.data[start:end].astype(np.float64)
            keep = candidate_items != source
            candidate_items, shared = candidate_items[keep], shared[keep]
            similarities = shared / (norms[source] * norms[candidate_items])
            similarities *= shared / (shared + self.shrinkage)
            order = np.lexsort((candidate_items, -similarities))[:self.neighbors]
            indices.extend(candidate_items[order])
            values.extend(similarities[order])
            indptr.append(len(values))
        self.similarity_ = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int64)), shape=(n_items, n_items))
        self.n_users_, self.n_items_ = n_users, n_items
        self.cutoff_ms_ = cutoff_ms
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        """Return unmasked scores; the common evaluator removes known items."""
        if not hasattr(self, "similarity_"):
            raise RuntimeError("fit must be called before score")
        users = validate_users(users, self.n_users_)
        return (self.interactions_[users] @ self.similarity_).toarray()
