"""Implicit ALS with an explicitly implemented weighted least-squares solver."""

import numpy as np

from ._als import WeightedALS


class ImplicitALS(WeightedALS):
    r"""Minimize sum_ui c_ui (p_ui - x_u.T y_i)^2 + lambda(||X||²+||Y||²).

    p_ui = 1[rating >= 4] for observed pairs and zero for missing pairs.
    c_ui = 1 + alpha * 1[observed], including ratings below four. This makes
    low ratings stronger observed zeros and leaves all missing confidence at one.
    Regularization is global and is not multiplied by row interaction counts.
    """

    def _observed_target(self, ratings: np.ndarray, timestamps: np.ndarray,
                         cutoff_ms: int) -> np.ndarray:
        return (ratings >= 4).astype(np.float64)
