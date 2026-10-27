"""ALS with a signed, time-decayed rating target and baseline confidence."""

import numpy as np

from ._als import WeightedALS, check_float


class CustomTargetALS(WeightedALS):
    r"""Use p_ui = ((r_ui - 3)/2) * 2^(-age_days/half_life_days).

    Missing targets are zero; observed confidences are exactly the same as in
    ImplicitALS: c_ui = 1 + alpha, regardless of rating, age or target value.
    The target encodes both attraction/aversion and fading relevance of old
    preferences. Time is measured relative to the supplied training cutoff,
    never the held-out interaction timestamps. Ratings of three have target
    zero but still carry observed confidence. The test relevance definition
    remains external to the model and does not change with this target.
    """

    def __init__(self, factors: int = 24, regularization: float = 1.0,
                 alpha: float = 10.0, iterations: int = 8, seed: int = 42,
                 half_life_days: float = 365.0):
        super().__init__(factors, regularization, alpha, iterations, seed)
        self.half_life_days = check_float("half_life_days", half_life_days, strictly_positive=True)

    def _observed_target(self, ratings: np.ndarray, timestamps: np.ndarray,
                         cutoff_ms: int) -> np.ndarray:
        age_days = (float(cutoff_ms) - timestamps) / 86_400_000.0
        decay = np.exp2(-age_days / self.half_life_days)
        return ((ratings - 3.0) / 2.0) * decay
