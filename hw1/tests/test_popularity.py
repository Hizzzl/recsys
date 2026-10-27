"""Numerical and evaluator checks for the fixed diagnostic baselines."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import evaluate
from models.popularity import Popularity


DAY_MS = 86_400_000
CUTOFF_MS = 365 * DAY_MS


def fixture():
    return pd.DataFrame({
        "user_idx": [0, 1, 0, 1],
        "item_idx": [0, 0, 1, 2],
        "rating": [5, 4, 3, 5],
        "timestamp": [0, CUTOFF_MS - DAY_MS, CUTOFF_MS - DAY_MS, 0],
    })


class PopularityTests(unittest.TestCase):
    def test_counts_and_unknown_item_against_explicit_oracle(self):
        model = Popularity().fit(fixture(), 2, 4, CUTOFF_MS)
        expected = np.array([[2., 0., 1., 0.], [2., 0., 1., 0.]])
        np.testing.assert_array_equal(model.score(np.array([0, 1])), expected)
        self.assertEqual(model.score(np.array([0])).dtype, np.float64)

    def test_decay_against_explicit_half_life_oracle(self):
        model = Popularity(half_life_days=365).fit(fixture(), 2, 4, CUTOFF_MS)
        expected = np.array([0.5 + 2. ** (-1. / 365.), 0., 0.5, 0.])
        np.testing.assert_allclose(model.score(np.array([0]))[0], expected,
                                   rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(model.score(np.array([0])),
                                      model.score(np.array([1])))

    def test_scores_are_writable_independent_copies(self):
        model = Popularity().fit(fixture(), 2, 4, CUTOFF_MS)
        scores = model.score(np.array([0, 1]))
        scores[0, 0] = -np.inf
        self.assertEqual(scores[1, 0], 2.)
        self.assertEqual(model.score(np.array([0]))[0, 0], 2.)
        self.assertEqual(model.score(np.array([], dtype=np.int64)).shape, (0, 4))

    def test_shared_evaluator_applies_different_personal_history_masks(self):
        history = fixture()
        model = Popularity().fit(history, 2, 4, CUTOFF_MS)
        heldout = pd.DataFrame({"user_idx": [0, 1], "item_idx": [2, 1], "rating": [5, 4]})
        result = evaluate(model, history, heldout, np.array([0, 1]), 2, 4, k=2)
        self.assertEqual(result["recommendations"], [[2, 3], [1, 3]])
        np.testing.assert_array_equal(model.score(np.array([0, 1])),
                                      [[2., 0., 1., 0.], [2., 0., 1., 0.]])

    def test_invalid_half_life_is_rejected(self):
        for invalid in (0, -1, np.inf, -np.inf, np.nan, True, "365"):
            with self.subTest(half_life_days=invalid):
                with self.assertRaises(ValueError):
                    Popularity(half_life_days=invalid)

    def test_future_interactions_and_invalid_cutoffs_are_rejected(self):
        for invalid_cutoff in (CUTOFF_MS - DAY_MS - 1, -1, np.inf, np.nan):
            with self.subTest(cutoff_ms=invalid_cutoff):
                with self.assertRaises(ValueError):
                    Popularity().fit(fixture(), 2, 4, invalid_cutoff)

    def test_unfitted_and_out_of_range_users_are_rejected(self):
        with self.assertRaises(RuntimeError):
            Popularity().score(np.array([0]))
        model = Popularity().fit(fixture(), 2, 4, CUTOFF_MS)
        with self.assertRaises(ValueError):
            model.score(np.array([2]))

    def test_no_positive_reviews_gives_zero_scores(self):
        frame = fixture().assign(rating=1)
        for half_life in (None, 365):
            with self.subTest(half_life_days=half_life):
                model = Popularity(half_life).fit(frame, 2, 4, CUTOFF_MS)
                np.testing.assert_array_equal(model.score(np.array([0, 1])),
                                              np.zeros((2, 4)))


if __name__ == "__main__":
    unittest.main()
