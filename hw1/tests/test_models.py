"""Small independent numerical checks; run `python -m unittest discover -s tests`.

These checks use tiny explicit matrices rather than a recommendation library.
From the repository root use `python -m unittest discover -s hw1/tests`.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models._als import least_squares_update
from models.custom_als import CustomTargetALS
from models.ease import EASE
from models.ials import ImplicitALS
from models.item_knn import ItemKNN


DAY_MS = 86_400_000
CUTOFF_MS = 365 * DAY_MS


def small_frame() -> pd.DataFrame:
    """Five observed users, four observed items, positive and low ratings."""
    observations = [
        (0, 0, 5), (0, 1, 5), (1, 0, 5), (1, 1, 4),
        (2, 0, 4), (2, 1, 5), (2, 2, 5), (3, 2, 5),
        (3, 3, 1), (4, 1, 3), (4, 3, 2),
    ]
    frame = pd.DataFrame(observations, columns=["user_idx", "item_idx", "rating"])
    frame["timestamp"] = np.arange(len(frame), dtype=np.int64) * DAY_MS
    return frame


class NumericalModelTests(unittest.TestCase):
    def test_als_half_step_matches_dense_confidence_equations(self):
        """A stored zero must receive observed confidence, not missing confidence."""
        fixed = np.random.default_rng(19).normal(size=(4, 3))
        rows = np.array([0, 0, 1, 1])
        columns = np.array([0, 2, 1, 3])
        values = np.array([1.0, 0.0, -0.5, 0.4])
        targets = sparse.csr_matrix((values, (rows, columns)), shape=(3, 4))
        alpha, regularization = 7.0, 0.8
        actual = least_squares_update(targets, fixed, regularization, alpha)
        expected = []
        for user in range(3):
            observed = rows == user
            confidence = np.ones(4)
            confidence[columns[observed]] += alpha
            target = np.zeros(4)
            target[columns[observed]] = values[observed]
            dense_c = np.diag(confidence)
            matrix = fixed.T @ dense_c @ fixed + regularization * np.eye(3)
            rhs = fixed.T @ dense_c @ target
            expected.append(np.linalg.solve(matrix, rhs))
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
        self.assertEqual(targets.nnz, 4)
        self.assertEqual(targets.T.tocsr().nnz, 4)
        # The fixture specifically distinguishes the correct formula from
        # accidentally dropping the observed zero at (user=0, item=2).
        missing_zero_c = np.diag([1 + alpha, 1, 1, 1])
        wrong = np.linalg.solve(
            fixed.T @ missing_zero_c @ fixed + regularization * np.eye(3),
            fixed.T @ missing_zero_c @ np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.assertGreater(np.linalg.norm(actual[0] - wrong), 1e-3)

    def test_ease_matches_independent_constrained_column_solutions(self):
        """Eliminate each forbidden diagonal coordinate before ridge solving."""
        frame = small_frame()
        model = EASE(regularization=2.0).fit(frame, 6, 5, CUTOFF_MS)
        x = np.zeros((6, 5))
        positive = frame[frame.rating >= 4]
        x[positive.user_idx.to_numpy(), positive.item_idx.to_numpy()] = 1
        for target in range(5):
            others = np.arange(5) != target
            z = x[:, others]
            expected = np.linalg.solve(z.T @ z + 2.0 * np.eye(4), z.T @ x[:, target])
            np.testing.assert_allclose(
                model.coefficients_[others, target], expected, atol=1e-7, rtol=1e-6)
        np.testing.assert_array_equal(model.coefficients_.diagonal(), np.zeros(5))
        np.testing.assert_allclose(model.score(np.arange(6)), x @ model.coefficients_)

    def test_knn_similarity_shrinkage_and_top_neighbor(self):
        frame = small_frame()
        model = ItemKNN(neighbors=1, shrinkage=2.0).fit(frame, 6, 5, CUTOFF_MS)
        # Item 0 and item 1 have the identical three-user positive history:
        # cosine=1 and support shrinkage=3/(3+2).
        self.assertEqual(model.similarity_[0].indices.tolist(), [1])
        self.assertAlmostEqual(float(model.similarity_[0, 1]), 3.0 / 5.0, places=6)
        np.testing.assert_array_equal(model.similarity_.diagonal(), np.zeros(5))
        self.assertLessEqual(model.similarity_.getnnz(axis=1).max(), 1)
        np.testing.assert_allclose(
            model.score(np.array([0])),
            model.interactions_[[0]].toarray() @ model.similarity_.toarray(),
        )

    def test_custom_target_is_signed_and_decays_without_changing_confidence(self):
        frame = small_frame()
        parameters = dict(factors=3, iterations=4, alpha=7.0, seed=7)
        baseline = ImplicitALS(**parameters).fit(frame, 6, 5, CUTOFF_MS)
        custom = CustomTargetALS(**parameters, half_life_days=365.0).fit(
            frame, 6, 5, CUTOFF_MS)
        self.assertEqual(baseline.alpha, custom.alpha)
        self.assertEqual(baseline.regularization, custom.regularization)
        np.testing.assert_allclose(
            custom._observed_target(np.arange(1, 6), np.zeros(5), CUTOFF_MS),
            [-0.5, -0.25, 0.0, 0.25, 0.5],
        )
        self.assertFalse(np.allclose(baseline.score(np.arange(6)), custom.score(np.arange(6))))

    def test_als_seed_is_reproducible(self):
        frame = small_frame()
        for model_class in (ImplicitALS, CustomTargetALS):
            with self.subTest(model=model_class.__name__):
                first = model_class(factors=3, iterations=3, seed=7).fit(frame, 6, 5, CUTOFF_MS)
                second = model_class(factors=3, iterations=3, seed=7).fit(frame, 6, 5, CUTOFF_MS)
                np.testing.assert_array_equal(first.score(np.arange(6)), second.score(np.arange(6)))

    def test_finite_scores_cold_empty_and_repeated_users(self):
        frame = small_frame()
        for model in (ItemKNN(), EASE(), ImplicitALS(factors=3, iterations=2),
                      CustomTargetALS(factors=3, iterations=2)):
            with self.subTest(model=type(model).__name__):
                model.fit(frame, 6, 5, CUTOFF_MS)
                scores = model.score(np.array([0, 2, 5]))
                self.assertEqual(scores.shape, (3, 5))
                self.assertEqual(scores.dtype, np.float32)
                self.assertTrue(np.isfinite(scores).all())
                np.testing.assert_array_equal(scores[-1], np.zeros(5))
                self.assertEqual(model.score(np.array([], dtype=np.int64)).shape, (0, 5))
                repeated = model.score(np.array([2, 2]))
                np.testing.assert_array_equal(repeated[0], repeated[1])

    def test_unfitted_duplicates_and_future_interactions_are_rejected(self):
        frame = small_frame()
        for model_class in (ItemKNN, EASE, ImplicitALS, CustomTargetALS):
            with self.subTest(model=model_class.__name__):
                with self.assertRaises(RuntimeError):
                    model_class().score(np.array([0]))
                with self.assertRaises(ValueError):
                    model_class().fit(pd.concat([frame, frame.iloc[[0]]]), 6, 5, CUTOFF_MS)
                with self.assertRaises(ValueError):
                    model_class().fit(frame, 6, 5, cutoff_ms=0)


if __name__ == "__main__":
    unittest.main()
