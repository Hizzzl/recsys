"""Focused FM checks; run ``python -m unittest discover -s tests -v`` in hw1.

The numerical-gradient and scoring oracles explicitly sum pairwise feature
interactions, independently of the model's efficient sum-of-squares formula.
All fixtures are synthetic and require no downloaded data.
"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.fm import (  # noqa: E402
    FactorizationMachine,
    _CausalFeatureEncoder,
    _batch_loss_and_gradient,
)


def _milliseconds(date):
    return pd.Timestamp(date, tz="UTC").value // 1_000_000


def _fixture():
    dates = ["2021-01-01", "2021-01-01", "2021-02-01", "2021-02-01",
             "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01"]
    return pd.DataFrame({
        "user_idx": [0, 1, 0, 0, 1, 2, 2, 0],
        "item_idx": [0, 1, 2, 3, 2, 3, 0, 1],
        "rating": [5, 1, 4, 1, 5, 4, 2, 4],
        "timestamp": [_milliseconds(date) for date in dates],
    })


def _pairwise_logits(feature_ids, bias, weights, factors):
    """Definition of order-2 FM, without its vectorized computation shortcut."""
    logits = np.empty(len(feature_ids), dtype=np.float64)
    for row, active in enumerate(feature_ids):
        score = float(bias + weights[active].sum())
        for left in range(len(active)):
            for right in range(left + 1, len(active)):
                score += float(factors[active[left]] @ factors[active[right]])
        logits[row] = score
    return logits


def _reference_loss(feature_ids, labels, bias, weights, factors, regularization):
    logits = _pairwise_logits(feature_ids, bias, weights, factors)
    data_loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
    # The documented regularizer averages squared active parameters per example.
    squared_active = np.square(weights[feature_ids]).sum(axis=1)
    squared_active += np.square(factors[feature_ids]).sum(axis=(1, 2))
    return data_loss + 0.5 * regularization * squared_active.mean()


class _CapturingFM(FactorizationMachine):
    """Capture actual fit-time samples without changing production behavior."""

    def _sample_training_pairs(self, frame, rng):
        result = super()._sample_training_pairs(frame, rng)
        self.sampled_pairs = tuple(array.copy() for array in result[:4])
        self.ordered_train = frame.copy()
        return result


class FactorizationMachineTests(unittest.TestCase):
    def test_gradient_matches_independent_finite_differences(self):
        rng = np.random.default_rng(7)
        # Feature zero is an active feature; repeated IDs across rows exercise
        # sparse gradient accumulation rather than a padding convention.
        feature_ids = np.array([[0, 3, 5], [1, 3, 6], [0, 4, 6], [2, 4, 5]])
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        weights = rng.normal(0, 0.1, 7)
        factors = rng.normal(0, 0.1, (7, 3))
        bias, regularization, epsilon = 0.12, 0.027, 1e-6
        loss, db, active, dw, dv = _batch_loss_and_gradient(
            feature_ids, labels, bias, weights, factors, regularization
        )

        def objective(intercept=bias):
            return _reference_loss(
                feature_ids, labels, intercept, weights, factors, regularization
            )

        self.assertAlmostEqual(loss, objective(), places=12)
        numeric_db = (objective(bias + epsilon) - objective(bias - epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(db, numeric_db, places=8)
        for local, feature in enumerate(active):
            original = weights[feature]
            weights[feature] = original + epsilon
            plus = objective()
            weights[feature] = original - epsilon
            minus = objective()
            weights[feature] = original
            self.assertAlmostEqual(dw[local], (plus - minus) / (2 * epsilon), places=8)
            for component in range(factors.shape[1]):
                original = factors[feature, component]
                factors[feature, component] = original + epsilon
                plus = objective()
                factors[feature, component] = original - epsilon
                minus = objective()
                factors[feature, component] = original
                self.assertAlmostEqual(
                    dv[local, component], (plus - minus) / (2 * epsilon), places=8
                )

    def test_full_catalog_score_matches_pairwise_fm_definition(self):
        frame, cutoff = _fixture(), _milliseconds("2021-07-01")
        users = np.array([0, 2, 3])  # Includes a user without fit history.
        for metadata in (True, False):
            with self.subTest(use_metadata=metadata):
                model = FactorizationMachine(
                    factors=3, epochs=2, batch_size=4, seed=23,
                    use_metadata=metadata,
                ).fit(frame, n_users=4, n_items=5, cutoff_ms=cutoff)
                scores = model.score(users)
                self.assertEqual(scores.shape, (3, 5))
                self.assertEqual(scores.dtype, np.float32)
                self.assertTrue(np.isfinite(scores).all())
                pairs = model.feature_encoder_.encode_pairs(
                    np.repeat(users, 5), np.tile(np.arange(5), len(users)),
                    np.full(len(users) * 5, cutoff),
                )
                expected = _pairwise_logits(pairs, model.bias_, model.w_, model.V_)
                np.testing.assert_allclose(scores.ravel(), expected, rtol=1e-5, atol=2e-6)

    def test_history_excludes_current_event_and_all_timestamp_ties(self):
        encoder = _CausalFeatureEncoder(_fixture(), 4, 5, True)
        january, february, march = map(
            _milliseconds, ["2021-01-01", "2021-02-01", "2021-03-01"]
        )
        user_stats = encoder.user_history.query(
            np.array([0, 0, 0, 3]), np.array([january, february, february + 1, march])
        )
        np.testing.assert_array_equal(user_stats["count"], [0, 1, 3, 0])
        np.testing.assert_allclose(user_stats["mean_rating"], [0, 5, 10 / 3, 0])
        np.testing.assert_allclose(user_stats["positive_fraction"], [0, 1, 2 / 3, 0])
        self.assertEqual(user_stats["recency_days"][1], 31)
        self.assertEqual(user_stats["tenure_days"][1], 31)
        item_stats = encoder.item_history.query(
            np.array([2, 2, 4]), np.array([march, march + 1, march])
        )
        np.testing.assert_array_equal(item_stats["count"], [1, 2, 0])
        np.testing.assert_allclose(item_stats["mean_rating"], [4, 4.5, 0])

    def test_current_and_future_ratings_cannot_change_training_features(self):
        frame = _fixture()
        query_time = _milliseconds("2021-02-01")
        changed = frame.copy()
        changed.loc[changed.timestamp >= query_time, "rating"] = 1
        changed = pd.concat([changed, pd.DataFrame({
            "user_idx": [0, 3], "item_idx": [1, 4], "rating": [1, 5],
            "timestamp": [_milliseconds("2021-09-01")] * 2,
        })], ignore_index=True)
        original_encoder = _CausalFeatureEncoder(frame, 4, 5, True)
        changed_encoder = _CausalFeatureEncoder(changed, 4, 5, True)
        users, items = np.array([0, 0, 1, 2, 3]), np.array([2, 3, 2, 0, 4])
        times = np.full(len(users), query_time)
        np.testing.assert_array_equal(
            original_encoder.encode_pairs(users, items, times),
            changed_encoder.encode_pairs(users, items, times),
        )
        # Check raw statistics too: coarse bins must not hide a leakage defect.
        for history_name, entities in (("user_history", users), ("item_history", items)):
            before = getattr(original_encoder, history_name).query(entities, times)
            after = getattr(changed_encoder, history_name).query(entities, times)
            for statistic in before:
                np.testing.assert_array_equal(before[statistic], after[statistic])

    def test_query_month_uses_known_query_context(self):
        encoder = _CausalFeatureEncoder(_fixture(), 4, 5, True)
        month_offset, size = encoder.schema["query_month"]
        self.assertEqual(size, 12)
        january = encoder.user_block(np.array([0]), _milliseconds("2021-01-01"))
        july = encoder.user_block(np.array([0]), _milliseconds("2021-07-01"))
        self.assertEqual(january[0, -1], month_offset)
        self.assertEqual(july[0, -1], month_offset + 6)

    def test_same_seed_repeats_training_exactly(self):
        configuration = dict(factors=3, epochs=2, batch_size=4, seed=23)
        frame, cutoff = _fixture(), _milliseconds("2021-07-01")
        first = FactorizationMachine(**configuration).fit(frame, 4, 5, cutoff)
        second = FactorizationMachine(**configuration).fit(frame, 4, 5, cutoff)
        np.testing.assert_array_equal(first.w_, second.w_)
        np.testing.assert_array_equal(first.V_, second.V_)
        self.assertEqual(first.bias_, second.bias_)
        self.assertEqual(first.loss_history_, second.loss_history_)
        np.testing.assert_array_equal(first.score(np.arange(4)), second.score(np.arange(4)))

    def test_ids_ablation_preserves_samples_and_causal_negative_policy(self):
        frame, cutoff = _fixture(), _milliseconds("2021-07-01")
        models = [
            _CapturingFM(factors=3, epochs=1, seed=23, use_metadata=metadata).fit(
                frame, 4, 5, cutoff
            ) for metadata in (True, False)
        ]
        for full, ablated in zip(models[0].sampled_pairs, models[1].sampled_pairs):
            np.testing.assert_array_equal(full, ablated)
        self.assertEqual(models[0].training_summary_["active_fields_per_row"], 13)
        self.assertEqual(models[1].training_summary_["active_fields_per_row"], 2)
        self.assertEqual(models[1].feature_encoder_.n_features, 4 + 5)
        users, items, timestamps, labels = models[0].sampled_pairs
        self.assertGreater(len(users), len(frame))  # Do not pass vacuously.
        np.testing.assert_array_equal(labels[len(frame):], 0)
        for user, item, timestamp in zip(
            users[len(frame):], items[len(frame):], timestamps[len(frame):]
        ):
            introduced = (frame.item_idx == item) & (frame.timestamp < timestamp)
            known_to_user = ((frame.user_idx == user) & (frame.item_idx == item)
                             & (frame.timestamp <= timestamp))
            self.assertTrue(introduced.any(), "negative item was not yet introduced")
            self.assertFalse(known_to_user.any(), "negative pair was already observed or tied")


if __name__ == "__main__":
    unittest.main()
