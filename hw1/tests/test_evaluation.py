"""Independent toy ranking checks for the shared evaluation protocol."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import evaluate, top_k


class FixedScores:
    def __init__(self, values):
        self.values = np.asarray(values)

    def score(self, users):
        return self.values[users]


class EvaluationTest(unittest.TestCase):
    def test_metric_values_and_all_seen_mask(self):
        # Item 0 has a negative review: it must still be excluded despite score 99.
        history = pd.DataFrame({'user_idx': [0], 'item_idx': [0], 'rating': [1]})
        future = pd.DataFrame({'user_idx': [0, 0], 'item_idx': [2, 11], 'rating': [5, 4]})
        model = FixedScores([[99] + list(range(11, 0, -1))])
        result = evaluate(model, history, future, np.array([0]), 1, 12)
        self.assertEqual(result['recommendations'][0], list(range(1, 11)))
        self.assertEqual(result['Recall@10'], 0.5)
        self.assertAlmostEqual(result['NDCG@10'], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))

    def test_score_ties_ascending_item(self):
        self.assertEqual(top_k(np.zeros((1, 20)))[0].tolist(), list(range(10)))

    def test_insufficient_unseen_candidates_fails(self):
        history = pd.DataFrame({'user_idx': [0], 'item_idx': [0], 'rating': [5]})
        future = pd.DataFrame({'user_idx': [0], 'item_idx': [1], 'rating': [5]})
        with self.assertRaises(ValueError):
            evaluate(FixedScores(np.zeros((1, 10))), history, future, np.array([0]), 1, 10)


if __name__ == '__main__':
    unittest.main()
