import unittest

from importance_agent import CriterionResult, InvalidResponseError
from importance_agent.scoring import calculate_importance


class ScoringTests(unittest.TestCase):
    def score(self, ratings, weights):
        return calculate_importance([
            CriterionResult(str(i), rating, weight, "reason", ())
            for i, (rating, weight) in enumerate(zip(ratings, weights))
        ])

    def test_weighted_example(self):
        self.assertEqual(self.score([90, 80, 60], [0.4, 0.35, 0.25]), 79)

    def test_boundaries_and_half_up(self):
        self.assertEqual(self.score([1, 1], [0.5, 0.5]), 1)
        self.assertEqual(self.score([100, 100], [0.5, 0.5]), 100)
        self.assertEqual(self.score([50, 51], [0.5, 0.5]), 51)

    def test_normalizes_accepted_weight_tolerance(self):
        self.assertEqual(self.score([100, 100, 100], [.3333333] * 3), 100)

    def test_rejects_invalid_ratings(self):
        for rating in (0, 101, True, 50.0, "50"):
            with self.subTest(rating=rating), self.assertRaises(InvalidResponseError):
                self.score([rating], [1])

    def test_rejects_empty_and_invalid_weights(self):
        with self.assertRaises(InvalidResponseError):
            calculate_importance([])
        for weight in (0, -1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(InvalidResponseError):
                self.score([50], [weight])
