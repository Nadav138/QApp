import unittest

import numpy as np

from benchmarks import classical
from benchmarks.classical import feasible_return_range, solve_classical_portfolio_cvxpy


@unittest.skipIf(classical.cp is None, "cvxpy is not installed")
class ClassicalPortfolioTests(unittest.TestCase):
    def test_feasible_return_range_and_solve(self):
        mu = np.array([0.10, 0.20])
        cov = np.eye(2)

        ret_min, ret_max, min_status, max_status = feasible_return_range(
            mu,
            max_weight=0.8,
            assets=["AAA", "BBB"],
        )

        self.assertEqual(min_status, "optimal")
        self.assertEqual(max_status, "optimal")
        self.assertAlmostEqual(ret_min, 0.12, places=6)
        self.assertAlmostEqual(ret_max, 0.18, places=6)

        weights, ok, status = solve_classical_portfolio_cvxpy(
            mu,
            cov,
            target_return=0.15,
            max_weight=0.8,
            assets=["AAA", "BBB"],
        )

        self.assertTrue(ok, status)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=6)
        self.assertAlmostEqual(float(weights @ mu), 0.15, places=6)
        self.assertLessEqual(float(np.max(weights)), 0.8 + 1e-6)

    def test_extra_inequalities_require_assets(self):
        mu = np.array([0.10, 0.20])
        cov = np.eye(2)

        with self.assertRaisesRegex(ValueError, "assets"):
            solve_classical_portfolio_cvxpy(
                mu,
                cov,
                target_return=0.15,
                max_weight=0.8,
                extra_inequalities=[({"AAA": 1.0}, 0.7)],
            )
