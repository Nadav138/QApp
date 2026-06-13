import unittest
import os
import tempfile

import numpy as np
import pandas as pd

from benchmarks.oos import compute_oos_benchmark, load_oos_prices_for_period, parse_oos_period
from benchmarks.plots import plot_qipm_convergence
from benchmarks.reports import build_solver_summary, portfolio_metrics, weights_table
from core.risk import factor_covariance_matrix


class RiskReportsOOSTests(unittest.TestCase):
    def test_factor_covariance_matrix_reconstructs_covariance(self):
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])

        M, diagnostics = factor_covariance_matrix(cov)

        np.testing.assert_allclose(M.T @ M, cov, atol=1e-12)
        self.assertLess(diagnostics["recon_error"], 1e-12)
        self.assertEqual(diagnostics["eigenvalues"].shape, (2,))

    def test_reports_helpers_build_tables_and_metrics(self):
        assets = ["AAA", "BBB"]
        weights = np.array([0.2, 0.8])
        mu = np.array([0.10, 0.20])
        cov = np.eye(2)

        table = weights_table(assets, weights)
        self.assertEqual(table.iloc[0]["asset"], "BBB")

        metrics = portfolio_metrics(weights, mu, cov)
        self.assertAlmostEqual(metrics["expected_return"], 0.18)
        self.assertAlmostEqual(metrics["annual_variance"], 0.68)

        summary = build_solver_summary(
            w_classical=weights,
            classical_ok=True,
            classical_status="optimal",
            w_quantum=weights,
            quantum_status="solved",
            mu_vec=mu,
            cov=cov,
            target_return=0.18,
            quantum_diagnostics={
                "sum_w": 1.0,
                "budget_eq": 0.0,
                "return_eq": 0.0,
                "primal_resid_inf": 1e-8,
                "dual_resid_inf": 2e-8,
            },
        )
        self.assertEqual(list(summary["Status"]), ["optimal", "solved"])

    def test_oos_period_parsing_and_price_loading(self):
        self.assertEqual(
            parse_oos_period("2025-01-01 to 2025-01-05"),
            ("2025-01-01", "2025-01-05"),
        )
        with self.assertRaises(ValueError):
            parse_oos_period("2025-01-01/2025-01-05")

        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        close = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 102.0],
                "BBB": [50.0, np.nan, 51.0],
                "CCC": [np.nan, 10.0, 11.0],
            },
            index=dates,
        )
        raw = pd.concat({"Close": close, "Open": close + 1.0}, axis=1)

        def downloader(assets, **kwargs):
            self.assertEqual(assets, ["AAA", "BBB", "CCC"])
            self.assertEqual(kwargs["start"], "2025-01-01")
            self.assertEqual(kwargs["end"], "2025-01-05")
            return raw

        prices = load_oos_prices_for_period(
            ["AAA", "BBB", "CCC"],
            "2025-01-01 to 2025-01-05",
            downloader=downloader,
        )
        self.assertEqual(list(prices.columns), ["AAA", "BBB"])
        self.assertFalse(prices.isna().any().any())

    def test_compute_oos_benchmark_aligns_available_assets(self):
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 110.0, 121.0],
                "BBB": [100.0, 90.0, 81.0],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="D"),
        )

        metrics = compute_oos_benchmark(
            prices,
            ["AAA", "BBB", "CCC"],
            np.array([0.5, 0.5, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertEqual(metrics["assets"], ["AAA", "BBB"])
        self.assertAlmostEqual(metrics["classical_total_return"], 0.01)
        self.assertAlmostEqual(metrics["quantum_total_return"], 0.21)
        self.assertAlmostEqual(metrics["classical_hhi"], 0.5)
        self.assertAlmostEqual(metrics["quantum_hhi"], 1.0)

    def test_qipm_convergence_plot_uses_mathtext_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MPLCONFIGDIR"] = tmpdir

            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            fig, ax = plot_qipm_convergence([1.0, 0.8, 0.64], 0.97418, show=False)
            fig.canvas.draw()

            legend_labels = [text.get_text() for text in ax.get_legend().texts]
            self.assertIn(r"Theory $\nu_0 \cdot \sigma^k$", legend_labels)
            self.assertIn(r"$\sigma \approx", ax.get_title())
            plt.close(fig)
