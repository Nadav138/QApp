"""
benchmarks/reports.py
=====================
Display-table and scalar metric helpers for portfolio benchmark notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def weights_table(assets: list, weights: np.ndarray) -> pd.DataFrame:
    """Return a descending weight table."""
    return (
        pd.DataFrame({"asset": list(assets), "weight": np.asarray(weights, dtype=float)})
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )


def portfolio_metrics(weights: np.ndarray, mu_vec: np.ndarray, cov: np.ndarray) -> dict:
    """Return scalar portfolio metrics used by the notebook's printed reports."""
    weights = np.asarray(weights, dtype=float)
    mu_vec = np.asarray(mu_vec, dtype=float)
    cov = np.asarray(cov, dtype=float)
    annual_variance = float(weights @ cov @ weights)
    return {
        "expected_return": float(weights @ mu_vec),
        "annual_variance": annual_variance,
        "annual_volatility": float(np.sqrt(annual_variance)),
        "sum_weights": float(np.sum(weights)),
    }


def build_solver_summary(
    *,
    w_classical: np.ndarray,
    classical_ok: bool,
    classical_status: str,
    w_quantum: np.ndarray,
    quantum_status: str,
    mu_vec: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    quantum_diagnostics: dict | None = None,
    total_allocation: float = 1.0,
    classical_label: str = "Classical CVXPY (paper SOCP)",
    quantum_label: str = "Quantum IPM (paper SOCP, raw)",
) -> pd.DataFrame:
    """Build the classical-vs-quantum solver summary table."""
    quantum_diagnostics = quantum_diagnostics or {}

    if classical_ok:
        cls_metrics = portfolio_metrics(w_classical, mu_vec, cov)
        cls_return_resid = cls_metrics["expected_return"] - target_return
        cls_budget_resid = cls_metrics["sum_weights"] - total_allocation
        cls_primal_resid = 0.0
        cls_dual_resid = 0.0
        cls_status = classical_status
    else:
        cls_metrics = {
            "expected_return": float("nan"),
            "annual_variance": float("nan"),
            "annual_volatility": float("nan"),
            "sum_weights": float("nan"),
        }
        cls_return_resid = float("nan")
        cls_budget_resid = float("nan")
        cls_primal_resid = float("nan")
        cls_dual_resid = float("nan")
        cls_status = "failed"

    q_metrics = portfolio_metrics(w_quantum, mu_vec, cov)
    q_sum = quantum_diagnostics.get("sum_w", q_metrics["sum_weights"])
    q_budget = quantum_diagnostics.get("budget_eq", q_sum - total_allocation)
    q_return = quantum_diagnostics.get("return_eq", q_metrics["expected_return"] - target_return)
    q_primal = quantum_diagnostics.get(
        "primal_resid_inf",
        quantum_diagnostics.get("primal_resid", float("nan")),
    )
    q_dual = quantum_diagnostics.get(
        "dual_resid_inf",
        quantum_diagnostics.get("dual_resid", float("nan")),
    )

    return pd.DataFrame(
        {
            "Method": [classical_label, quantum_label],
            "Expected Return": [cls_metrics["expected_return"], q_metrics["expected_return"]],
            "Annual Variance": [cls_metrics["annual_variance"], q_metrics["annual_variance"]],
            "Annual Volatility": [cls_metrics["annual_volatility"], q_metrics["annual_volatility"]],
            "Sum Weights": [cls_metrics["sum_weights"], q_sum],
            "Budget Residual": [cls_budget_resid, q_budget],
            "Return Residual": [cls_return_resid, q_return],
            "Primal Residual": [cls_primal_resid, q_primal],
            "Dual Residual": [cls_dual_resid, q_dual],
            "Status": [cls_status, quantum_status],
        }
    )
