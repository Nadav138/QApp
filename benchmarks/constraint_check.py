"""
benchmarks/constraint_check.py
==============================
Portfolio constraint validation — verifies that a weight vector satisfies
budget, return, long-only, max-weight, and any extra inequality constraints.

Produces a tidy DataFrame report suitable for ``display(...)`` in a notebook.
"""

import numpy as np
import pandas as pd


def build_constraint_report(
    w: np.ndarray,
    mu_vec: np.ndarray,
    target_return: float,
    max_weight: float,
    total_allocation: float = 1.0,
    extra_inequalities: list | None = None,
    assets: list | None = None,
) -> pd.DataFrame:
    """
    Build a constraint-validation report for a portfolio weight vector.

    Parameters
    ----------
    w                : np.ndarray   Portfolio weights.
    mu_vec           : np.ndarray   Expected-return vector (same length as w).
    target_return    : float        Required portfolio return.
    max_weight       : float        Upper bound on individual weights.
    total_allocation : float        Budget target (default 1.0 → fully invested).
    extra_inequalities : list[(dict, float)] | None
        Optional list of ``(coeffs, rhs)`` pairs encoding ``Σ coeffs[a]·w_a ≤ rhs``.
        ``coeffs`` is a dict mapping asset ticker → coefficient.
        Requires ``assets`` to be provided so coefficients can be resolved.
    assets : list[str] | None
        Ordered list of tickers matching ``w``. Required only when
        ``extra_inequalities`` is non-empty.

    Returns
    -------
    pd.DataFrame
        Columns: ``constraint, value, condition, is_satisfied``.
    """
    rows = []
    rows.append(("budget_eq",              float(np.sum(w) - total_allocation), "~0"))
    rows.append(("return_ineq (>=target)", float(w @ mu_vec - target_return),   ">=0"))
    rows.append(("long_only_min",          float(np.min(w)),                    ">=0"))
    rows.append(("max_weight_cap",         float(max_weight - np.max(w)),       ">=0"))

    if extra_inequalities:
        if assets is None:
            raise ValueError("`assets` must be provided when `extra_inequalities` is non-empty.")
        for i, (coeffs, rhs) in enumerate(extra_inequalities):
            row = np.array([coeffs.get(a, 0.0) for a in assets], dtype=float)
            rows.append((f"extra_ineq_{i} (rhs-row@w)", float(rhs - row @ w), ">=0"))

    out = pd.DataFrame(rows, columns=["constraint", "value", "condition"])
    out["is_satisfied"] = out.apply(
        lambda r: abs(r["value"]) <= 1e-6 if r["condition"] == "~0" else r["value"] >= -1e-6,
        axis=1,
    )
    return out
