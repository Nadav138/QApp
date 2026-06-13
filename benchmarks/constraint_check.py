"""
benchmarks/constraint_check.py
==============================
Portfolio constraint validation for raw or normalized weight vectors.
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
    return_relation: str = "ineq",
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a constraint-validation report for a portfolio weight vector.

    ``return_relation='eq'`` matches the paper formulation ``mu.T @ w = R``.
    The report does not renormalize or clip weights; violations are shown raw.
    """
    if return_relation not in ("ineq", "eq"):
        raise ValueError("return_relation must be either 'ineq' or 'eq'.")

    w = np.asarray(w, dtype=float)
    mu_vec = np.asarray(mu_vec, dtype=float)

    rows = [
        ("budget_eq", float(np.sum(w) - total_allocation), "~0"),
    ]
    if return_relation == "eq":
        rows.append(("return_eq", float(w @ mu_vec - target_return), "~0"))
    else:
        rows.append(("return_ineq (>=target)", float(w @ mu_vec - target_return), ">=0"))
    rows.extend([
        ("long_only_min", float(np.min(w)), ">=0"),
        ("max_weight_cap", float(max_weight - np.max(w)), ">=0"),
        ("sum_w_raw", float(np.sum(w)), "report"),
    ])

    if extra_inequalities:
        if assets is None:
            raise ValueError("`assets` must be provided when `extra_inequalities` is non-empty.")
        for i, (coeffs, rhs) in enumerate(extra_inequalities):
            row = np.array([coeffs.get(a, 0.0) for a in assets], dtype=float)
            rows.append((f"extra_ineq_{i} (rhs-row@w)", float(rhs - row @ w), ">=0"))

    out = pd.DataFrame(rows, columns=["constraint", "value", "condition"])

    def _satisfied(row):
        if row["condition"] == "~0":
            return abs(row["value"]) <= tol
        if row["condition"] == ">=0":
            return row["value"] >= -tol
        return True

    out["is_satisfied"] = out.apply(_satisfied, axis=1)
    return out
