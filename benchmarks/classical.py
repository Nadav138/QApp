"""
benchmarks/classical.py
=======================
Classical CVXPY baselines for the portfolio SOCP benchmark.
"""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - surfaced when the helpers are called.
    cp = None


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


def _require_cvxpy():
    if cp is None:
        raise ImportError("cvxpy is required for classical portfolio solves.")


def _validate_extra_inequalities(extra_inequalities, assets):
    if extra_inequalities and assets is None:
        raise ValueError("`assets` must be provided when `extra_inequalities` is non-empty.")


def _extra_inequality_rows(extra_inequalities, assets):
    if not extra_inequalities:
        return []
    return [
        (np.array([coeffs.get(asset, 0.0) for asset in assets], dtype=float), rhs)
        for coeffs, rhs in extra_inequalities
    ]


def _portfolio_constraints_for_w(
    w,
    max_weight: float,
    total_allocation: float = 1.0,
    min_weight: float = 0.0,
    extra_inequalities: list | None = None,
    assets: list | None = None,
):
    _require_cvxpy()
    _validate_extra_inequalities(extra_inequalities, assets)

    constraints = [
        cp.sum(w) == total_allocation,
        w >= min_weight,
        w <= max_weight,
    ]
    for row, rhs in _extra_inequality_rows(extra_inequalities, assets):
        constraints.append(row @ w <= rhs)
    return constraints


def _solve_problem(prob):
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(verbose=False)
    return prob.status


def feasible_return_range(
    mu: np.ndarray,
    max_weight: float,
    total_allocation: float = 1.0,
    min_weight: float = 0.0,
    extra_inequalities: list | None = None,
    assets: list | None = None,
) -> tuple[float | None, float | None, str, str]:
    """
    Return the min/max feasible ``mu.T @ w`` under all non-return constraints.

    ``assets`` is required whenever ``extra_inequalities`` is non-empty because
    those inequalities are specified as ticker-keyed coefficient dictionaries.
    """
    _require_cvxpy()
    _validate_extra_inequalities(extra_inequalities, assets)

    mu = np.asarray(mu, dtype=float)

    w_min = cp.Variable(len(mu))
    min_prob = cp.Problem(
        cp.Minimize(mu.T @ w_min),
        _portfolio_constraints_for_w(
            w_min,
            max_weight,
            total_allocation=total_allocation,
            min_weight=min_weight,
            extra_inequalities=extra_inequalities,
            assets=assets,
        ),
    )
    min_status = _solve_problem(min_prob)

    w_max = cp.Variable(len(mu))
    max_prob = cp.Problem(
        cp.Maximize(mu.T @ w_max),
        _portfolio_constraints_for_w(
            w_max,
            max_weight,
            total_allocation=total_allocation,
            min_weight=min_weight,
            extra_inequalities=extra_inequalities,
            assets=assets,
        ),
    )
    max_status = _solve_problem(max_prob)

    ok = min_status in _OPTIMAL_STATUSES and max_status in _OPTIMAL_STATUSES
    if not ok:
        return None, None, min_status, max_status
    return float(min_prob.value), float(max_prob.value), min_status, max_status


def solve_classical_portfolio_cvxpy(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    max_weight: float,
    total_allocation: float = 1.0,
    min_weight: float = 0.0,
    extra_inequalities: list | None = None,
    assets: list | None = None,
) -> tuple[np.ndarray | None, bool, str]:
    """
    Solve the paper-aligned classical portfolio SOCP with a target equality.

    ``assets`` is required whenever ``extra_inequalities`` is non-empty because
    those inequalities are specified as ticker-keyed coefficient dictionaries.
    """
    _require_cvxpy()
    _validate_extra_inequalities(extra_inequalities, assets)

    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    n_assets = len(mu)
    w = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = _portfolio_constraints_for_w(
        w,
        max_weight,
        total_allocation=total_allocation,
        min_weight=min_weight,
        extra_inequalities=extra_inequalities,
        assets=assets,
    )
    constraints.append(mu.T @ w == target_return)

    prob = cp.Problem(objective, constraints)
    try:
        status = _solve_problem(prob)
        ok = status in _OPTIMAL_STATUSES
        return (np.asarray(w.value, dtype=float) if ok else None), ok, status
    except Exception as exc:
        return None, False, str(exc)
