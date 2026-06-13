"""
core/qipm.py
============
Quantum Interior-Point Method for the Markowitz SOCP.

The implementation follows the paper's SOCP reduction and Newton system, with
an optional max-weight slack extension used by the notebooks. Each Newton-step
linear solve is delegated to ``quantum_newton_solver``.
"""

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - surfaced at runtime if Phase-I is used.
    cp = None

from core.hhl import quantum_newton_solver
from core.socp import arrowhead_product, fraction_to_boundary_step


def _build_portfolio_socp(mu_vec, M_mat, target_return, max_weight):
    """Build Eq. 5 SOCP matrices plus max-weight slack variables."""
    n = len(mu_vec)
    m = M_mat.shape[0]
    n_vars = 1 + m + 2 * n          # v = [t0; t_tilde; w; s_cap]
    r_cones = 1 + 2 * n             # 1 Lorentz + n weights + n cap slacks

    row1 = np.hstack([np.zeros((m, 1)), -np.eye(m), M_mat, np.zeros((m, n))])
    b1 = np.zeros(m)
    row2 = np.hstack([
        np.zeros((1, 1)), np.zeros((1, m)),
        mu_vec.reshape(1, -1), np.zeros((1, n)),
    ])
    b2 = np.array([target_return])
    row3 = np.hstack([
        np.zeros((1, 1)), np.zeros((1, m)),
        np.ones((1, n)), np.zeros((1, n)),
    ])
    b3 = np.array([1.0])
    row4 = np.hstack([
        np.zeros((n, 1)), np.zeros((n, m)),
        np.eye(n), np.eye(n),
    ])
    b4 = np.ones(n) * max_weight

    A = np.vstack([row1, row2, row3, row4])
    b = np.concatenate([b1, b2, b3, b4])
    c = np.concatenate([[1.0], np.zeros(m), np.zeros(2 * n)])
    e_cone = np.concatenate([[1.0], np.zeros(m), np.ones(2 * n)])
    return A, b, c, e_cone, n_vars, r_cones


def _solve_problem(prob):
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(verbose=False)
    return prob.status in ("optimal", "optimal_inaccurate")


def phase1_strictly_feasible(A, b, c, m, margin=1e-9):
    """Find strictly primal/dual feasible starts for the SOCP.

    The paper assumes a strictly feasible starting point but does not prescribe
    one. This Phase-I helper solves two tiny max-margin SOCPs and fails loudly
    when the configured constraints do not admit a strict interior point.
    """
    if cp is None:
        raise ImportError("cvxpy is required for QIPM Phase-I initialization.")

    n_vars = A.shape[1]
    v = cp.Variable(n_vars)
    d = cp.Variable()
    primal_prob = cp.Problem(
        cp.Maximize(d),
        [
            A @ v == b,
            v[1 + m:] >= d,
            v[0] >= cp.norm(v[1:1 + m], 2) + d,
        ],
    )
    if not _solve_problem(primal_prob) or d.value is None or d.value <= margin:
        raise RuntimeError(
            f"Phase-I: no strictly feasible primal start (margin={d.value})."
        )

    y_var = cp.Variable(A.shape[0])
    e = cp.Variable()
    s_expr = c - A.T @ y_var
    dual_prob = cp.Problem(
        cp.Maximize(e),
        [
            s_expr[1 + m:] >= e,
            s_expr[0] >= cp.norm(s_expr[1:1 + m], 2) + e,
        ],
    )
    if not _solve_problem(dual_prob) or e.value is None or e.value <= margin:
        raise RuntimeError(
            f"Phase-I: no strictly feasible dual start (margin={e.value})."
        )

    x = np.asarray(v.value, dtype=float)
    y = np.asarray(y_var.value, dtype=float)
    s = np.asarray(c - A.T @ y, dtype=float)
    phase1_info = {
        "primal_margin": float(d.value),
        "dual_margin": float(e.value),
    }
    return x, y, s, phase1_info


def _fallback_start(n, m, M_mat, max_weight, n_cons, n_vars):
    """Legacy infeasible-start fallback, kept only for explicit experiments."""
    w0 = np.ones(n) / n
    s_cap0 = np.ones(n) * max_weight - w0
    t_tilde0 = M_mat @ w0
    t0 = np.linalg.norm(t_tilde0) + 1.0
    x = np.concatenate([[t0], t_tilde0, w0, s_cap0])
    y = np.zeros(n_cons)
    s = np.ones(n_vars) * 0.5
    s[0] = np.linalg.norm(s[1:1 + m]) + 1.0
    return x, y, s, {"primal_margin": None, "dual_margin": None}


def _cone_margins(v, m):
    lorentz = float(v[0] - np.linalg.norm(v[1:1 + m]))
    scalar = float(np.min(v[1 + m:]))
    return lorentz, scalar, min(lorentz, scalar)


def _diagnostics(x, y, s, A, b, c, mu_vec, target_return, max_weight, m, n, r_cones):
    w = x[1 + m:1 + m + n]
    primal_resid = A @ x - b
    dual_resid = A.T @ y + s - c
    p_lorentz, p_scalar, p_margin = _cone_margins(x, m)
    d_lorentz, d_scalar, d_margin = _cone_margins(s, m)
    primal_inf = float(np.linalg.norm(primal_resid, ord=np.inf))
    dual_inf = float(np.linalg.norm(dual_resid, ord=np.inf))
    sum_w = float(np.sum(w))
    ret = float(w @ mu_vec)
    return {
        "sum_w": sum_w,
        "budget_eq": sum_w - 1.0,
        "return_value": ret,
        "target_return": float(target_return),
        "return_eq": ret - float(target_return),
        "long_only_min": float(np.min(w)),
        "max_weight_cap": float(max_weight - np.max(w)),
        "primal_resid": primal_inf,
        "dual_resid": dual_inf,
        "primal_resid_inf": primal_inf,
        "dual_resid_inf": dual_inf,
        "duality_gap": float(np.dot(x, s) / r_cones),
        "cone_primal": p_lorentz,
        "cone_dual": d_lorentz,
        "cone_margin_primal": p_margin,
        "cone_margin_dual": d_margin,
        "lorentz_margin_primal": p_lorentz,
        "scalar_margin_primal": p_scalar,
        "lorentz_margin_dual": d_lorentz,
        "scalar_margin_dual": d_scalar,
    }


def _gap_ratios(gaps):
    if len(gaps) < 2:
        return []
    return [float(gaps[i + 1] / gaps[i]) for i in range(len(gaps) - 1) if gaps[i] != 0]


def run_socp_quantum_ipm(
    mu_vec,
    M_mat,
    target_return,
    max_weight,
    max_iter=15,
    tol=1e-3,
    n_clk=4,
    pad_eig=0.1,
    use_adaptive_step=True,
    alpha_fixed=0.5,
    step_rule=None,
    use_phase1=True,
    residual_tol=1e-5,
    return_result=False,
):
    """
    Quantum Interior-Point Method for the Markowitz SOCP.

    The SOCP uses the paper's equality target ``mu.T @ w = target_return`` and
    a max-weight slack extension. The returned weights are raw final-iterate
    weights: no clipping and no normalization are applied.

    ``n_clk`` is the low-level HHL name for the number of QPE clock-register
    qubits used in each Newton-system solve. Research notebook configs expose
    this as ``quantum_qipm_n_clk`` to distinguish it from plain HHL demos.
    """
    mu_vec = np.asarray(mu_vec, dtype=float)
    M_mat = np.asarray(M_mat, dtype=float)
    n = len(mu_vec)
    m = M_mat.shape[0]
    A, b, c, e_cone, n_vars, r_cones = _build_portfolio_socp(
        mu_vec, M_mat, target_return, max_weight
    )
    n_cons = A.shape[0]
    sigma_theory = float(1.0 - 0.1 / np.sqrt(r_cones))

    if step_rule is None:
        step_rule = "adaptive" if use_adaptive_step else "fixed"
    if step_rule not in ("paper", "adaptive", "fixed"):
        raise ValueError("step_rule must be one of: 'paper', 'adaptive', 'fixed'.")

    if use_phase1:
        x, y, s, phase1_info = phase1_strictly_feasible(A, b, c, m)
    else:
        x, y, s, phase1_info = _fallback_start(n, m, M_mat, max_weight, n_cons, n_vars)

    print(f"Starting Full SOCP IPM loop for {n} assets (step_rule={step_rule})...")
    print(f"  {'Iter':>4}  {'Alpha':>8}  {'Duality Gap':>14}  {'Primal ||r||∞':>14}  {'Dual ||r||∞':>12}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*12}")

    status = "max_iter"
    status_reason = f"Reached max_iter={max_iter}."
    gaps = []
    alphas = []
    hhl_diagnostics = []

    for i in range(max_iter):
        gap = np.dot(x, s) / r_cones
        mu_barrier = sigma_theory * gap

        Arw_x = arrowhead_product(x, m, n)
        Arw_s = arrowhead_product(s, m, n)

        # KKT Newton system, matching Eq. 6 up to row ordering of primal/dual residuals.
        K_top = np.hstack([np.zeros((n_vars, n_vars)), A.T, np.eye(n_vars)])
        K_mid = np.hstack([A, np.zeros((n_cons, n_cons)), np.zeros((n_cons, n_vars))])
        K_bot = np.hstack([Arw_s, np.zeros((n_vars, n_cons)), Arw_x])
        KKT_full = np.vstack([K_top, K_mid, K_bot])

        rp = b - A @ x
        rd = c - s - A.T @ y
        rc = mu_barrier * e_cone - Arw_x @ s
        rhs = np.concatenate([rd, rp, rc])

        dz, hhl_info = quantum_newton_solver(
            KKT_full,
            rhs,
            n_clk=n_clk,
            pad_eig=pad_eig,
            return_diagnostics=True,
        )
        hhl_diagnostics.append(hhl_info)
        dx = dz[:n_vars]
        dy = dz[n_vars:n_vars + n_cons]
        ds = dz[-n_vars:]

        if step_rule == "paper":
            alpha = 1.0
        elif step_rule == "adaptive":
            alpha = fraction_to_boundary_step(x, s, dx, ds, m, scaling=0.95)
        else:
            alpha = alpha_fixed
        alphas.append(float(alpha))

        x += alpha * dx
        y += alpha * dy
        s += alpha * ds

        info = _diagnostics(x, y, s, A, b, c, mu_vec, target_return, max_weight, m, n, r_cones)
        gap_abs = abs(info["duality_gap"])
        gaps.append(gap_abs)
        print(
            f"  {i:>4}  {alpha:>8.4f}  {gap_abs:>14.6f}  "
            f"{info['primal_resid_inf']:>14.2e}  {info['dual_resid_inf']:>12.2e}"
        )

        if info["cone_margin_primal"] <= -residual_tol or info["cone_margin_dual"] <= -residual_tol:
            status = "left_cone"
            status_reason = "Final iterate left the cone beyond residual_tol."
            break
        if gap_abs < tol and info["primal_resid_inf"] < residual_tol and info["dual_resid_inf"] < residual_tol:
            status = "converged"
            status_reason = "Duality gap and linear residual tolerances reached."
            break

    w_opt = x[1 + m:1 + m + n]
    diagnostics = _diagnostics(
        x, y, s, A, b, c, mu_vec, target_return, max_weight, m, n, r_cones
    )
    result = {
        "weights": w_opt,
        "x": x,
        "y": y,
        "s": s,
        "A": A,
        "b": b,
        "c": c,
        "e_cone": e_cone,
        "gaps": gaps,
        "gap_ratios": _gap_ratios(gaps),
        "alphas": alphas,
        "diagnostics": diagnostics,
        "phase1_diagnostics": phase1_info,
        "hhl_diagnostics": hhl_diagnostics,
        "status": status,
        "status_reason": status_reason,
        "step_rule": step_rule,
        "iterations": len(gaps),
        "sigma_theory": sigma_theory,
        "qpe_clock_qubits": int(n_clk),
        "n_clk": int(n_clk),
        "raw_weights": True,
    }
    return result if return_result else w_opt
