"""
core/qipm.py
============
Quantum Interior-Point Method for the Markowitz SOCP.

Every Newton-step linear solve is delegated to ``quantum_newton_solver`` (HHL).
Step-size selection (when adaptive) is delegated to ``fraction_to_boundary_step``.
"""

import numpy as np

from core.hhl  import quantum_newton_solver
from core.socp import arrowhead_product, fraction_to_boundary_step


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
):
    """
    Quantum Interior-Point Method for the Markowitz SOCP.

    Enforces four constraints (same as the classical CVXPY baseline):
        1. Budget        :  1ᵀx = 1
        2. Long-only     :  xᵢ ≥ 0
        3. Max-weight    :  xᵢ ≤ max_weight
        4. Return target :  µᵀx = target_return

    Parameters
    ----------
    mu_vec        : np.ndarray   Expected-return vector.
    M_mat         : np.ndarray   Risk-factor matrix s.t. Σ = MᵀM.
    target_return : float        Required portfolio return.
    max_weight    : float        Upper bound on individual weights.
    max_iter      : int          Newton iterations cap.
    tol           : float        Duality-gap stopping tolerance.
    n_clk         : int          HHL clock qubits passed to ``quantum_newton_solver``.
    pad_eig       : float        HHL eigenvalue padding.
    use_adaptive_step : bool     Use fraction-to-boundary step-size if True.
    alpha_fixed   : float        Step size when ``use_adaptive_step`` is False.

    Returns
    -------
    np.ndarray
        Normalised portfolio weights of length ``n`` (sums to 1).
    """
    n      = len(mu_vec)
    m      = M_mat.shape[0]
    n_vars = 1 + m + 2 * n          # v = [t0; t_tilde; x; s_max]
    r_cones = 1 + 2 * n             # 1 Lorentz + 2n non-negativity cones

    # ── Equality-constraint matrix A and RHS b ───────────────────────────────
    row1 = np.hstack([np.zeros((m, 1)), -np.eye(m), M_mat, np.zeros((m, n))])
    b1   = np.zeros(m)
    row2 = np.hstack([np.zeros((1, 1)), np.zeros((1, m)),
                       mu_vec.reshape(1, -1), np.zeros((1, n))])
    b2   = np.array([target_return])
    row3 = np.hstack([np.zeros((1, 1)), np.zeros((1, m)),
                       np.ones((1, n)), np.zeros((1, n))])
    b3   = np.array([1.0])
    row4 = np.hstack([np.zeros((n, 1)), np.zeros((n, m)),
                       np.eye(n), np.eye(n)])
    b4   = np.ones(n) * max_weight

    A      = np.vstack([row1, row2, row3, row4])
    b      = np.concatenate([b1, b2, b3, b4])
    c      = np.concatenate([[1.0], np.zeros(m), np.zeros(2 * n)])
    n_cons = A.shape[0]
    e_cone = np.concatenate([[1.0], np.zeros(m), np.ones(2 * n)])

    # ── Strictly-interior starting point ─────────────────────────────────────
    x_orig    = np.ones(n) / n
    s_max_0   = np.ones(n) * max_weight - x_orig
    t_tilde_0 = M_mat @ x_orig
    t0_init   = np.linalg.norm(t_tilde_0) + 1.0
    x = np.concatenate([[t0_init], t_tilde_0, x_orig, s_max_0])
    y = np.zeros(n_cons)
    s = np.ones(n_vars) * 0.5
    s[0] = np.linalg.norm(s[1:1 + m]) + 1.0

    print(f"Starting Full SOCP IPM loop for {n} assets...")

    for i in range(max_iter):
        gap        = np.dot(x, s) / r_cones
        sigma      = 1.0 - 0.1 / np.sqrt(r_cones)
        mu_barrier = sigma * gap

        Arw_x = arrowhead_product(x, m, n)
        Arw_s = arrowhead_product(s, m, n)

        # KKT Newton system (Eq. 6 of the paper)
        K_top    = np.hstack([np.zeros((n_vars, n_vars)), A.T, np.eye(n_vars)])
        K_mid    = np.hstack([A, np.zeros((n_cons, n_cons)), np.zeros((n_cons, n_vars))])
        K_bot    = np.hstack([Arw_s, np.zeros((n_vars, n_cons)), Arw_x])
        KKT_full = np.vstack([K_top, K_mid, K_bot])

        rp  = b - A @ x
        rd  = c - s - A.T @ y
        rc  = mu_barrier * e_cone - Arw_x @ s
        rhs = np.concatenate([rd, rp, rc])

        # ── Quantum Newton step (HHL) ────────────────────────────────────────
        dz = quantum_newton_solver(KKT_full, rhs, n_clk=n_clk, pad_eig=pad_eig)
        dx = dz[:n_vars]
        dy = dz[n_vars:n_vars + n_cons]
        ds = dz[-n_vars:]

        # ── Step-size selection ──────────────────────────────────────────────
        if use_adaptive_step:
            alpha = fraction_to_boundary_step(x, s, dx, ds, m, scaling=0.95)
        else:
            alpha = alpha_fixed

        x += alpha * dx
        y += alpha * dy
        s += alpha * ds

        # Maintain strict feasibility
        x[0]      = max(x[0], np.linalg.norm(x[1:1 + m]) + 1e-4)
        x[1 + m:] = np.maximum(x[1 + m:], 1e-6)
        s[0]      = max(s[0], np.linalg.norm(s[1:1 + m]) + 1e-4)
        s[1 + m:] = np.maximum(s[1 + m:], 1e-6)

        gap = abs(np.dot(x, s) / r_cones)
        print(f"  Iteration {i}: Alpha = {alpha:.4f}, Duality Gap = {gap:.6f}")
        if gap < tol:
            break

    w_opt = x[1 + m: 1 + m + n]
    return w_opt / np.sum(w_opt)
