"""
core/socp.py
============
SOCP (Second-Order Cone Program) cone-algebra utilities used by the Quantum IPM.

Functions
---------
- arrowhead_product(v, m, n)
        Block-diagonal arrowhead-matrix operator for the cone product L^m × (L^0)^{2n}.

- fraction_to_boundary_step(x, s, dx, ds, m, scaling=0.95)
        Maximum step size that keeps (x + α·dx, s + α·ds) strictly inside
        the cone product. Implements the standard fraction-to-boundary rule
        across one Lorentz cone L^m (block: t₀, t̃) and 2n non-negativity cones L^0.
"""

import numpy as np


def arrowhead_product(v, m, n):
    """
    Construct the block-diagonal arrowhead matrix for the cone product
    ``L^m × (L^0)^{2n}``.

    Parameters
    ----------
    v : np.ndarray
        Full SOCP variable vector, length ``1 + m + 2n``, with blocks
        ``[t₀; t̃ (size m); x (size n); s_max (size n)]``.
    m : int
        Lorentz-cone dimension (= rank of the risk-factor matrix M
        = number of risk eigenvalues).
    n : int
        Number of portfolio weights.

    Returns
    -------
    np.ndarray
        Square arrowhead matrix of shape ``(len(v), len(v))``.

    Block structure
    ---------------
    L^m  block :  Arw(t)  = [[t₀, t̃ᵀ], [t̃, t₀·I_m]]
    L^0  blocks:  Arw(xᵢ) = xᵢ                    (scalar — non-negativity cone)
    """
    t   = v[:1 + m]
    x_s = v[1 + m:]

    t0, t_bar = t[0], t[1:].reshape(-1, 1)
    Arw_t = np.vstack([
        np.hstack([[[t0]], t_bar.T]),
        np.hstack([t_bar, t0 * np.eye(m)]),
    ])
    Arw_x_s = np.diag(x_s)

    Arw = np.zeros((len(v), len(v)))
    Arw[:1 + m, :1 + m] = Arw_t
    Arw[1 + m:, 1 + m:] = Arw_x_s
    return Arw


def fraction_to_boundary_step(x, s, dx, ds, m, scaling=0.95):
    """
    Compute the largest step size ``α ∈ (0, 1]`` such that
    ``x + α·dx`` and ``s + α·ds`` stay strictly inside the cone product
    ``L^m × (L^0)^{2n}``.

    Implements the standard IPM fraction-to-boundary rule:

    1. For each L^0 (non-negativity) block, the largest α keeping the
       coordinate non-negative is ``-coord / direction`` whenever the
       direction is negative.
    2. For the L^m (Lorentz) block, solve a quadratic in α arising from
       the condition ``(t₀ + α·dt₀)² ≥ ‖t̃ + α·dt̃‖²``.

    The final step is ``min(1.0, scaling · min(all the αs))`` — the
    ``scaling`` factor (default 0.95) keeps the iterate strictly interior.

    Parameters
    ----------
    x, s   : np.ndarray   Current primal and dual iterates.
    dx, ds : np.ndarray   Newton directions for ``x`` and ``s``.
    m      : int          Lorentz-cone dimension.
    scaling: float        Boundary-pullback factor, default 0.95.

    Returns
    -------
    float
        Step size ``α ∈ (0, 1]``.
    """
    # ── L^0 cones: non-negativity for x[1+m:] and s[1+m:] ────────────────────
    alpha_L0_x = 1.0
    idx_neg_dx = np.where(dx[1 + m:] < 0)[0]
    if len(idx_neg_dx) > 0:
        alpha_L0_x = float(np.min(-x[1 + m:][idx_neg_dx] / dx[1 + m:][idx_neg_dx]))

    alpha_L0_s = 1.0
    idx_neg_ds = np.where(ds[1 + m:] < 0)[0]
    if len(idx_neg_ds) > 0:
        alpha_L0_s = float(np.min(-s[1 + m:][idx_neg_ds] / ds[1 + m:][idx_neg_ds]))

    alpha_L0 = min(alpha_L0_x, alpha_L0_s)

    # ── L^m cone (primal): (t₀ + α·dx₀)² ≥ ‖t̃ + α·dt‖² ──────────────────────
    dx0, dt   = dx[0], dx[1:1 + m]
    t0, t_bar = x[0], x[1:1 + m]
    a_p = dx0**2 - np.dot(dt, dt)
    b_p = 2 * (t0 * dx0 - np.dot(t_bar, dt))
    c_p = t0**2 - np.dot(t_bar, t_bar)
    alpha_Lm_x = 1.0
    if a_p < 0 or b_p < 0:
        roots = [r.real for r in np.roots([a_p, b_p, c_p])
                 if np.isreal(r) and r.real > 0]
        if roots:
            alpha_Lm_x = min(roots)

    # ── L^m cone (dual): same quadratic on (s₀, s̃) ──────────────────────────
    ds0, dst  = ds[0], ds[1:1 + m]
    s0, s_bar = s[0], s[1:1 + m]
    a_d = ds0**2 - np.dot(dst, dst)
    b_d = 2 * (s0 * ds0 - np.dot(s_bar, dst))
    c_d = s0**2 - np.dot(s_bar, s_bar)
    alpha_Lm_s = 1.0
    if a_d < 0 or b_d < 0:
        roots = [r.real for r in np.roots([a_d, b_d, c_d])
                 if np.isreal(r) and r.real > 0]
        if roots:
            alpha_Lm_s = min(roots)

    return min(1.0, scaling * min(alpha_L0, alpha_Lm_x, alpha_Lm_s))
