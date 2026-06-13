"""
core/risk.py
============
Risk-factor utilities shared by the research notebook and benchmarks.
"""

from __future__ import annotations

import numpy as np


def factor_covariance_matrix(cov: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Factor a covariance matrix as ``cov ~= M.T @ M``.

    The research SOCP uses the constraint ``t_tilde = M @ w``, so the factor
    convention must satisfy ``w.T @ cov @ w == ||M @ w||_2**2``.

    Parameters
    ----------
    cov : np.ndarray
        Symmetric covariance matrix.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``M`` and diagnostics containing the original eigenvalues, clipped
        eigenvalues, and Frobenius reconstruction error.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix.")

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    M = np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T
    reconstructed = M.T @ M

    diagnostics = {
        "recon_error": float(np.linalg.norm(cov - reconstructed, ord="fro")),
        "eigenvalues": eigvals,
        "eigenvalues_clipped": eigvals_clipped,
    }
    return M, diagnostics
