"""
core/hhl.py
===========
Custom HHL (Harrow-Hassidim-Lloyd) quantum linear-system solver.

Public API
----------
- quantum_newton_solver(K_mat, r_vec, n_clk=4, pad_eig=0.1)
        Solve K·dz = r approximately on a simulated Qiskit circuit.

- last_qc : module-level global
        The most recent HHL circuit built by ``quantum_newton_solver``.
        Access via ``from core import hhl; hhl.last_qc`` to always read
        the current value (binding it at import time would capture ``None``).
"""

import numpy as np
import scipy.linalg

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate, RYGate
from qiskit.quantum_info import Operator, Statevector


# Module-level handle to the most recent circuit, mutated inside the solver.
last_qc: QuantumCircuit | None = None


def quantum_newton_solver(K_mat, r_vec, n_clk=4, pad_eig=0.1):
    """
    Approximately solve the linear system ``K · dz = r`` using a Qiskit HHL
    circuit (Phase Estimation + eigenvalue inversion + uncompute).

    Parameters
    ----------
    K_mat   : np.ndarray   Real square matrix (will be Hermitianised internally).
    r_vec   : np.ndarray   Right-hand side vector.
    n_clk   : int          Number of clock qubits for Phase Estimation (default 4).
    pad_eig : float        Identity-padding eigenvalue when zero-padding K to size
                           ``2^n_sys`` (default 0.1).

    Returns
    -------
    np.ndarray
        Real approximation of ``dz`` (length = original ``len(K_mat)``).

    Side-effects
    ------------
    Sets the module-level ``core.hhl.last_qc`` to the constructed circuit.
    """
    global last_qc

    # ── 1. Hermitianise, pad to power-of-two dimension ───────────────────────
    K_herm  = (K_mat + K_mat.T) / 2
    dim     = len(K_herm)
    n_sys   = int(np.ceil(np.log2(dim)))
    dim_pad = 2 ** n_sys

    K_pad = np.eye(dim_pad) * pad_eig
    K_pad[:dim, :dim] = K_herm

    r_pad = np.zeros(dim_pad)
    r_pad[:len(r_vec)] = r_vec
    r_norm = np.linalg.norm(r_pad)
    if r_norm < 1e-12:
        return np.zeros(dim)
    r_normalized = r_pad / r_norm

    # ── 2. Scale Hamiltonian evolution time so phases stay in (-π, π) ────────
    eig_max = np.max(np.abs(np.linalg.eigvalsh(K_pad)))
    t = np.pi / (eig_max * 1.2)

    # ── 3. Build registers and circuit ───────────────────────────────────────
    qr_sys = QuantumRegister(n_sys, "sys")
    qr_clk = QuantumRegister(n_clk, "clk")
    qr_anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(qr_sys, qr_clk, qr_anc)
    last_qc = qc  # expose to callers (e.g., notebook visualization)

    # ── 4. State preparation ─────────────────────────────────────────────────
    qc.initialize(r_normalized, qr_sys)

    # ── 5. Quantum Phase Estimation ──────────────────────────────────────────
    qc.h(qr_clk)
    for i in range(n_clk):
        power = 2 ** i
        U_power = Operator(scipy.linalg.expm(1j * K_pad * t * power))
        cU = U_power.to_instruction().control(1)
        qc.append(cU, [qr_clk[i]] + list(qr_sys))
    qc.append(QFTGate(n_clk).inverse(), qr_clk)

    # ── 6. Eigenvalue inversion (controlled RY on ancilla) ───────────────────
    C = 0.05  # scale factor for ancilla rotation
    for x in range(1, 2 ** n_clk):
        bin_str = format(x, f"0{n_clk}b")

        phase = x / (2 ** n_clk)
        if phase >= 0.5:
            phase -= 1.0
        lmbda = phase * (2 * np.pi) / t
        if np.abs(lmbda) < 1e-5:
            continue

        ratio = C / lmbda
        if np.abs(ratio) > 1.0:
            ratio = np.sign(ratio)

        theta = 2 * np.arcsin(ratio)
        mcry = RYGate(theta).control(n_clk, ctrl_state=bin_str)
        qc.append(mcry, list(qr_clk) + [qr_anc[0]])

    # ── 7. Uncompute Phase Estimation ────────────────────────────────────────
    qc.append(QFTGate(n_clk), qr_clk)
    for i in reversed(range(n_clk)):
        power = 2 ** i
        U_power_inv = Operator(scipy.linalg.expm(-1j * K_pad * t * power))
        cU_inv = U_power_inv.to_instruction().control(1)
        qc.append(cU_inv, [qr_clk[i]] + list(qr_sys))
    qc.h(qr_clk)

    # ── 8. Extract solution from the post-selected statevector ───────────────
    sv = Statevector(qc)
    half_dim = 2 ** (n_sys + n_clk)
    raw_data = sv.data[half_dim: half_dim + dim_pad]
    dz = np.real(raw_data)[:dim] * (r_norm / C)
    return dz
