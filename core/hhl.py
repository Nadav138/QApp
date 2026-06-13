"""
core/hhl.py
===========
Custom HHL (Harrow-Hassidim-Lloyd) quantum linear-system solver.

The paper assumes Hermitian linear-system input for quantum linear algebra.
Symmetric systems are used directly; non-symmetric systems are embedded as
sym(K) = [[0, K], [K.T, 0]] and the solution is read from the second block.
"""

import numpy as np
import scipy.linalg

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate, RYGate
from qiskit.quantum_info import Operator, Statevector


# Module-level handle to the most recent circuit, mutated inside the solver.
last_qc: QuantumCircuit | None = None


def _is_symmetric(mat, atol=1e-10, rtol=1e-10):
    return np.allclose(mat, mat.T, atol=atol, rtol=rtol)


def _embed_hermitian_if_needed(K_mat, r_vec, symmetrization):
    """Return Hermitian work system plus solution offset for the original solve."""
    input_is_symmetric = _is_symmetric(K_mat)
    d_in = K_mat.shape[0]

    if symmetrization == "auto":
        if input_is_symmetric:
            return K_mat, r_vec, 0, input_is_symmetric
    elif symmetrization in ("none", None):
        if not input_is_symmetric:
            raise ValueError("K_mat is not symmetric. Use symmetrization='auto' or 'block'.")
        return K_mat, r_vec, 0, input_is_symmetric
    elif symmetrization != "block":
        raise ValueError("symmetrization must be one of: 'auto', 'none', 'block'.")

    zeros = np.zeros((d_in, d_in))
    K_herm = np.block([[zeros, K_mat], [K_mat.T, zeros]])
    r_work = np.concatenate([r_vec, np.zeros(d_in)])
    return K_herm, r_work, d_in, input_is_symmetric


def _controlled_system_on_clock_one(U, I_dim, P0, P1):
    """Qiskit little-endian controlled-U matrix, with the clock qubit as LSB.

    This is Convention B, verified against Qiskit's controlled-X matrix:
        kron(I_sys, |0><0|) + kron(U_sys, |1><1|)
    The operator must be appended to qubits [clock] + system_qubits.
    """
    return np.kron(I_dim, P0) + np.kron(U, P1)


def quantum_newton_solver(
    K_mat,
    r_vec,
    n_clk=4,
    pad_eig=0.1,
    symmetrization="auto",
    return_diagnostics=False,
):
    """
    Approximately solve ``K @ dz = r`` using a simulated HHL circuit.

    Parameters
    ----------
    K_mat : np.ndarray
        Real square matrix. Symmetric matrices are used directly. Non-symmetric
        matrices are embedded as ``[[0, K], [K.T, 0]]`` by default, following
        the paper's quantum linear-algebra reduction.
    r_vec : np.ndarray
        Right-hand side vector.
    n_clk : int
        Number of QPE clock-register qubits; precision is roughly ``2**-n_clk``.
    pad_eig : float
        Identity-padding eigenvalue when zero-padding K to ``2**n_sys``.
    symmetrization : {"auto", "none", "block"}
        ``auto`` embeds only non-symmetric systems; ``none`` requires symmetric
        input; ``block`` always embeds.
    return_diagnostics : bool
        If True, return ``(solution, diagnostics)``.

    Notes
    -----
    The QPE controlled unitaries are built directly using Qiskit's little-endian
    Convention B and repeated squaring. This avoids synthesizing dense
    ``.control(1)`` unitaries and requires only one matrix exponential.
    """
    global last_qc

    K_input = np.asarray(K_mat, dtype=float)
    r_input = np.asarray(r_vec, dtype=float)
    if K_input.ndim != 2 or K_input.shape[0] != K_input.shape[1]:
        raise ValueError("K_mat must be a square matrix.")
    if r_input.shape != (K_input.shape[0],):
        raise ValueError("r_vec must be a vector with length matching K_mat.")

    K_herm, r_work, sol_off, input_is_symmetric = _embed_hermitian_if_needed(
        K_input, r_input, symmetrization
    )
    dim = len(K_herm)
    n_sys = int(np.ceil(np.log2(dim)))
    dim_pad = 2 ** n_sys

    K_pad = np.eye(dim_pad) * pad_eig
    K_pad[:dim, :dim] = K_herm

    r_pad = np.zeros(dim_pad)
    r_pad[:len(r_work)] = r_work
    r_norm = np.linalg.norm(r_pad)

    diagnostics = {
        "input_dim": int(K_input.shape[0]),
        "effective_dim": int(dim),
        "padded_dim": int(dim_pad),
        "solution_offset": int(sol_off),
        "input_is_symmetric": bool(input_is_symmetric),
        "embedded_system": bool(sol_off > 0),
        "n_sys": int(n_sys),
        "n_clk": int(n_clk),
        "clock_qubits": int(n_clk),
        "r_norm": float(r_norm),
    }

    if r_norm < 1e-12:
        solution = np.zeros(dim - sol_off)
        if return_diagnostics:
            return solution, diagnostics
        return solution

    r_normalized = r_pad / r_norm

    eig_max = np.max(np.abs(np.linalg.eigvalsh(K_pad)))
    t = np.pi / (eig_max * 1.2)
    diagnostics["eig_max"] = float(eig_max)
    diagnostics["evolution_time"] = float(t)

    qr_sys = QuantumRegister(n_sys, "sys")
    qr_clk = QuantumRegister(n_clk, "clk")
    qr_anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(qr_sys, qr_clk, qr_anc)
    last_qc = qc

    qc.initialize(r_normalized, qr_sys)

    # One expm, then powers U^(2^i) by repeated squaring.
    U_base = scipy.linalg.expm(1j * K_pad * t)
    U_pows = [U_base.copy()]
    for _ in range(n_clk - 1):
        U_pows.append(U_pows[-1] @ U_pows[-1])

    P0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    I_dim = np.eye(dim_pad, dtype=complex)

    # Phase estimation.
    qc.h(qr_clk)
    for i, U_power in enumerate(U_pows):
        cU = _controlled_system_on_clock_one(U_power, I_dim, P0, P1)
        qc.append(Operator(cU), [qr_clk[i]] + list(qr_sys))
    qc.append(QFTGate(n_clk).inverse(), qr_clk)

    # Eigenvalue inversion.
    C = 0.05
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

    # Uncompute phase estimation using adjoints of the squared powers.
    qc.append(QFTGate(n_clk), qr_clk)
    for i in reversed(range(n_clk)):
        cU_inv = _controlled_system_on_clock_one(U_pows[i].conj().T, I_dim, P0, P1)
        qc.append(Operator(cU_inv), [qr_clk[i]] + list(qr_sys))
    qc.h(qr_clk)

    sv = Statevector(qc)
    half_dim = 2 ** (n_sys + n_clk)
    raw_data = sv.data[half_dim: half_dim + dim_pad]
    dz = np.real(raw_data)[:dim] * (r_norm / C)
    solution = dz[sol_off:]

    diagnostics["num_qubits"] = int(qc.num_qubits)
    diagnostics["circuit_depth"] = int(qc.depth())
    diagnostics["circuit_size"] = int(qc.size())
    if return_diagnostics:
        return solution, diagnostics
    return solution
