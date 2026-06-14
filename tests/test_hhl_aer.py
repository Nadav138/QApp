"""
tests/test_hhl_aer.py
=====================
Equivalence and regression tests for the Aer-backed HHL solver in core/hhl.py.

These tests verify that switching from qiskit.quantum_info.Statevector to
Qiskit Aer's C++ statevector backend produces bit-exact (within floating-point
tolerance) results, and that the solver correctly solves small known systems.
"""

import unittest

import numpy as np


class TestHHLAerEquivalence(unittest.TestCase):
    """Verify Aer statevector matches the old Statevector engine."""

    def test_aer_vs_statevector_equivalence(self):
        """Build a small HHL circuit, simulate through both engines,
        and assert max|sv_aer - sv_qi| < 1e-9."""
        import scipy.linalg
        from qiskit import QuantumCircuit, QuantumRegister, transpile
        from qiskit.circuit.library import QFTGate, RYGate
        from qiskit.quantum_info import Operator, Statevector
        from qiskit_aer import AerSimulator

        # Small 2x2 symmetric positive-definite system.
        K = np.array([[2.0, 0.5], [0.5, 3.0]])
        r = np.array([1.0, 0.5])

        n_sys = 1
        n_clk = 3
        pad_eig = 0.1
        dim = 2
        dim_pad = 2 ** n_sys

        K_pad = np.eye(dim_pad) * pad_eig
        K_pad[:dim, :dim] = K

        r_pad = np.zeros(dim_pad)
        r_pad[:len(r)] = r
        r_norm = np.linalg.norm(r_pad)
        r_normalized = r_pad / r_norm

        eig_max = np.max(np.abs(np.linalg.eigvalsh(K_pad)))
        t = np.pi / (eig_max * 1.2)

        from core.hhl import _controlled_system_on_clock_one

        qr_sys = QuantumRegister(n_sys, "sys")
        qr_clk = QuantumRegister(n_clk, "clk")
        qr_anc = QuantumRegister(1, "anc")
        qc = QuantumCircuit(qr_sys, qr_clk, qr_anc)

        qc.initialize(r_normalized, qr_sys)

        U_base = scipy.linalg.expm(1j * K_pad * t)
        U_pows = [U_base.copy()]
        for _ in range(n_clk - 1):
            U_pows.append(U_pows[-1] @ U_pows[-1])

        P0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        P1 = np.array([[0.0, 0.0], [0.0, 1.0]])
        I_dim = np.eye(dim_pad, dtype=complex)

        qc.h(qr_clk)
        for i, U_power in enumerate(U_pows):
            cU = _controlled_system_on_clock_one(U_power, I_dim, P0, P1)
            qc.append(Operator(cU), [qr_clk[i]] + list(qr_sys))
        qc.append(QFTGate(n_clk).inverse(), qr_clk)

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

        qc.append(QFTGate(n_clk), qr_clk)
        for i in reversed(range(n_clk)):
            cU_inv = _controlled_system_on_clock_one(
                U_pows[i].conj().T, I_dim, P0, P1
            )
            qc.append(Operator(cU_inv), [qr_clk[i]] + list(qr_sys))
        qc.h(qr_clk)

        # --- Old engine: qiskit.quantum_info.Statevector ---
        sv_qi = Statevector(qc).data

        # --- New engine: Aer C++ statevector ---
        sim = AerSimulator(method="statevector")
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        sv_aer = (
            sim.run(transpile(qc_sv, sim, optimization_level=0))
            .result()
            .get_statevector()
            .data
        )

        # Assert bit-exact equivalence within floating-point tolerance.
        max_diff = np.max(np.abs(sv_aer - sv_qi))
        self.assertLess(
            max_diff,
            1e-9,
            f"Statevector mismatch: max|sv_aer - sv_qi| = {max_diff:.2e}",
        )


class TestQuantumNewtonSolverAer(unittest.TestCase):
    """Verify quantum_newton_solver produces correct results with Aer backend."""

    def test_solver_returns_approximate_solution(self):
        """Solve a small symmetric system and verify approximate correctness."""
        from core.hhl import quantum_newton_solver

        K = np.array([[4.0, 1.0], [1.0, 3.0]])
        r = np.array([1.0, 2.0])

        solution, diag = quantum_newton_solver(
            K, r, n_clk=4, return_diagnostics=True
        )

        # The HHL solution should approximate np.linalg.solve(K, r).
        exact = np.linalg.solve(K, r)
        # HHL with limited clock qubits won't be exact, but should be
        # in the right ballpark (within ~50% for n_clk=4).
        rel_error = np.linalg.norm(solution - exact) / np.linalg.norm(exact)
        self.assertLess(
            rel_error,
            0.5,
            f"HHL solution too far from exact: rel_error = {rel_error:.4f}",
        )

    def test_diagnostics_contain_simulator_backend(self):
        """Verify the diagnostics dict contains the new simulator_backend field."""
        from core.hhl import quantum_newton_solver

        K = np.array([[2.0, 0.0], [0.0, 2.0]])
        r = np.array([1.0, 1.0])

        _, diag = quantum_newton_solver(
            K, r, n_clk=3, return_diagnostics=True
        )

        self.assertIn("simulator_backend", diag)
        self.assertEqual(diag["simulator_backend"], "aer_statevector_cpu")

    def test_diagnostics_contain_standard_fields(self):
        """Verify diagnostics still contain all expected fields."""
        from core.hhl import quantum_newton_solver

        K = np.array([[2.0, 0.0], [0.0, 2.0]])
        r = np.array([1.0, 1.0])

        _, diag = quantum_newton_solver(
            K, r, n_clk=3, return_diagnostics=True
        )

        expected_fields = [
            "input_dim", "effective_dim", "padded_dim",
            "solution_offset", "input_is_symmetric", "embedded_system",
            "n_sys", "n_clk", "clock_qubits", "r_norm",
            "eig_max", "evolution_time",
            "simulator_backend", "num_qubits", "circuit_depth", "circuit_size",
        ]
        for field in expected_fields:
            self.assertIn(field, diag, f"Missing diagnostics field: {field}")

    def test_optimization_level_execution(self):
        """Verify quantum_newton_solver accepts different optimization levels."""
        from core.hhl import quantum_newton_solver

        K = np.array([[2.0, 0.0], [0.0, 2.0]])
        r = np.array([1.0, 1.0])

        # Run with level 0
        sol_0, _ = quantum_newton_solver(K, r, n_clk=3, return_diagnostics=True, optimization_level=0)
        # Run with level 1
        sol_1, _ = quantum_newton_solver(K, r, n_clk=3, return_diagnostics=True, optimization_level=1)

        # The results should be identical.
        np.testing.assert_allclose(sol_0, sol_1, atol=1e-12)



if __name__ == "__main__":
    unittest.main()
