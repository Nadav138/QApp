# Workshop Insights

- **Hermitian dilation, not arithmetic averaging:** Paper §6.1 prescribes `sym(K)=[[0,K],[Kᵀ,0]]` for non-symmetric systems. Averaging `(K+Kᵀ)/2` solves a different system; it happens to give plausible directions here but is wrong.
- **Equality return target:** Paper uses `μᵀx = R` (Eq. 2, 5, 9). The `≥` form is numerically identical for this data (constraint binds) but conceptually wrong.
- **Gap "floor" = iteration budget, not HHL noise:** σ=0.9698 at r=11 → ~112 iters to 1e-2. Stalling in 6-20 steps is exactly predicted; there is no noise floor.
- **n_clk=5 with dilation tracks theory exactly:** Measured σ=0.9695 vs 0.9698 theory. Algorithm is correct; old slowness was a Qiskit overhead issue, not a physics limit.
- **n_clk=4 with dilation gave bad directions:** Inadequate resolution for the 12-qubit system — eigenvalues alias. n_clk=4 with averaging appeared to work because the averaged (smaller) system was easier to resolve.
- **Bottleneck: `.control(1)` synthesis, not simulation:** `Operator(U).to_instruction().control(1)` = 12.75 s/call at dim=64; Statevector = 2.21 s. 85% of runtime was Qiskit synthesising dense controlled unitaries in Python — not quantum math.
- **Fix: build controlled-U as a direct numpy matrix — 230× speedup:** `np.kron(I,|0><0|) + np.kron(U,|1><1|)`. Verified correct (matches Qiskit's little-endian CX convention). Makes 20 live iterations feasible in ~20 s.
- **Squaring optimization alone saves nothing:** expm was always negligible; synthesis was the bottleneck. Combine with direct-matrix fix for code cleanliness, not for speed.
- **GPU didn't help:** Bottleneck was Python/Qiskit gate-synthesis overhead, not BLAS. GPUs accelerate linear algebra, not Python call overhead.
- **Clipping/renormalization manufactured false convergence:** Post-step clipping and `w/sum(w)` masked the true mid-path iterate. Raw weights with `sum(w)≠1` are the honest output (paper Thm 6.6).
- **Final QIPM Newton circuit is structurally identical to §3 HHL:** Same 4 stages (state-prep → QPE → eigenvalue-inversion → uncompute QPE), just on a larger register. The 44×44 Newton matrix dilates to 88×88, pads to 128×128 → n_sys=7; with n_clk=8 the full circuit is 16 qubits. Captured via `hhl.last_qc` after the final IPM iteration.
