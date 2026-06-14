# Future Work

## TODO — Accelerate the HHL solver: `quantum_info.Statevector` → Aer (CPU)

> **Status: not implemented (research path).** The workshop notebook
> (`workshop/quantum_portfolio_tutorial.ipynb`) has already had this swap applied to its
> self-contained cell-24 solver. The research notebook imports the solver from `core/`
> (cell 6: `from core.hhl import quantum_newton_solver`), so it needs the same change made
> in `core/hhl.py` — deferred to here.

**Why.** Each QIPM Newton step solves a ~16-qubit HHL circuit. Profiling showed ~97% of
wall-clock sits in `qiskit.quantum_info.Statevector` (the pure-NumPy reference simulator,
[core/hhl.py:192](../core/hhl.py#L192)) — ~8.5 s/solve, ~127 s for a 15-iteration solve. A
free Colab GPU does **not** help: that class has no GPU backend, and 16 qubits is well below
the ~22+ qubit break-even where GPU statevector sim beats CPU.

**The change.** Swap to Qiskit Aer's C++ statevector backend on **CPU**:
- `requirements.txt`: add `qiskit-aer>=0.17`.
- [core/hhl.py:16](../core/hhl.py#L16): `from qiskit_aer import AerSimulator`, add `transpile` to
  the `qiskit` import, and create one module-level `_SV_SIM = AerSimulator(method="statevector")`.
- [core/hhl.py:192-196](../core/hhl.py#L192-L196): replace `sv = Statevector(qc)` with
  ```python
  qc_sv = qc.copy(); qc_sv.save_statevector()
  sv_data = _SV_SIM.run(transpile(qc_sv, _SV_SIM, optimization_level=0)).result().get_statevector().data
  raw_data = sv_data[half_dim: half_dim + dim_pad]
  ```
  Keep the `half_dim` / `dim_pad` / `sol_off` slicing unchanged — Aer uses the same little-endian
  ordering. `optimization_level=0` is basis translation only (a simulator needs no routing/gate
  optimization); the default level roughly doubles per-call transpile time for no benefit. An
  explicit `transpile` is required — Aer rejects the raw circuit (`unknown instruction: qft_dg`).

**Measured (on the equivalent workshop circuit, qiskit 2.4.1 + qiskit-aer 0.17.2).**
- Bit-exact: `max|sv_aer − sv_qi| ≈ 3e-13`, so the QIPM convergence trajectory is unchanged.
- ~2.7 s/solve vs ~8.5 s; 15 solves 40.9 s vs ~127 s → **~3.1× end-to-end**. (The headline
  sim-only step is ~5.7×, but the per-call `transpile` is unavoidable, so end-to-end is ~3×.)

**Verification when implemented.** (1) Equivalence gate: one circuit through both engines,
assert `max|sv_aer − sv_qi| < 1e-9`. (2) Run the research notebook QIPM cell; confirm the
per-iter duality-gap / residual table matches the committed run. (3) Run `tests/`.

**Optional, future.** Make the device switchable —
`AerSimulator(method="statevector", device=os.getenv("QAPP_SIM_DEVICE", "CPU"))` — so that *if*
the problem later grows past ~22 qubits, flipping to GPU on Colab becomes a one-env-var change.
Stay on CPU at today's 16 qubits.

---

# Future Work — Notebook Function Extraction

> **Status: completed / fully implemented on main.** This was the roadmap for the
> refactor phase — pulling the research notebook's inline functions into reusable
> `core/`/`benchmarks/` modules. All of the target modules below now exist and are fully implemented
> (`core/risk.py`, `benchmarks/classical.py`, `benchmarks/oos.py`,
> `benchmarks/plots.py`, `benchmarks/reports.py`). The notebook has been updated to import
> and use these modules. Treat this file as a historical record of the roadmap.

Verified against: `research/quantum_ipm_research.ipynb` (37 cells),
`benchmarks/result_logger.py`, `benchmarks/constraint_check.py`,
`core/{hhl,socp,qipm}.py`.

---

## ✅ Confirmed Correct

| Plan claim | Verification |
|---|---|
| Cell 12 does price download + returns + annualisation + covariance | **Confirmed.** Contains full yfinance download, ffill, `returns_daily`, `mu_annual`, `sigma_annual` + regularisation. Good target for `market_data.py`. |
| Cell 15 contains an inline `solve_classical_portfolio_cvxpy` | **Confirmed.** Full CVXPY def is inlined; reads `CONFIG` and outer `assets` (closure). |
| Cell 19 contains covariance factorisation (`M = diag(sqrt(eigvals)) @ eigvecs.T`) | **Confirmed.** Also prints reconstruction error. Good target for `core/risk.py`. |
| Cell 29 is the OOS benchmark (download + metrics + plots) | **Confirmed.** Fully inlined; mirrors `run_benchmark_from_result` in `result_logger.py`. |
| Cell 27 contains a repeated summary table | **Confirmed.** Computes `ret_star, var_star, std_star` and a `pd.DataFrame` summary. |
| `result_logger.run_benchmark_from_result` duplicates OOS metric logic | **Confirmed.** Lines 257–273 are nearly identical to cell 29. |
| `benchmarks.constraint_check` and `benchmarks.result_logger` APIs are stable | **Confirmed.** Both files exist with published signatures. |
| Notebook keeps `CONFIG` | **Confirmed.** Cell 8; plan correctly leaves it in-notebook. |

---

## ⚠️ Issues & Required Corrections

### 1. `solve_classical_portfolio_cvxpy` reads notebook globals — plan must be more explicit

Cell 15's inline function closes over **two** notebook globals, not just `CONFIG`:
- `CONFIG["total_allocation"]`, `CONFIG["default_min_weight"]`, `CONFIG["extra_inequalities"]` — from `CONFIG`
- `assets` — outer notebook variable used inside the constraint loop

The proposed signature:
```python
solve_classical_portfolio_cvxpy(mu, cov, target_return, max_weight, total_allocation=1.0, min_weight=0.0, extra_inequalities=None, assets=None)
```
…correctly covers all of these. **But the `assets` parameter is required (not optional) whenever `extra_inequalities` is non-empty**, matching the same guard already in `constraint_check.build_constraint_report`. State this explicitly.

**Also:** decide which file this goes in — `core/portfolio.py` **or** `benchmarks/classical.py`. Recommendation: **`benchmarks/classical.py`**, because it has a network/data dependency path (via `extra_inequalities` with asset names) and is not a pure algorithmic primitive like `core/hhl.py` or `core/qipm.py`.

### 2. `core/risk.py` — diagnostics spec is inconsistent with what cell 19 already prints

Cell 19 computes and prints:
- Frobenius reconstruction error `||Sigma - M^T M||_F`
- `w^T Sigma w` vs `||Mw||^2` and their absolute difference (only when `cls_ok`)

The conditional `cls_ok`-gated check should **not** be in `factor_covariance_matrix` — it requires `w_star`, a downstream variable. Either:
- (a) keep the `w`-based verification outside `factor_covariance_matrix` (just in the notebook cell), or
- (b) add a separate `verify_factorization(M, w, cov)` helper.

Recommend **option (a)**: `factor_covariance_matrix(cov)` returns `(M, {"recon_error": ..., "eigenvalues": ...})`, and the notebook cell retains the `w`-verification inline.

### 3. `benchmarks/oos.py` — `load_oos_prices` hardcodes the OOS period in cell 29

Cell 29 hardcodes `start="2025-01-01", end="2025-12-31"`. The proposed signature `load_oos_prices(assets, start, end)` correctly parameterises this. **However**, `result_logger.run_benchmark_from_result` parses the OOS period from the result dict (`"2025-01-01 to 2025-12-31"` string). The parsing at line 239:
```python
oos_start, oos_end = [s.strip() for s in test_period.split(" to ")]
```
…should live in `load_oos_prices` or a private helper — not scattered. State this explicitly.

### 4. `benchmarks/plots.py` — `plot_training_diagnostics` covers cell 13

Cell 13 (`## Diagnostics plots`) produces a **normalised-price line plot (left panel)** and a **daily-returns correlation heatmap (right panel, `sns.heatmap(returns_daily.corr())`)** — *not* a daily-return time-series plot. `plot_training_diagnostics(prices, returns_daily)` should reproduce both panels. Cell 13 is a display-only code cell, so it simply becomes a call to that helper.

> Cleanup note: cell 13's left panel is titled `"Normalized Prices (2025)"` but the
> data plotted is the **2024 training** window — a pre-existing mislabel. Fix the
> title to "2024" (or to the configured training range) during extraction.

### 5. `benchmarks/reports.py` — cell 15 has more display logic than first noted

Cell 15 contains a sorted weights table:
```python
display(pd.DataFrame({"asset": assets, "weight": w_star}).sort_values(...))
```
`weights_table(assets, weights)` covers this. ✅

But cell 15 also prints scalar lines (`Solved in Xs`, `Expected annual return:`, `Annual variance:`, `Annual volatility:`). These are **not** covered by `portfolio_metrics(weights, mu_vec, cov)` or `build_solver_summary`. Either `portfolio_metrics` returns a dict the notebook prints, or the scalar prints stay inline. **Resolve this before coding.**

### 6. `run_benchmark_from_result` duplication removal is underspecified

`run_benchmark_from_result` should reuse the shared OOS helpers instead of carrying a second copy of the metric logic. Also address:
- It does the weight-bar-chart + equity-curve visualisation (lines 287–307), overlapping `plot_oos_comparison`.
- The logging cell (cell 33) calls `run_benchmark_from_result(load_result(log_path))` immediately after logging — so the refactored version must still accept a `result` dict and remain callable **without** notebook state.

### 7. Test plan — `load_result_by_id` test is mislabelled

"Unit-test result logger replay through `load_result_by_id` and `run_benchmark_from_result`" is a **smoke/integration** test (needs a fixture JSON on disk), not a unit test. Label it accordingly and create a fixture (`tests/fixtures/sample_run.json`) as part of setup.

### 8. Cell-reference sanity check

| Plan says | Actual notebook cell |
|---|---|
| "cell 12" for data ingestion | Cell 12 ✅ |
| "cell 15" for CVXPY inline solver | Cell 15 ✅ |
| "cell 19" for covariance factorization | Cell 19 ✅ |
| repeated metric tables | Cell 15 (weights table); cell 26 (weights table + scalars); cell 27 (3-column summary) ✅ |
| "cell 29" for OOS benchmarking | Cell 29 ✅ |
| diagnostics plots | Cell 13 ✅ |

All cell references check out.

---

## 🔧 Recommended Pre-Execution Clarifications

1. **Decide: `core/portfolio.py` vs `benchmarks/classical.py`** for the CVXPY wrapper. (Recommend `benchmarks/classical.py`.)
2. **Clarify `factor_covariance_matrix` boundary**: include the `w`-based verification or not? (Recommend not.)
3. **Clarify scalar-print ownership** in cell 15: does `portfolio_metrics` return a printed dict, or do scalar prints stay in-notebook?
4. **Document the OOS period string format** (`"YYYY-MM-DD to YYYY-MM-DD"`) as a convention, and move parsing into `load_oos_prices` or a shared util.
5. **Create a test fixture JSON** for the `run_benchmark_from_result` replay test.

---

## Summary

The plan is **structurally sound** and well-grounded in the actual notebook. All
cell references, variable names (`prices`, `returns_daily`, `mu_annual`,
`sigma_annual`, `assets`, `mu_vec`, `cov`, `M`, `w_star`, `w_ipm_final`), and API
compatibility claims are verified correct. The issues above are mostly
precision/clarity gaps rather than logical errors — none are blockers, but items
1, 2, and 3 should be resolved before writing code to avoid mid-implementation
decisions.
