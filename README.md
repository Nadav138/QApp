# Quantum Interior-Point Portfolio Optimization

A **Quantum Interior-Point Method (IPM)** for constrained Markowitz portfolio
optimization, implemented after *Kerenidis, Prakash & Szilágyi (2019),
"Quantum Algorithms for Portfolio Optimization"*. The repository pairs a
production-grade research notebook with a self-contained educational tutorial,
backed by a small set of reusable quantum building blocks.

---

## About

The constrained portfolio problem (budget, long-only, diversification cap, and a
return target) is reduced to a **Second-Order Cone Program (SOCP)** and solved
with a **short-step interior-point method** whose Newton step is computed by a
**quantum linear-system solver (HHL)** with Phase Estimation. The paper is treated
as the source of truth: the SOCP reduction (Eq. 5), the Newton system (Eq. 6),
the Hermitian dilation used for quantum linear algebra (§6.1), and the short-step
contraction rate `σ = 1 − 0.1/√r` are all reproduced directly.

The work was developed for the graduate course *Applications of Quantum
Computing* (PD Dr. Jeanette Lorenz, LMU Munich). Shared logic was extracted from
the research notebook into a modular `core/` package; the workshop tutorial is
kept deliberately self-contained for teaching. Every research run is logged to
JSON for reproducibility, and findings and known limitations are tracked in
[`workshop/insights.md`](workshop/insights.md) and
[`research/FUTURE_WORK.md`](research/FUTURE_WORK.md).

---

## Repository Structure

```
QApp/
├── core/                                # Reusable quantum building blocks
│   ├── hhl.py                           # HHL solver (Phase Estimation; Hermitian dilation for non-symmetric systems)
│   ├── qipm.py                          # Short-step quantum IPM for the Markowitz SOCP
│   └── socp.py                          # SOCP cone algebra (arrowhead operator, fraction-to-boundary step)
├── benchmarks/
│   ├── constraint_check.py              # Portfolio constraint-validation report
│   └── result_logger.py                 # Run logger, replay, and analysis
├── research/
│   ├── quantum_ipm_research.ipynb       # Full research notebook (Qiskit 2.x, real market data)
│   ├── results/                         # Timestamped JSON logs from research runs
│   └── FUTURE_WORK.md                   # Roadmap for further notebook → module extraction
├── workshop/
│   ├── quantum_portfolio_tutorial.ipynb         # Self-contained ~45-min workshop tutorial (Qiskit)
│   ├── classiq_quantum_portfolio_tutorial.ipynb # Classiq/Qmod variant of the tutorial
│   ├── insights.md                      # Engineering & correctness notes
│   └── scripts/                         # Toggle student ↔ instructor versions (Qiskit + Classiq)
├── docs/
│   ├── papers/                          # Reference papers
│   └── benchmarking_insights.md
├── requirements.txt
└── README.md
```

---

## Notebooks

### 📗 `workshop/quantum_portfolio_tutorial.ipynb` — Workshop Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nadav138/QApp/blob/main/workshop/quantum_portfolio_tutorial.ipynb)

A self-contained notebook for a **~45-minute workshop**. It installs its own
dependencies in the first cell and walks through:

1. Live market data via `yfinance` (2024 training, 2025 out-of-sample)
2. Classical SOCP baseline (CVXPY / CLARABEL)
3. Plain HHL on the equality-only KKT system → short positions appear
4. Why HHL alone cannot enforce inequality constraints
5. SOCP reformulation + Quantum IPM (HHL as the Newton-step subroutine)
6. Three-way comparison + out-of-sample validation

The notebook ships as a **student version** (three task cells contain `...`
skeletons). See [Workshop Scripts](#workshop-scripts).

```bash
jupyter notebook workshop/quantum_portfolio_tutorial.ipynb
```

### 📘 `research/quantum_ipm_research.ipynb` — Full Research Notebook

The production-grade implementation. It imports the solvers from `core/`, fetches
real market data via `yfinance`, runs the full Phase-Estimation HHL Newton step
with a boundary-aware (fraction-to-boundary) step size, and logs every run to
`research/results/`.

```bash
pip install -r requirements.txt
jupyter notebook research/quantum_ipm_research.ipynb
```

Configure via the `CONFIG` block at the top:

```python
CONFIG = {
    "tickers":       ["AAPL", "INTC", "NVDA", "AMZN", "META", "GOOGL", "TSLA"],
    "start_date":    "2024-01-01",
    "end_date":      "2024-12-31",
    "target_return": 0.35,   # paper formulation: muᵀw = target_return
    "max_weight":    0.30,   # diversification cap
    "quantum_hhl_n_clk": 8,  # QPE clock qubits (eigenvalue resolution)
}
```

---

## Workshop Scripts

The tutorial ships in **student mode** (task cells have `...` placeholders). Two
scripts toggle between modes by matching `# TODO` patterns, so they are robust to
cell reordering (Classiq variants provided as well):

```bash
# Fill task cells → notebook runs end-to-end (instructor / testing mode)
python workshop/scripts/fill_solutions.py

# Restore `...` skeletons → student version for distribution
python workshop/scripts/strip_solutions.py
```

---

## Method & Validation

- **Paper-driven.** The SOCP reduction, Newton system, Hermitian dilation, and
  short-step rate match the reference paper; the research notebook, the workshop
  tutorial, and the `core/` modules are kept algorithmically consistent.
- **Honest iterates.** Weights are returned raw — neither clipped nor
  renormalized — so reported infeasibility reflects the true mid-path iterate
  (paper Thm 6.6), rather than being masked by post-processing.
- **Reproducible runs.** Each research run is persisted as timestamped JSON in
  `research/results/` and can be replayed without notebook state.

---

## Features

- **Quantum HHL solver** — native Qiskit Phase-Estimation circuit using `QFTGate`
  (Qiskit 2.x), with Hermitian dilation `sym(K) = [[0, K], [Kᵀ, 0]]` for the
  non-symmetric Newton system.
- **Boundary-aware step size** — per-iteration fraction-to-boundary rule across
  the Lorentz and non-negativity cones keeps iterates strictly feasible.
- **Apples-to-apples benchmarking** — the classical (CVXPY / CLARABEL) and quantum
  solvers enforce identical constraints.
- **Run logger** ([`benchmarks/result_logger.py`](benchmarks/result_logger.py)) —
  research runs saved as timestamped JSON, with replay and summary helpers.

---

## Analysing Results

```python
from benchmarks.result_logger import summarise_runs
import pandas as pd

df = pd.DataFrame(summarise_runs())
print(df[["run_id", "n_clk", "cls_oos_pct", "quantum_oos_pct", "oos_gap_pct"]])
```

---

## References

- Kerenidis, Prakash & Szilágyi (2019) — *"Quantum Algorithms for Portfolio Optimization"* (arXiv:1908.08040; ACM AFT 2019)
- Harrow, Hassidim & Lloyd (2009) — *"Quantum Algorithm for Linear Systems of Equations"*
- Boyd & Vandenberghe — *"Convex Optimization"*
