# Quantum Interior-Point Portfolio Optimization

This repository implements a **Quantum Interior-Point Method (IPM)** for portfolio optimization, modeled after the theoretical framework by *Kerenidis, Prakash & Szilágyi (2021)*. It contains both a full research implementation and a self-contained educational tutorial.

---

## Repository Structure

```
QApp/
├── research/
│   ├── quantum_ipm_research.ipynb      # Full research notebook (Qiskit 2.x, real market data)
│   └── results/                        # Timestamped JSON logs from research runs
├── workshop/
│   ├── quantum_portfolio_tutorial.ipynb # Self-contained 45-min workshop tutorial
│   └── scripts/
│       ├── fill_solutions.py           # Fill task cells → runnable instructor version
│       └── strip_solutions.py         # Restore `...` skeletons → student version
├── core/                               # Shared quantum building blocks (extracted from research notebook)
├── benchmarks/
│   └── result_logger.py               # Run logger & analyser for the research notebook
├── docs/
│   ├── papers/                         # Reference papers
│   ├── tasks.txt                       # Workshop task design notes
│   └── benchmarking_insights.md
├── requirements.txt
└── README.md
```

---

## Notebooks

### 📗 `workshop/quantum_portfolio_tutorial.ipynb` — Workshop Tutorial

A self-contained notebook for a **~45-minute workshop** (*Applications of Quantum Computing*, Prof. Jeanette Lorenz, LMU Munich). Installs its own dependencies in the first cell. Covers:

1. Live market data via `yfinance` (2024 training, 2025 OOS)
2. Classical SOCP baseline (CVXPY / CLARABEL)
3. Plain HHL on the equality-only KKT system → short positions appear
4. Why HHL alone can't enforce inequality constraints
5. SOCP reformulation + Quantum IPM (HHL as Newton-step subroutine)
6. Three-way comparison + out-of-sample validation

The notebook ships as a **student version** (three task cells contain `...` skeletons). See [Workshop scripts](#workshop-scripts) below.

```bash
jupyter notebook workshop/quantum_portfolio_tutorial.ipynb
```

### 📘 `research/quantum_ipm_research.ipynb` — Full Research Notebook

The production-grade implementation. Fetches real market data via `yfinance`, runs the full Phase Estimation HHL solver, applies an Adaptive Ratio Test step-size, and logs every run to `research/results/`.

```bash
pip install -r requirements.txt
jupyter notebook research/quantum_ipm_research.ipynb
```

Configure via the `CONFIG` block at the top:

```python
CONFIG = {
    "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"],
    "start_date": "2024-01-01",
    "end_date":   "2024-12-31",
    "target_return": 0.20,
    "max_weight": 0.30,
    "quantum_ipm_use_adaptive_step": True,
    "quantum_hhl_n_clk": 8,
}
```

---

## Workshop Scripts

The tutorial ships in **student mode** (task cells have `...` placeholders). Two scripts toggle between modes:

```bash
# Fill task cells → notebook runs end-to-end (instructor / testing mode)
python workshop/scripts/fill_solutions.py

# Restore `...` skeletons → student version for distribution
python workshop/scripts/strip_solutions.py
```

Both scripts identify task cells by their `# TODO` comment patterns, so they are robust to cell reordering.

---

## Features

- **Quantum HHL Solver**: Native Qiskit Phase Estimation circuit using `QFTGate` (Qiskit 2.x compatible)
- **Adaptive Ratio Test**: Dynamic step-size per iteration — boundary-aware, faster convergence
- **Apples-to-Apples Benchmarking**: Both solvers enforce identical constraints (CVXPY CLARABEL vs Quantum IPM)
- **Run Logger** (`benchmarks/result_logger.py`): Research notebook runs saved as timestamped JSON in `research/results/`

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

- Kerenidis, Prakash & Szilágyi (2021) — *"Quantum Algorithms for Portfolio Optimization"*
- Harrow, Hassidim & Lloyd (2009) — *"Quantum Algorithm for Linear Systems of Equations"*
- Boyd & Vandenberghe — *"Convex Optimization"*
