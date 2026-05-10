# Quantum Interior-Point Portfolio Optimization

This repository implements a **Quantum Interior-Point Method (IPM)** for portfolio optimization, modeled after the theoretical framework by *Kerenidis, Prakash & Szilágyi (2021)*. It contains both a full research implementation and a self-contained educational tutorial.

---

## Repository Structure

```
QApp/
├── quantum_ipm_research.ipynb         # Full research notebook (Qiskit 2.x, real market data)
├── quantum_portfolio_tutorial.ipynb   # Self-contained 1-hour workshop tutorial
├── utils/
│   └── result_logger.py               # Standalone run logger & analyser
├── results/                           # Timestamped JSON logs from each run
├── requirements.txt                   # Full deps for the research notebook
├── tutorial_requirements.txt          # Minimal deps for the tutorial
└── README.md
```

---

## Notebooks

### 📗 `quantum_portfolio_tutorial.ipynb` — Workshop Tutorial
A self-contained, beginner-friendly notebook for a **~1-hour workshop** (e.g., *Applications of Quantum Computing*, Prof. Jeanette Lorenz). Uses synthetic data embedded directly — no internet required. Covers:

1. Markowitz SOCP formulation
2. Classical Interior-Point baseline (CVXPY / CLARABEL)
3. HHL quantum linear-system theory
4. Simulated Quantum IPM with Qiskit
5. Out-of-sample validation
6. Discussion & extensions

```bash
pip install -r tutorial_requirements.txt
jupyter notebook quantum_portfolio_tutorial.ipynb
```

### 📘 `quantum_ipm_research.ipynb` — Full Research Notebook
The production-grade implementation. Fetches real market data via `yfinance`, runs the full 13-qubit Phase Estimation HHL solver, applies the Adaptive Ratio Test step-size, and logs every run to `results/`.

```bash
pip install -r requirements.txt
jupyter notebook quantum_ipm_research.ipynb
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

## Features

- **Quantum HHL Solver**: Native Qiskit Phase Estimation circuit using `QFTGate` (Qiskit 2.x compatible)
- **Adaptive Ratio Test**: Dynamic step-size per iteration — boundary-aware, faster convergence
- **Apples-to-Apples Benchmarking**: Both solvers use identical SOCP/IPM formulation (CVXPY CLARABEL vs Quantum IPM)
- **Run Logger** (`utils/result_logger.py`): Every run saved as a JSON in `results/` for cross-run analysis

---

## Analysing Results

```python
from utils.result_logger import summarise_runs
import pandas as pd

df = pd.DataFrame(summarise_runs())
print(df[["run_id", "n_clk", "cls_oos_pct", "quantum_oos_pct", "oos_gap_pct"]])
```

---

## References

- Kerenidis, Prakash & Szilágyi (2021) — *"Quantum Interior Point Methods for SDPs"*
- Harrow, Hassidim & Lloyd (2009) — *"Quantum Algorithm for Linear Systems of Equations"*
- Boyd & Vandenberghe — *"Convex Optimization"*
