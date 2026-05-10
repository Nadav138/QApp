# Quantum Interior-Point Portfolio Optimization

This repository implements a robust, research-aligned **Quantum Interior-Point Method (IPM)** for portfolio optimization, modeled strictly after the theoretical framework proposed by *Kerenidis et al.*

By formulating the Modern Portfolio Theory (MPT) constraints into a **Second Order Cone Program (SOCP)** using Lorentz Cones, this solver guarantees exact constraint matching against classical baseline solvers, with support for strict equality return targets, non-negative long-only weights, budget caps, and dynamic maximum asset allocations.

## Features

- **Custom Quantum Linear System Solver (QLSS)**: A native Qiskit implementation of the HHL-style Phase Estimation algorithm built from scratch using raw multi-controlled `RYGates`, `QFTGate`, and `scipy.linalg.expm` matrix evolution.
- **Adaptive Step Size (Ratio Test)**: Includes a mathematically robust boundary checker that calculates the absolute maximum step size ($\alpha_{\text{max}}$) allowed for both $L^0$ and $L^m$ Lorentz cones at every iteration, ensuring rapid and safe IPM convergence.
- **Apples-to-Apples Benchmarking**: Automatically validates the Quantum algorithm's performance against a classical SciPy SLSQP solver, ensuring both algorithms solve identically constrained topological spaces.
- **Out-of-Sample Testing**: Dynamically fetches unseen market data to backtest and compare the classical and quantum portfolios under real-world market conditions.

## Environment Setup

This project uses modern **Qiskit 2.x**. Ensure you are using a clean Python 3.10+ virtual environment.

```bash
# Clone the repository
git clone <your-repo-url>
cd QApp

# Create a virtual environment
python3 -m venv quantum_env
source quantum_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Open the core notebook to configure and run the algorithm:
```bash
jupyter notebook portfolio_optimization_qiskit_step_by_step.ipynb
```

At the top of the notebook, you can modify the `CONFIG` block to test different constraints, market dates, and algorithmic parameters:

```python
CONFIG = {
    # Data controls
    "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"],
    
    # Financial constraints
    "total_allocation": 1.0,          
    "max_weight": 0.30,               
    "target_return": 0.35,            

    # Advanced Quantum Solver Controls
    "quantum_ipm_use_adaptive_step": True, # Adaptive step-size (fraction-to-boundary)
    "quantum_ipm_alpha": 0.8,              # Static step-size fallback
    "quantum_hhl_n_clk": 8,                # Clock qubits for QPE precision
}
```

## Architecture

1. **Setup and Data Ingestion**: Pulls raw historical ticker data from `yfinance`.
2. **Classical Baseline**: Solves the portfolio using SLSQP to establish a Global Minimum Variance benchmark on the exact topological curve.
3. **Quantum-Ready Reformulations**: Casts the problem constraints into $L^m \times L^0$ Lorentz Cones.
4. **Quantum Linear Algebra Subroutines**: Defines the 13-qubit Phase Estimation simulation logic to invert the KKT matrix.
5. **Full Quantum IPM Loop**: Iteratively calculates the Newton step via the Quantum Simulator, applies the Adaptive Ratio Test, and safely descends toward the optimal weights.
6. **Out-of-Sample Performance Testing**: Computes the true returns on completely unseen future data.
