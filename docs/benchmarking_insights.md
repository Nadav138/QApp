# Benchmarking Insights: Quantum vs. Classical Portfolio Optimization

Based on the initial run results in `research/results/run_20260510_175027.json`, here are the findings and recommendations for improving our benchmarking framework.

## 1. Initial Findings
*   **Precision Parity**: The Quantum IPM implementation (using HHL-simulated Newton steps) achieves results very close to the `CLARABEL` SOCP solver. 
    *   Classical Variance: 0.0351
    *   Quantum Variance: 0.0357
*   **Out-of-Sample (OOS) Variance**: In the test run, the Quantum portfolio slightly outperformed the Classical one (25.45% vs 24.26%). This might be a result of the "natural regularization" introduced by the approximate nature of the HHL solver, preventing over-fitting to the training covariance.

## 2. Recommended Benchmarking Enhancements

### A. Risk-Adjusted Metrics
Total return is a "noisy" metric. We should add:
*   **Sharpe Ratio**: Annualized Return / Annualized Volatility for the OOS period.
*   **Maximum Drawdown**: The largest peak-to-trough decline during the OOS period.

### B. Portfolio Concentration
*   **HHI (Herfindahl-Hirschman Index)**: Sum of squared weights. This will tell us if the Quantum solver tends to produce more or less diversified portfolios compared to the classical interior point method.

### C. Convergence Performance
*   **Iteration Count**: Track how many IPM iterations are required for the duality gap to drop below the threshold for both methods.
*   **Stability**: Monitor the duality gap curve. Quantum steps might be "bumpier" than classical ones.

### D. Sensitivity Analysis
*   **Qubit Precision (`n_clk`)**: Benchmark the OOS performance vs. the number of clock qubits. Is there a "sweet spot" where approximation error actually helps generalization?
*   **Asset Scalability**: Compare runtime and accuracy as we scale from 4 to 20+ assets.

## 3. Implementation Steps
1.  Update the OOS cell in the research notebook to calculate daily returns and volatility.
2.  Incorporate Sharpe Ratio and HHI into the `benchmarks/result_logger.py` data structure.
3.  Run a sweep across different `n_clk` values (e.g., 5, 6, 7, 8) and record the impact on OOS metrics.
