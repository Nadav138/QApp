"""
utils/result_logger.py

Standalone result logger for the Quantum IPM Portfolio Optimization research notebook.
Call `log_run(...)` at the end of a notebook run to persist results to results/.
"""

import os
import json
import datetime
import numpy as np


def log_run(
    config: dict,
    assets: list,
    mu_vec: np.ndarray,
    cov: np.ndarray,
    # Classical
    w_cls: np.ndarray,
    cls_ok: bool,
    cls_status: str,
    # Quantum
    w_qipm: np.ndarray,
    ipm_ret: float,
    ipm_var: float,
    # Out-of-sample
    classical_oos_pct: float,
    quantum_oos_pct: float,
    oos_period: str = "2025-01-01 to 2025-12-31",
    results_dir: str = None,
) -> str:
    """
    Persist one experiment run to a timestamped JSON file inside results/.

    Returns the path of the written file.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"

    run_log = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": {
            "tickers": config.get("tickers", assets),
            "train_start": config.get("start_date", ""),
            "train_end": config.get("end_date", ""),
            "target_return": config.get("target_return"),
            "max_weight": config.get("max_weight"),
            "total_allocation": config.get("total_allocation", 1.0),
            "quantum_ipm_use_adaptive_step": config.get("quantum_ipm_use_adaptive_step", False),
            "quantum_ipm_alpha": config.get("quantum_ipm_alpha", 0.8),
            "quantum_hhl_n_clk": config.get("quantum_hhl_n_clk", 6),
        },
        "classical": {
            "solver": "CVXPY CLARABEL (SOCP IPM)",
            "status": cls_status,
            "success": bool(cls_ok),
            "weights": {a: float(w) for a, w in zip(assets, w_cls)} if w_cls is not None else {},
            "expected_return": float(mu_vec @ w_cls) if w_cls is not None and cls_ok else None,
            "annual_variance": float(w_cls @ cov @ w_cls) if w_cls is not None and cls_ok else None,
        },
        "quantum": {
            "solver": "Quantum IPM (SOCP / HHL Phase Estimation)",
            "n_clock_qubits": config.get("quantum_hhl_n_clk", 6),
            "adaptive_step": config.get("quantum_ipm_use_adaptive_step", False),
            "weights": {a: float(w) for a, w in zip(assets, w_qipm)},
            "expected_return": float(ipm_ret),
            "annual_variance": float(ipm_var),
        },
        "out_of_sample": {
            "test_period": oos_period,
            "classical_return_pct": float(classical_oos_pct),
            "quantum_return_pct": float(quantum_oos_pct),
        },
    }

    path = os.path.join(results_dir, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)

    return path


def summarise_runs(results_dir: str = None) -> list[dict]:
    """
    Load all JSON run logs and return a list of summary dicts,
    sorted by timestamp (newest first).

    Useful for analysis:
        from utils.result_logger import summarise_runs
        import pandas as pd
        df = pd.DataFrame(summarise_runs())
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

    summaries = []
    for fname in sorted(os.listdir(results_dir), reverse=True):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        summaries.append({
            "run_id":            data.get("run_id"),
            "timestamp":         data.get("timestamp"),
            "tickers":           ",".join(data["config"].get("tickers", [])),
            "train_start":       data["config"].get("train_start"),
            "train_end":         data["config"].get("train_end"),
            "target_return":     data["config"].get("target_return"),
            "max_weight":        data["config"].get("max_weight"),
            "n_clk":             data["config"].get("quantum_hhl_n_clk"),
            "adaptive":          data["config"].get("quantum_ipm_use_adaptive_step"),
            "cls_ok":            data["classical"].get("success"),
            "cls_return":        data["classical"].get("expected_return"),
            "cls_variance":      data["classical"].get("annual_variance"),
            "qipm_return":       data["quantum"].get("expected_return"),
            "qipm_variance":     data["quantum"].get("annual_variance"),
            "cls_oos_pct":       data["out_of_sample"].get("classical_return_pct"),
            "quantum_oos_pct":   data["out_of_sample"].get("quantum_return_pct"),
            "oos_gap_pct":       (data["out_of_sample"].get("quantum_return_pct", 0)
                                  - data["out_of_sample"].get("classical_return_pct", 0)),
        })
    return summaries
