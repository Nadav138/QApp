"""
benchmarks/result_logger.py

Standalone result logger for the Quantum IPM Portfolio Optimization research notebook.
Call `log_run(...)` at the end of a notebook run to persist results to research/results/.
Call `run_benchmark_from_result(result)` to re-run the OOS benchmark from a saved JSON.
"""

import os
import json
import datetime
import numpy as np

from benchmarks.oos import compute_oos_benchmark, load_oos_prices_for_period, print_oos_summary
from benchmarks.plots import plot_oos_comparison


def _config_clock_qubits(config: dict, default: int | None = 6):
    """Return the QPE clock-qubit count, accepting old notebook keys."""
    return config.get(
        "qpe_clock_qubits",
        config.get(
            "quantum_qipm_n_clk",
            config.get("quantum_hhl_clock_qubits", config.get("quantum_hhl_n_clk", default)),
        ),
    )


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
    qipm_status: str | None = None,
    qipm_diagnostics: dict | None = None,
    qipm_result: dict | None = None,
) -> str:
    """
    Persist one experiment run to a timestamped JSON file inside research/results/.

    Returns the path of the written file.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clear run naming: asset count, representative tickers, QPE clock-qubit count, step rule.
    num_assets = len(assets)
    top_tickers = "-".join(sorted(assets)[:3])
    plus_suffix = "-plus" if num_assets > 3 else ""
    clock_qubits = _config_clock_qubits(config)
    step_rule = config.get("quantum_ipm_step_rule")
    adaptive = step_rule or ("adaptive" if config.get("quantum_ipm_use_adaptive_step", False) else "static")
    
    run_id = f"run_{num_assets}ast_{top_tickers}{plus_suffix}_{clock_qubits}clockq_{adaptive}_{timestamp}"

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
            "quantum_ipm_step_rule": config.get("quantum_ipm_step_rule"),
            "quantum_qipm_n_clk": clock_qubits,
            "qpe_clock_qubits": clock_qubits,
            "quantum_hhl_n_clk": config.get("quantum_hhl_n_clk", clock_qubits),
            "quantum_hhl_clock_qubits": clock_qubits,
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
            "n_clock_qubits": clock_qubits,
            "qpe_clock_qubits": clock_qubits,
            "adaptive_step": config.get("quantum_ipm_use_adaptive_step", False),
            "step_rule": config.get("quantum_ipm_step_rule"),
            "status": qipm_status,
            "diagnostics": qipm_diagnostics or {},
            "raw_weights": True,
            "sum_weights": float(np.sum(w_qipm)),
            "iterations": (qipm_result or {}).get("iterations"),
            "sigma_theory": (qipm_result or {}).get("sigma_theory"),
            "gap_ratios": (qipm_result or {}).get("gap_ratios"),
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
        from benchmarks.result_logger import summarise_runs
        import pandas as pd
        df = pd.DataFrame(summarise_runs())
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "results")

    summaries = []
    for fname in sorted(os.listdir(results_dir), reverse=True):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        clock_qubits = _config_clock_qubits(data.get("config", {}), default=None)
        summaries.append({
            "run_id":            data.get("run_id"),
            "timestamp":         data.get("timestamp"),
            "tickers":           ",".join(data["config"].get("tickers", [])),
            "train_start":       data["config"].get("train_start"),
            "train_end":         data["config"].get("train_end"),
            "target_return":     data["config"].get("target_return"),
            "max_weight":        data["config"].get("max_weight"),
            "clock_qubits":      clock_qubits,
            "n_clk":             clock_qubits,  # Backward-compatible alias.
            "adaptive":          data["config"].get("quantum_ipm_use_adaptive_step"),
            "step_rule":         data["config"].get("quantum_ipm_step_rule"),
            "cls_ok":            data["classical"].get("success"),
            "cls_return":        data["classical"].get("expected_return"),
            "cls_variance":      data["classical"].get("annual_variance"),
            "qipm_status":       data["quantum"].get("status"),
            "qipm_return":       data["quantum"].get("expected_return"),
            "qipm_variance":     data["quantum"].get("annual_variance"),
            "qipm_sum_w":        data["quantum"].get("sum_weights"),
            "qipm_primal_resid": data["quantum"].get("diagnostics", {}).get("primal_resid_inf") or data["quantum"].get("diagnostics", {}).get("primal_resid"),
            "qipm_dual_resid":   data["quantum"].get("diagnostics", {}).get("dual_resid_inf") or data["quantum"].get("diagnostics", {}).get("dual_resid"),
            "cls_oos_pct":       data["out_of_sample"].get("classical_return_pct"),
            "quantum_oos_pct":   data["out_of_sample"].get("quantum_return_pct"),
            "oos_gap_pct":       (data["out_of_sample"].get("quantum_return_pct", 0)
                                  - data["out_of_sample"].get("classical_return_pct", 0)),
        })
    return summaries


def load_result(path: str) -> dict:
    """
    Load a single JSON run-log from *path* and return it as a dict.

    Example
    -------
        from benchmarks.result_logger import load_result, run_benchmark_from_result
        result = load_result("research/results/run_20260511_162804.json")
        run_benchmark_from_result(result)
    """
    with open(path) as f:
        return json.load(f)


def load_result_by_id(run_id: str, results_dir: str = None) -> dict:
    """
    Load a saved run-log by its *run_id*.

    This is a convenience wrapper around :func:`load_result` that builds the
    file path from the ``run_id`` string printed by the logging cell, so you
    do not need to know the full path.

    Parameters
    ----------
    run_id : str
        The run identifier printed by the notebook after ``log_run``.
        Format: ``"run_<N>ast_<TICKERS>_<N>clockq_<rule>_<YYYYMMDD_HHMMSS>"``.
        Accepts both bare IDs and full file names (with ``.json`` suffix).
        Example: ``"run_4ast_AAPL-AMZN-GOOGL-plus_6clockq_paper_20260613_110701"``.
    results_dir : str, optional
        Override the default ``research/results/`` directory.

    Example
    -------
        from benchmarks.result_logger import load_result_by_id, run_benchmark_from_result
        run_benchmark_from_result(
            load_result_by_id("run_4ast_AAPL-AMZN-GOOGL-plus_6clockq_paper_20260613_110701")
        )
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "results")

    # Accept either "run_YYYYMMDD_HHMMSS" or "run_YYYYMMDD_HHMMSS.json"
    if not run_id.endswith(".json"):
        run_id = run_id + ".json"

    return load_result(os.path.join(results_dir, run_id))


def run_benchmark_from_result(result: dict) -> None:
    """
    Re-run the out-of-sample benchmark from a saved JSON result dict.

    The function mirrors the hardcoded benchmark cell in
    ``notebooks/quantum_ipm_research.ipynb`` but is driven entirely by the
    data already stored in *result* (weights, tickers, OOS period), so no
    notebook state is required.

    Parameters
    ----------
    result : dict
        A result dict as produced by ``log_run`` or ``load_result``.

    What it does
    ------------
    1. Extracts tickers, classical weights and quantum weights from *result*.
    2. Parses the OOS test period (e.g. ``"2025-01-01 to 2025-12-31"``).
    3. Downloads daily close prices for that period via ``yfinance``.
    4. Computes per-solver:
       - Total cumulative return over the OOS window.
       - Annualised Sharpe ratio (risk-free rate = 0).
       - Herfindahl-Hirschman Index (HHI) of weight concentration.
    5. Prints a formatted summary table.
    6. Shows a two-panel ``matplotlib`` figure:
       - Left  : weight-allocation bar chart (classical vs quantum).
       - Right : OOS cumulative equity curves.

    Example
    -------
        from benchmarks.result_logger import load_result, run_benchmark_from_result
        run_benchmark_from_result(load_result("research/results/run_20260511_162804.json"))
    """
    # ------------------------------------------------------------------ #
    # 1.  Unpack the result dict                                           #
    # ------------------------------------------------------------------ #
    assets = result["config"]["tickers"]

    cls_weights_dict  = result["classical"]["weights"]
    qipm_weights_dict = result["quantum"]["weights"]

    # Build ordered numpy arrays that match *assets* order
    w_star      = np.array([cls_weights_dict.get(t, 0.0) for t in assets])
    w_ipm_final = np.array([qipm_weights_dict[t] for t in assets])

    test_period = result["out_of_sample"]["test_period"]

    # ------------------------------------------------------------------ #
    # 2.  Download OOS price data                                          #
    # ------------------------------------------------------------------ #
    print(f"Downloading unseen OOS data ({test_period}) for Out-of-Sample testing...")
    test_data = load_oos_prices_for_period(assets, test_period)

    # ------------------------------------------------------------------ #
    # 3.  Compute metrics                                                  #
    # ------------------------------------------------------------------ #
    metrics = compute_oos_benchmark(
        test_data,
        assets,
        w_star,
        w_ipm_final,
        classical_ok=result["classical"].get("success", True),
    )

    # ------------------------------------------------------------------ #
    # 4.  Print summary table                                              #
    # ------------------------------------------------------------------ #
    print_oos_summary(metrics, total_return_label="Total Return (OOS)")

    # ------------------------------------------------------------------ #
    # 5.  Visualisation                                                    #
    # ------------------------------------------------------------------ #
    plot_oos_comparison(metrics)


def validate_result(log_path: str) -> bool:
    """
    Checks if the result logged at log_path is valid (contains full data).
    If valid, prints "Valid full run".
    If not valid, prints "Not valid" and renames the file to include "_not_valid" in the name.

    Returns True if valid, False otherwise.
    """
    if not os.path.exists(log_path):
        print(f"Error: Path {log_path} does not exist.")
        return False

    try:
        with open(log_path, 'r') as f:
            data = json.load(f)

        # Define what "full data" means
        # 1. Classical solver must have succeeded
        classical_ok = data.get("classical", {}).get("success", False)
        # 2. Both solvers must have produced expected returns (not None)
        classical_ret = data.get("classical", {}).get("expected_return")
        quantum_ret = data.get("quantum", {}).get("expected_return")
        # 3. Quantum weights must exist and be non-empty
        quantum_weights = data.get("quantum", {}).get("weights", {})

        is_valid = (
            classical_ok and
            classical_ret is not None and
            quantum_ret is not None and
            len(quantum_weights) > 0
        )

        if is_valid:
            print("Valid full run")
            return True
        else:
            print("Not valid")
            # Rename file
            base, ext = os.path.splitext(log_path)
            if "_not_valid" not in base:
                new_path = f"{base}_not_valid{ext}"
                os.rename(log_path, new_path)
                print(f"Renamed {os.path.basename(log_path)} to {os.path.basename(new_path)}")
            return False

    except Exception as e:
        print(f"Not valid (Error reading file: {e})")
        # Rename file on error too
        base, ext = os.path.splitext(log_path)
        if "_not_valid" not in base:
            new_path = f"{base}_not_valid{ext}"
            os.rename(log_path, new_path)
            print(f"Renamed {os.path.basename(log_path)} to {os.path.basename(new_path)} due to error.")
        return False
