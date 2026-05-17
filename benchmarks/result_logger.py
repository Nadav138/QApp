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
    Persist one experiment run to a timestamped JSON file inside research/results/.

    Returns the path of the written file.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inductive naming: count, top 3 tickers, n_clk, adaptive/static
    num_assets = len(assets)
    top_tickers = "-".join(sorted(assets)[:3])
    plus_suffix = "-plus" if num_assets > 3 else ""
    n_clk = config.get("quantum_hhl_n_clk", 6)
    adaptive = "adaptive" if config.get("quantum_ipm_use_adaptive_step", False) else "static"
    
    run_id = f"run_{num_assets}ast_{top_tickers}{plus_suffix}_{n_clk}clk_{adaptive}_{timestamp}"

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
    Load a saved run-log by its *run_id* (e.g. ``"run_20260511_162804"``).

    This is a convenience wrapper around :func:`load_result` that builds the
    file path from the ``run_id`` string printed by the logging cell, so you
    do not need to know the full path.

    Parameters
    ----------
    run_id : str
        The run identifier printed by the notebook after ``log_run``.
        Accepts both bare IDs (``"run_20260511_162804"``) and full file
        names (``"run_20260511_162804.json"``).
    results_dir : str, optional
        Override the default ``research/results/`` directory.

    Example
    -------
        from benchmarks.result_logger import load_result_by_id, run_benchmark_from_result
        run_benchmark_from_result(load_result_by_id("run_20260511_162804"))
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
    # 0.  Lazy imports so the module stays importable without these deps  #
    # ------------------------------------------------------------------ #
    import yfinance as yf            # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # ------------------------------------------------------------------ #
    # 1.  Unpack the result dict                                           #
    # ------------------------------------------------------------------ #
    assets = result["config"]["tickers"]

    cls_weights_dict  = result["classical"]["weights"]
    qipm_weights_dict = result["quantum"]["weights"]

    # Build ordered numpy arrays that match *assets* order
    w_star      = np.array([cls_weights_dict[t]  for t in assets])
    w_ipm_final = np.array([qipm_weights_dict[t] for t in assets])

    # Parse OOS period  "YYYY-MM-DD to YYYY-MM-DD"
    test_period = result["out_of_sample"]["test_period"]
    oos_start, oos_end = [s.strip() for s in test_period.split(" to ")]

    # ------------------------------------------------------------------ #
    # 2.  Download OOS price data                                          #
    # ------------------------------------------------------------------ #
    print(f"Downloading unseen OOS data ({test_period}) for Out-of-Sample testing...")
    raw = yf.download(assets, start=oos_start, end=oos_end, progress=False)

    # yfinance returns a MultiIndex (Price-type × Ticker) for multiple tickers
    if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "levels"):
        test_data = raw["Close"][assets]
    else:
        test_data = raw[assets] if set(assets).issubset(raw.columns) else raw

    # ------------------------------------------------------------------ #
    # 3.  Compute metrics                                                  #
    # ------------------------------------------------------------------ #
    # Daily portfolio returns
    daily_returns       = test_data.pct_change().dropna()
    cls_daily_port_ret  = daily_returns @ w_star
    qipm_daily_port_ret = daily_returns @ w_ipm_final

    # (a) Cumulative total return
    asset_cumulative_returns = (test_data.iloc[-1] / test_data.iloc[0]) - 1
    classical_total_return   = float(np.dot(w_star,      asset_cumulative_returns))
    quantum_total_return     = float(np.dot(w_ipm_final, asset_cumulative_returns))

    # (b) Annualised Sharpe ratio  (risk-free = 0)
    ann_factor  = np.sqrt(252)
    cls_sharpe  = float((cls_daily_port_ret.mean()  * 252) / (cls_daily_port_ret.std()  * ann_factor))
    qipm_sharpe = float((qipm_daily_port_ret.mean() * 252) / (qipm_daily_port_ret.std() * ann_factor))

    # (c) HHI  (lower = better diversification)
    cls_hhi  = float(np.sum(w_star      ** 2))
    qipm_hhi = float(np.sum(w_ipm_final ** 2))

    # ------------------------------------------------------------------ #
    # 4.  Print summary table                                              #
    # ------------------------------------------------------------------ #
    print(f"\n{'Metric':<25} | {'Classical':<12} | {'Quantum':<12}")
    print("-" * 55)
    print(f"{'Total Return (OOS)':<25} | {classical_total_return:12.2%} | {quantum_total_return:12.2%}")
    print(f"{'Sharpe Ratio (OOS)':<25} | {cls_sharpe:12.4f} | {qipm_sharpe:12.4f}")
    print(f"{'HHI (Diversification)':<25} | {cls_hhi:12.4f} | {qipm_hhi:12.4f}")

    # ------------------------------------------------------------------ #
    # 5.  Visualisation                                                    #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    x_indices = np.arange(len(assets))
    width = 0.35

    # Left panel – weight allocation
    ax[0].bar(x_indices - width / 2, w_star,      width, label="Classical", color="teal")
    ax[0].bar(x_indices + width / 2, w_ipm_final, width, label="Quantum",   color="coral")
    ax[0].set_xticks(x_indices)
    ax[0].set_xticklabels(assets)
    ax[0].set_title("Weight Allocation")
    ax[0].legend()

    # Right panel – cumulative equity curve
    ax[1].plot((1 + cls_daily_port_ret).cumprod(),  label="Classical", color="teal")
    ax[1].plot((1 + qipm_daily_port_ret).cumprod(), label="Quantum",   color="coral")
    ax[1].set_title("OOS Cumulative Equity Curve")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


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
