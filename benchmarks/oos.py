"""
benchmarks/oos.py
=================
Out-of-sample price loading and metric computation.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd


def parse_oos_period(period: str) -> tuple[str, str]:
    """
    Parse the saved OOS-period convention ``"YYYY-MM-DD to YYYY-MM-DD"``.
    """
    parts = [part.strip() for part in period.split(" to ")]
    if len(parts) != 2 or not all(parts):
        raise ValueError('OOS period must use the format "YYYY-MM-DD to YYYY-MM-DD".')
    start, end = parts
    _dt.date.fromisoformat(start)
    _dt.date.fromisoformat(end)
    return start, end


def _download_prices(assets, start, end, downloader=None, auto_adjust=True):
    if downloader is None:
        import yfinance as yf  # noqa: PLC0415

        downloader = yf.download
    return downloader(
        assets,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )


def _extract_close_prices(raw, assets):
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded OOS data does not contain Close prices.")
        prices = raw["Close"].copy()
    elif len(assets) == 1 and "Close" in raw.columns:
        prices = raw[["Close"]].rename(columns={"Close": assets[0]})
    else:
        prices = raw.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=assets[0])
    return prices


def load_oos_prices(
    assets: list,
    start: str,
    end: str,
    *,
    downloader=None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download and clean OOS close prices for the requested assets.

    Missing tickers are dropped after forward-fill, matching the notebook's OOS
    benchmark behavior.
    """
    assets = list(assets)
    raw = _download_prices(
        assets,
        start=start,
        end=end,
        downloader=downloader,
        auto_adjust=auto_adjust,
    )
    prices = _extract_close_prices(raw, assets)
    prices = prices.dropna(how="all").ffill().dropna(axis=1, how="any")

    assets_oos = [asset for asset in assets if asset in prices.columns]
    if not assets_oos:
        raise ValueError("No OOS close prices are available for the requested assets.")
    return prices[assets_oos]


def load_oos_prices_for_period(
    assets: list,
    period: str,
    *,
    downloader=None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Load OOS prices from a saved ``"YYYY-MM-DD to YYYY-MM-DD"`` period."""
    start, end = parse_oos_period(period)
    return load_oos_prices(
        assets,
        start=start,
        end=end,
        downloader=downloader,
        auto_adjust=auto_adjust,
    )


def _portfolio_sharpe(daily_returns: pd.Series, trading_days_per_year: int) -> float:
    std = daily_returns.std()
    if pd.isna(std) or std == 0:
        return float("nan")
    ann_factor = np.sqrt(trading_days_per_year)
    return float((daily_returns.mean() * trading_days_per_year) / (std * ann_factor))


def compute_oos_benchmark(
    prices: pd.DataFrame,
    assets: list,
    classical_weights: np.ndarray,
    quantum_weights: np.ndarray,
    *,
    classical_ok: bool = True,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Compute OOS total return, Sharpe ratio, and HHI for two portfolios.
    """
    assets = list(assets)
    prices = prices.copy()
    assets_oos = [asset for asset in assets if asset in prices.columns]
    if not assets_oos:
        raise ValueError("No overlap between requested assets and OOS price columns.")
    if len(prices) < 2:
        raise ValueError("At least two OOS price rows are required.")

    idx_oos = [assets.index(asset) for asset in assets_oos]
    w_classical_all = np.asarray(classical_weights, dtype=float)
    w_quantum_all = np.asarray(quantum_weights, dtype=float)
    if len(w_classical_all) != len(assets) or len(w_quantum_all) != len(assets):
        raise ValueError("Weight vectors must have the same length as assets.")

    if not classical_ok:
        w_classical_all = np.zeros(len(assets), dtype=float)

    w_classical = w_classical_all[idx_oos]
    w_quantum = w_quantum_all[idx_oos]
    test_data = prices[assets_oos]

    daily_returns = test_data.pct_change().dropna(how="any")
    cls_daily_port_ret = daily_returns @ w_classical
    qipm_daily_port_ret = daily_returns @ w_quantum

    asset_cumulative_returns = (test_data.iloc[-1] / test_data.iloc[0]) - 1
    classical_total_return = float(np.dot(w_classical, asset_cumulative_returns))
    quantum_total_return = float(np.dot(w_quantum, asset_cumulative_returns))

    return {
        "assets": assets_oos,
        "prices": test_data,
        "daily_returns": daily_returns,
        "classical_weights": w_classical,
        "quantum_weights": w_quantum,
        "classical_daily_returns": cls_daily_port_ret,
        "quantum_daily_returns": qipm_daily_port_ret,
        "asset_cumulative_returns": asset_cumulative_returns,
        "classical_total_return": classical_total_return,
        "quantum_total_return": quantum_total_return,
        "classical_sharpe": _portfolio_sharpe(cls_daily_port_ret, trading_days_per_year),
        "quantum_sharpe": _portfolio_sharpe(qipm_daily_port_ret, trading_days_per_year),
        "classical_hhi": float(np.sum(w_classical**2)),
        "quantum_hhi": float(np.sum(w_quantum**2)),
    }


def oos_summary_table(
    metrics: dict,
    total_return_label: str = "Total Return (OOS)",
) -> pd.DataFrame:
    """Return the three-row OOS summary table used by the notebook/logger."""
    return pd.DataFrame(
        {
            "Metric": [
                total_return_label,
                "Sharpe Ratio (OOS)",
                "HHI (Diversification)",
            ],
            "Classical": [
                metrics["classical_total_return"],
                metrics["classical_sharpe"],
                metrics["classical_hhi"],
            ],
            "Quantum": [
                metrics["quantum_total_return"],
                metrics["quantum_sharpe"],
                metrics["quantum_hhi"],
            ],
        }
    )


def print_oos_summary(metrics: dict, total_return_label: str = "Total Return (OOS)") -> None:
    """Print the OOS summary in the historical notebook/logger format."""
    print(f"\n{'Metric':<25} | {'Classical':<12} | {'Quantum':<12}")
    print("-" * 55)
    print(
        f"{total_return_label:<25} | "
        f"{metrics['classical_total_return']:12.2%} | "
        f"{metrics['quantum_total_return']:12.2%}"
    )
    print(f"{'Sharpe Ratio (OOS)':<25} | {metrics['classical_sharpe']:12.4f} | {metrics['quantum_sharpe']:12.4f}")
    print(f"{'HHI (Diversification)':<25} | {metrics['classical_hhi']:12.4f} | {metrics['quantum_hhi']:12.4f}")
