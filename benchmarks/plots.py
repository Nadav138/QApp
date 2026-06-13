"""
benchmarks/plots.py
===================
Plotting helpers for training diagnostics and OOS benchmark comparison.
"""

from __future__ import annotations


def plot_training_diagnostics(prices, returns_daily, training_label: str | None = None, show: bool = True):
    """
    Plot normalized training prices and the daily-return correlation heatmap.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    norm_prices = prices / prices.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    norm_prices.plot(ax=axes[0], lw=1.3)
    suffix = f" ({training_label})" if training_label else ""
    axes[0].set_title(f"Normalized Prices{suffix}")
    axes[0].set_ylabel("Normalized level")

    sns.heatmap(returns_daily.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title("Daily Returns Correlation")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_oos_comparison(metrics: dict, show: bool = True):
    """
    Plot OOS weight allocations and cumulative equity curves.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    assets = metrics["assets"]
    w_classical = metrics["classical_weights"]
    w_quantum = metrics["quantum_weights"]
    cls_daily_port_ret = metrics["classical_daily_returns"]
    qipm_daily_port_ret = metrics["quantum_daily_returns"]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x_indices = np.arange(len(assets))
    width = 0.35

    ax[0].bar(x_indices - width / 2, w_classical, width, label="Classical", color="teal")
    ax[0].bar(x_indices + width / 2, w_quantum, width, label="Quantum", color="coral")
    ax[0].set_xticks(x_indices)
    ax[0].set_xticklabels(assets)
    ax[0].set_title("Weight Allocation")
    ax[0].legend()

    ax[1].plot((1 + cls_daily_port_ret).cumprod(), label="Classical", color="teal")
    ax[1].plot((1 + qipm_daily_port_ret).cumprod(), label="Quantum", color="coral")
    ax[1].set_title("OOS Cumulative Equity Curve")
    ax[1].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_qipm_convergence(
    gaps,
    sigma_theory: float,
    *,
    tolerance: float = 1e-2,
    show: bool = True,
):
    """
    Plot QIPM duality-gap convergence with Matplotlib mathtext labels.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    gaps = np.asarray(gaps, dtype=float)
    if len(gaps) == 0:
        raise ValueError("At least one duality gap is required.")

    k = np.arange(len(gaps))
    theory = gaps[0] * (sigma_theory**k)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(k, gaps, marker="o", color="darkorange", linewidth=2, label="Quantum IPM")
    ax.semilogy(k, theory, linestyle="--", color="steelblue", label=r"Theory $\nu_0 \cdot \sigma^k$")

    tolerance_label = f"Tolerance {tolerance:g}"
    ax.axhline(tolerance, color="gray", linestyle="--", alpha=0.7, label=tolerance_label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Duality Gap (log scale)")
    ax.set_title(rf"Quantum IPM — Convergence (short-step rate $\sigma \approx {sigma_theory:.3f}$)")
    ax.legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
