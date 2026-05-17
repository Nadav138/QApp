"""
fill_solutions.py
-----------------
Fills the three student task cells in quantum_portfolio_tutorial.ipynb with
their complete solutions so the notebook can be run end-to-end.

Usage:
    python scripts/fill_solutions.py
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "quantum_portfolio_tutorial.ipynb"

SOLUTIONS = {
    # Task 1 — identified by the '# TODO (1)' marker for n_clk_demo
    "n_clk_demo = ...": (
        "# Equality-only KKT:  K = [[2Σ, Aᵀ], [A, 0]],  rhs = [0,…,0, 1, target_return]\n"
        "H_eq   = 2 * cov_mat\n"
        "A_eq   = np.vstack([np.ones((1, n)), mu_vec.reshape(1, n)])      # budget, return\n"
        "K_eq   = np.block([[H_eq, A_eq.T], [A_eq, np.zeros((2, 2))]])\n"
        'rhs_eq = np.concatenate([np.zeros(n), [1.0, CONFIG["target_return"]]])\n\n'
        "# TODO (1): read the number of clock qubits from CONFIG\n"
        'n_clk_demo = CONFIG["quantum_hhl_n_clk"]\n\n'
        "n_sys_demo = int(np.ceil(np.log2(K_eq.shape[0])))\n"
        'print(f"Equality-only KKT: {K_eq.shape[0]}×{K_eq.shape[1]}")\n'
        'print(f"Circuit registers: sys={n_sys_demo} qubits | clk={n_clk_demo} qubits | anc=1 qubit")\n'
        'print(f"Total qubits:      {n_sys_demo + n_clk_demo + 1}\\n")\n\n'
        "# Run HHL once (also captures `last_qc` for the visualisation below)\n"
        "dz_hhl    = quantum_newton_solver(K_eq, rhs_eq,\n"
        "                                   n_clk=n_clk_demo,\n"
        '                                   pad_eig=CONFIG["quantum_hhl_pad_eig"])\n'
        "w_hhl_raw = dz_hhl[:n]\n\n"
        'print(f"Circuit depth: {last_qc.depth()} | gates: {last_qc.size()}")\n\n'
        "# TODO (2): draw the most recent HHL circuit (style='iqp', fold=40)\n"
        "last_qc.draw('mpl', style='iqp', fold=40)\n"
    ),
    # Task 2 — identified by 'ret_qipm = ...'
    "ret_qipm = ...": (
        "# TODO: compute the QIPM portfolio's return, variance, and volatility\n"
        "ret_qipm = float(w_qipm @ mu_vec)\n"
        "var_qipm = float(w_qipm @ cov_mat @ w_qipm)\n"
        "std_qipm = float(np.sqrt(var_qipm))\n\n"
        'print(f"\\n⏱  Quantum IPM finished in {elapsed_qipm:.1f} s")\n'
        "print(f\"   Expected return  : {ret_qipm:.2%}   (target ≥ {CONFIG['target_return']:.0%})\")\n"
        'print(f"   Annual variance  : {var_qipm:.4f}")\n'
        'print(f"   Annual volatility: {std_qipm:.2%}")\n'
    ),
    # Task 3 — identified by 'test_start = ...'
    "test_start = ...": (
        'print("Downloading OOS data…")\n\n'
        "# TODO: choose the out-of-sample test period (full calendar year after training)\n"
        'test_start = "2025-01-01"\n'
        'test_end   = "2025-12-31"\n\n'
        "try:\n"
        "    test_raw = yf.download(assets, start=test_start, end=test_end,\n"
        "                            auto_adjust=True, progress=False)\n"
        "    if isinstance(test_raw.columns, pd.MultiIndex):\n"
        '        test_prices = test_raw["Close"][[a for a in assets if a in test_raw["Close"].columns]]\n'
        "    else:\n"
        "        test_prices = test_raw\n"
        "    test_returns = test_prices.pct_change().dropna()\n"
        '    oos_source   = f"yfinance {test_start} → {test_end} ({len(test_prices)} trading days)"\n'
        '    print(f"✅  {oos_source}")\n'
        "except Exception as e:\n"
        '    print(f"⚠️  yfinance failed ({e}) — using synthetic OOS returns.")\n'
        "    rng_oos      = np.random.default_rng(2025)\n"
        "    oos_daily    = rng_oos.multivariate_normal(mu_vec / 252, cov_mat / 252, 252)\n"
        "    test_returns = pd.DataFrame(oos_daily, columns=assets)\n"
        '    oos_source   = "synthetic (seed=2025)"\n'
    ),
}


def fill(nb_path: Path = NOTEBOOK) -> None:
    with open(nb_path) as f:
        nb = json.load(f)

    patched = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for marker, solution in SOLUTIONS.items():
            if marker in src:
                cell["source"] = [solution]
                patched += 1
                break

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"✅  Filled {patched}/{len(SOLUTIONS)} task cells in {nb_path.name}")


if __name__ == "__main__":
    fill()
