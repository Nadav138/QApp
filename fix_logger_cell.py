import json

nb_path = '/Users/nadav.ben-ami/Documents/dev/repos/Aca/QApp/quantum_ipm_research.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# New logger cell that calls utils/result_logger instead of having inline logic
new_logger_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Results Logger ────────────────────────────────────────────────────────────\n",
        "# Delegates to utils/result_logger.py — no inline logging logic here.\n",
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath('.'))\n",
        "from utils.result_logger import log_run\n",
        "\n",
        "log_path = log_run(\n",
        "    config               = CONFIG,\n",
        "    assets               = assets,\n",
        "    mu_vec               = mu_vec,\n",
        "    cov                  = cov,\n",
        "    w_cls                = w_star,\n",
        "    cls_ok               = cls_ok if 'cls_ok' in dir() else False,\n",
        "    cls_status           = cls_status if 'cls_status' in dir() else 'N/A',\n",
        "    w_qipm               = w_ipm_final,\n",
        "    ipm_ret              = ipm_ret,\n",
        "    ipm_var              = ipm_var,\n",
        "    classical_oos_pct    = classical_total_return * 100,\n",
        "    quantum_oos_pct      = quantum_total_return * 100,\n",
        "    oos_period           = '2025-01-01 to 2025-12-31',\n",
        ")\n",
        "\n",
        "print(f'Results saved to: {log_path}')\n",
        "print(f'\\n--- Run Summary ---')\n",
        "print(f'  Training:     {CONFIG[\"start_date\"]} to {CONFIG[\"end_date\"]}')\n",
        "print(f'  Tickers:      {len(CONFIG[\"tickers\"])} assets')\n",
        "print(f'  Classical:    {\"OK\" if cls_ok else \"FAILED\"} | OOS Return: {classical_total_return*100:.2f}%')\n",
        "print(f'  Quantum:      OK | OOS Return: {quantum_total_return*100:.2f}%')\n",
        "print(f'  Clock Qubits: {CONFIG[\"quantum_hhl_n_clk\"]}')\n",
        "print(f'  Adaptive:     {CONFIG.get(\"quantum_ipm_use_adaptive_step\", False)}')\n"
    ]
}

# Find and replace the logger cell (the one with inline json.dump logic)
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'json.dump(run_log' in src:
            nb['cells'][i] = new_logger_cell
            print(f'Replaced inline logger at cell index {i}')
            break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=2)
print('Done.')
