import json

with open('workshop/quantum_portfolio_tutorial.ipynb', 'r') as f:
    nb = json.load(f)

markdown_cell = {
   "cell_type": "markdown",
   "id": "plot_hhl_alloc_md",
   "metadata": {},
   "source": [
    "#### Let's see our allocation (Plain HHL):\n",
    "Notice the negative weights (short positions) which are not allowed by the long-only constraint."
   ]
}

code_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "plot_hhl_alloc_code",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(7, 3.5))\n",
    "ax.bar(assets, w_hhl_raw, color=\"coral\", edgecolor=\"white\")\n",
    "ax.axhline(CONFIG[\"max_weight\"], color=\"red\", linestyle=\"--\",\n",
    "           label=f\"Max weight ({CONFIG['max_weight']:.0%})\")\n",
    "ax.axhline(0, color=\"black\", linewidth=0.8)\n",
    "ax.set_ylabel(\"Allocation weight\")\n",
    "ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n",
    "ax.set_title(\"Plain HHL: Portfolio Weights (Equality Constraints Only)\")\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
}

# Insert after cell 32
nb['cells'].insert(33, markdown_cell)
nb['cells'].insert(34, code_cell)

with open('workshop/quantum_portfolio_tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

