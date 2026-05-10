import json, copy

nb_path = '/Users/nadav.ben-ami/Documents/dev/repos/Aca/QApp/quantum_portfolio_tutorial.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Build new markdown cell
new_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "⚠️ **Self‑contained tutorial**\n",
        "* This notebook installs all required packages in the first code cell (just run it).\n",
        "* No external data files or repository cloning are needed – everything is generated on‑the‑fly.\n"
    ]
}

# Insert after the first title markdown cell (index 0)
nb['cells'].insert(1, new_md)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=2)
print('Inserted self‑contained note cell')
