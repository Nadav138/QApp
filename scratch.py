import json

with open('workshop/quantum_portfolio_tutorial.ipynb', 'r') as f:
    nb = json.load(f)

for i in range(25, 36):
    cell = nb['cells'][i]
    print(f"--- Cell {i} ({cell['cell_type']}) ---")
    print("".join(cell['source']))
