import json
import sys

nb_path = '/Users/nadav.ben-ami/Documents/dev/repos/Aca/QApp/quantum_portfolio_tutorial.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Find the cell containing the problematic import line
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        src = cell.get('source', [])
        for i, line in enumerate(src):
            if 'from qiskit.extensions import UnitaryGate' in line:
                # Replace with try/except fallback
                new_lines = [
                    '    # Compatibility import for UnitaryGate across Qiskit versions\n',
                    '    try:\n',
                    '        from qiskit.extensions import UnitaryGate\n',
                    '    except ImportError:\n',
                    '        from qiskit.circuit import UnitaryGate\n'
                ]
                src[i:i+1] = new_lines
                cell['source'] = src
                print('Replaced import line')
                break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=2)
print('Notebook updated')
