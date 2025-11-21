"""
Script to replicate FD001 methods to FD002 automatically.
This script copies and adapts notebooks from FD001 to FD002.
"""

import json
from pathlib import Path
import shutil

def adapt_notebook_for_fd002(fd001_notebook_path, fd002_notebook_path, title_suffix=" - FD002"):
    """Adapt a notebook from FD001 to FD002"""

    # Read FD001 notebook
    with open(fd001_notebook_path, 'r') as f:
        nb = json.load(f)

    # Adapt cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # Replace data paths
            cell['source'] = [line.replace('../data/train.csv', '../data/train.csv')
                             .replace('../data/test.csv', '../data/test.csv')
                             .replace('../data/rul.csv', '../data/rul.csv')
                             for line in cell['source']]
        elif cell['cell_type'] == 'markdown':
            # Update title if it's the first cell
            if len(cell['source']) > 0 and cell['source'][0].startswith('# '):
                if 'FD001' in cell['source'][0]:
                    cell['source'][0] = cell['source'][0].replace('FD001', 'FD002')
                elif title_suffix and not 'FD002' in cell['source'][0]:
                    cell['source'][0] = cell['source'][0].rstrip() + title_suffix + '\n'

    # Write FD002 notebook
    fd002_notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fd002_notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"✓ Created {fd002_notebook_path.name}")

def main():
    """Main function to replicate all methods"""

    base_dir = Path(__file__).parent
    fd001_dir = base_dir.parent / 'FD001'
    fd002_dir = base_dir

    # Methods to replicate
    methods = {
        'clasic_methods': [],  # Z-Score and PCA already done manually
        'unsupervised_learning': [
            'isolation_forest.ipynb',
            'One_Class_SVM.ipynb'
        ]
    }

    print("Replicating methods from FD001 to FD002...")
    print("=" * 60)

    for method_type, notebooks in methods.items():
        print(f"\n{method_type}:")
        for notebook in notebooks:
            fd001_nb = fd001_dir / method_type / notebook
            fd002_nb = fd002_dir / method_type / notebook

            if fd001_nb.exists():
                try:
                    adapt_notebook_for_fd002(fd001_nb, fd002_nb)
                except Exception as e:
                    print(f"✗ Error with {notebook}: {e}")
            else:
                print(f"✗ {notebook} not found in FD001")

    print("\n" + "=" * 60)
    print("Done!")

if __name__ == '__main__':
    main()
