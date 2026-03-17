"""
One-time converter: .ipynb -> .py modules for W&B sweep execution.

For each vR.P.x notebook:
  1. Read notebook JSON
  2. Extract code cells (skip markdown)
  3. Comment out shell commands and IPython magics
  4. Wrap visualization cells in try/except for headless environments
  5. Add path_config import
  6. Write to sweep/modules/vrpX.py

Usage:
    python convert_notebooks.py
"""

import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.dirname(SCRIPT_DIR)  # New Research Approach/
MODULE_DIR = os.path.join(SCRIPT_DIR, "modules")

# Notebook filename -> module filename
NOTEBOOK_MAP = {
    "vR.P.0 Image Detection and Localisation.ipynb": "vrp0.py",
    "vR.P.1 Image Detection and Localisation.ipynb": "vrp1.py",
    "vR.P.1.5 Image Detection and Localisation.ipynb": "vrp1_5.py",
    "vR.P.2 Image Detection and Localisation.ipynb": "vrp2.py",
    "vR.P.3.1 Image Detection and Localisation.ipynb": "vrp3.py",
    "vR.P.4 Image Detection and Localisation.ipynb": "vrp4.py",
    "vR.P.5 Image Detection and Localisation.ipynb": "vrp5.py",
    "vR.P.6 Image Detection and Localisation.ipynb": "vrp6.py",
    "vR.P.7 Image Detection and Localisation.ipynb": "vrp7.py",
    "vR.P.8 Image Detection and Localisation.ipynb": "vrp8.py",
    "vR.P.9 Image Detection and Localisation.ipynb": "vrp9.py",
    "vR.P.10 Image Detection and Localisation.ipynb": "vrp10.py",
    "vR.P.11 Image Detection and Localisation.ipynb": "vrp11.py",
    "vR.P.12 Image Detection and Localisation.ipynb": "vrp12.py",
    "vR.P.13 Image Detection and Localisation.ipynb": "vrp13.py",
    "vR.P.14 Image Detection and Localisation.ipynb": "vrp14.py",
    "vR.P.15 Image Detection and Localisation.ipynb": "vrp15.py",
    "vR.P.16 Image Detection and Localisation.ipynb": "vrp16.py",
    "vR.P.17 Image Detection and Localisation.ipynb": "vrp17.py",
    "vR.P.18 Image Detection and Localisation.ipynb": "vrp18.py",
    "vR.P.19 Image Detection and Localisation.ipynb": "vrp19.py",
    "vR.P.20 Image Detection and Localisation.ipynb": "vrp20.py",
    "vR.P.21 Image Detection and Localisation.ipynb": "vrp21.py",
    "vR.P.22 Image Detection and Localisation.ipynb": "vrp22.py",
    "vR.P.23 Image Detection and Localisation.ipynb": "vrp23.py",
    "vR.P.24 Image Detection and Localisation.ipynb": "vrp24.py",
    "vR.P.25 Image Detection and Localisation.ipynb": "vrp25.py",
    "vR.P.26 Image Detection and Localisation.ipynb": "vrp26.py",
    "vR.P.27 Image Detection and Localisation.ipynb": "vrp27.py",
    "vR.P.28 Image Detection and Localisation.ipynb": "vrp28.py",
}

# Cells that are visualization-only (safe to wrap in try/except)
VIZ_CELLS = {10, 19, 20, 22, 23}


def get_indent(line):
    """Return the number of leading spaces in a line."""
    return len(line) - len(line.lstrip())


def is_block_opener(line):
    """Check if a line ends with ':' (opens a new indented block)."""
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        return False
    return stripped.endswith(':') or stripped.endswith(':\\')


def fix_indentation(lines):
    """Fix indentation jumps that are valid in Jupyter cells but not in scripts.

    In Jupyter, each cell runs independently via exec(), so indentation
    within a cell is relative to the cell itself. When cells are concatenated
    into a single .py file, indentation must be consistent across cells.

    This function detects lines where indentation increases without a
    block-opening statement on the previous line, and dedents them to match.
    """
    if not lines:
        return lines

    fixed = []
    prev_indent = 0
    prev_is_opener = False

    for line in lines:
        stripped = line.strip()

        # Preserve blank lines and comments as-is
        if not stripped:
            fixed.append(line)
            continue

        curr_indent = get_indent(line)

        # If indentation increased but previous line wasn't a block opener,
        # dedent to match the previous line's indentation
        if curr_indent > prev_indent and not prev_is_opener and prev_indent >= 0:
            excess = curr_indent - prev_indent
            # Dedent this line and all subsequent lines at this or deeper level
            line = ' ' * prev_indent + stripped
            curr_indent = prev_indent

        fixed.append(line)
        prev_indent = curr_indent
        prev_is_opener = is_block_opener(line)

    return fixed


def clean_source_lines(lines):
    """Clean code cell source lines for script execution."""
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Comment out shell commands
        if stripped.startswith('!'):
            cleaned.append(f"# [SHELL] {line}")
            continue
        # Comment out IPython magics
        if stripped.startswith('%') or stripped.startswith('get_ipython('):
            cleaned.append(f"# [MAGIC] {line}")
            continue
        cleaned.append(line)
    return cleaned


def convert_notebook(nb_path, module_path):
    """Convert a single notebook to a .py module."""
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    nb_basename = os.path.basename(nb_path)

    output_lines = [
        "#!/usr/bin/env python",
        f"# Auto-generated from: {nb_basename}",
        "# Do not edit directly -- regenerate with convert_notebooks.py",
        "",
        "import sys, os",
        "# Add sweep directory to path for path_config import",
        "sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))",
        "",
    ]

    for cell_idx, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell.get('source', []))
        if not source.strip():
            continue

        lines = source.split('\n')
        cleaned = clean_source_lines(lines)
        cleaned = fix_indentation(cleaned)

        # Wrap visualization cells in try/except
        if cell_idx in VIZ_CELLS:
            output_lines.append(f"\n# {'='*60}")
            output_lines.append(f"# Cell {cell_idx} (visualization — wrapped for headless)")
            output_lines.append(f"# {'='*60}")
            output_lines.append("try:")
            for cl in cleaned:
                output_lines.append(f"    {cl}")
            output_lines.append("except Exception as _viz_err:")
            output_lines.append(f"    print(f'[Cell {cell_idx}] Visualization skipped: {{_viz_err}}')")
        else:
            output_lines.append(f"\n# {'='*60}")
            output_lines.append(f"# Cell {cell_idx}")
            output_lines.append(f"# {'='*60}")
            output_lines.extend(cleaned)

    with open(module_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        f.write('\n')


def main():
    os.makedirs(MODULE_DIR, exist_ok=True)

    converted = 0
    missing = []

    for nb_name, module_name in sorted(NOTEBOOK_MAP.items()):
        nb_path = os.path.join(NOTEBOOK_DIR, nb_name)
        module_path = os.path.join(MODULE_DIR, module_name)

        if os.path.exists(nb_path):
            convert_notebook(nb_path, module_path)
            converted += 1
            print(f"  OK: {nb_name} -> {module_name}")
        else:
            missing.append(nb_name)
            print(f"  MISSING: {nb_name}")

    print(f"\nConverted {converted}/{len(NOTEBOOK_MAP)} notebooks to {MODULE_DIR}")
    if missing:
        print(f"Missing notebooks: {missing}")

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
