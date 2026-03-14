"""
Execute vK.7.5 notebook cell-by-cell with controlled timeouts.
Training cells get a reduced epoch count for quick validation.
Reports all errors encountered.
"""

import json
import nbformat
from nbclient import NotebookClient
import re
import time
import os

os.chdir(r"c:\D Drive\Projects\BigVision Assignment\Notebooks\source")

INPUT_PATH = "vK.7.5 Image Detection and Localisation.ipynb"
OUTPUT_PATH = "../runs/vK.7.5_run_output.ipynb"

# Load the notebook
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Normalize cell IDs (fixes MissingIDFieldWarning)
for i, cell in enumerate(nb.cells):
    if "id" not in cell:
        cell["id"] = f"cell-{i:04d}"

# Patch training cells to use reduced epochs for quick validation
PATCHED_CELLS = {}
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source

    # Prior experiment: NUM_EPOCHS = 30 -> 2
    if "NUM_EPOCHS = 30" in src and "best_acc = 0" in src:
        PATCHED_CELLS[i] = src
        cell.source = src.replace("NUM_EPOCHS = 30", "NUM_EPOCHS = 2")
        print(f"Cell {i}: Patched NUM_EPOCHS 30 -> 2 (prior experiment)")

    # Effective training: NUM_EPOCHS = 50 -> 2
    if "NUM_EPOCHS = 50" in src and "best_val_acc = 0" in src:
        PATCHED_CELLS[i] = src
        cell.source = src.replace("NUM_EPOCHS = 50", "NUM_EPOCHS = 2")
        print(f"Cell {i}: Patched NUM_EPOCHS 50 -> 2 (effective training)")

print(f"\nTotal cells: {len(nb.cells)}")
print(f"Code cells: {sum(1 for c in nb.cells if c.cell_type == 'code')}")
print(f"Patched cells: {len(PATCHED_CELLS)}")
print()

# Create the client
client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": r"c:\D Drive\Projects\BigVision Assignment\Notebooks"}},
)

# Execute cell by cell
errors = []
print("Starting execution...")
print("=" * 60)

try:
    client.setup_kernel()

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        src_preview = cell.source[:80].replace("\n", " | ")
        print(f"Cell {i:3d}: {src_preview}...")

        t0 = time.time()
        try:
            client.execute_cell(cell, i)
            elapsed = time.time() - t0
            print(f"         OK ({elapsed:.1f}s)")

            # Print any stdout output (abbreviated)
            for out in cell.outputs:
                if out.output_type == "stream" and out.name == "stdout":
                    text = out.text
                    lines = text.strip().split("\n")
                    if len(lines) <= 5:
                        for l in lines:
                            print(f"         > {l}")
                    else:
                        for l in lines[:3]:
                            print(f"         > {l}")
                        print(f"         > ... ({len(lines)} lines total)")
                        for l in lines[-2:]:
                            print(f"         > {l}")

        except Exception as exc:
            elapsed = time.time() - t0
            err_msg = str(exc)[:500]
            print(f"         FAILED ({elapsed:.1f}s): {err_msg}")
            errors.append({"cell": i, "error": str(exc)})

            # Print cell outputs for debugging
            for out in cell.outputs:
                if out.output_type == "error":
                    tb = "\n".join(out.get("traceback", []))[-500:]
                    print(f"         Traceback: {tb}")

finally:
    try:
        client.cleanup_kernel()
    except:
        pass

# Restore patched cells
for i, original_src in PATCHED_CELLS.items():
    nb.cells[i].source = original_src

# Save output
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print()
print("=" * 60)
print(f"Execution complete. Errors: {len(errors)}")
for e in errors:
    print(f"  Cell {e['cell']}: {e['error'][:200]}")
print(f"Output saved to: {OUTPUT_PATH}")
