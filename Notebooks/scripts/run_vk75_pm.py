"""
Execute vK.7.5 notebook using papermill with training epochs reduced for quick validation.
Creates a temporary copy with reduced epochs, runs it, then reports results.
"""

import json
import copy
import os
import sys

os.chdir(r"c:\D Drive\Projects\BigVision Assignment\Notebooks\source")

INPUT_PATH = "vK.7.5 Image Detection and Localisation.ipynb"
TEMP_PATH = "vK.7.5_temp_run.ipynb"
OUTPUT_PATH = "../runs/vK.7.5_run_output.ipynb"

# Load notebook
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Patch training cells for quick validation
patched = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])

    # Prior experiment: NUM_EPOCHS = 30 -> 2
    if "NUM_EPOCHS = 30" in src and "best_acc = 0" in src:
        cell["source"] = [s.replace("NUM_EPOCHS = 30", "NUM_EPOCHS = 2") for s in cell["source"]]
        patched.append((i, "prior experiment: 30 -> 2 epochs"))

    # Effective training: NUM_EPOCHS = 50 -> 2
    if "NUM_EPOCHS = 50" in src and "best_val_acc = 0" in src:
        cell["source"] = [s.replace("NUM_EPOCHS = 50", "NUM_EPOCHS = 2") for s in cell["source"]]
        patched.append((i, "effective training: 50 -> 2 epochs"))

    # Normalize cell IDs
    if "id" not in cell:
        cell["id"] = f"cell-{i:04d}"

for ci, desc in patched:
    print(f"Patched cell {ci}: {desc}")

# Also add cell IDs to markdown cells
for i, cell in enumerate(nb["cells"]):
    if "id" not in cell:
        cell["id"] = f"cell-{i:04d}"

# Save temp copy
with open(TEMP_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Saved temp notebook: {TEMP_PATH}")
print("Running with papermill...")
sys.stdout.flush()

# Run with papermill
import papermill as pm

try:
    pm.execute_notebook(
        TEMP_PATH,
        OUTPUT_PATH,
        kernel_name="python3",
        cwd=r"c:\D Drive\Projects\BigVision Assignment\Notebooks",
        progress_bar=True,
        execution_timeout=-1,  # No per-cell timeout
    )
    print("\nExecution completed successfully!")
except Exception as e:
    print(f"\nExecution failed: {e}")

# Report results
with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
    result = json.load(f)

errors = []
for i, cell in enumerate(result["cells"]):
    if cell["cell_type"] != "code":
        continue
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            src = "".join(cell["source"])[:80].replace("\n", " | ")
            errors.append({
                "cell": i,
                "name": out.get("ename", "?"),
                "value": out.get("evalue", "?")[:200],
                "src": src,
            })

print(f"\n{'='*60}")
print(f"Cells with errors: {len(errors)}")
for e in errors:
    print(f"  Cell {e['cell']}: {e['name']}: {e['value']}")
    print(f"    Source: {e['src']}")

# Keep temp file for inspection
# os.remove(TEMP_PATH)
print(f"Temp notebook kept at: {TEMP_PATH}")
