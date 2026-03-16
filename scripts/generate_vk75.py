"""
Generate vK.7.5 from vK.7 by adapting the notebook for local execution.

Key changes:
- Add an environment-detection cell that sets up paths for Colab OR local
- Replace hardcoded /kaggle/input and /kaggle/working refs
- Make !pip install conditional
- Skip Google Drive mount when not in Colab
"""

import json
import copy
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
INPUT_PATH = NOTEBOOKS_DIR / "source" / "vK.7 Image Detection and Localisation [Subsections by Opus].ipynb"
OUTPUT_PATH = NOTEBOOKS_DIR / "source" / "vK.7.5 Image Detection and Localisation.ipynb"


def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Saved: {path}")


def make_md_cell(text):
    lines = text.split("\n")
    source = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source.append(line + "\n")
        else:
            source.append(line)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def make_code_cell(text):
    lines = text.split("\n")
    source = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source.append(line + "\n")
        else:
            source.append(line)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def get_source(cell):
    return "".join(cell["source"])


def set_source(cell, text):
    lines = text.split("\n")
    source = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source.append(line + "\n")
        else:
            source.append(line)
    cell["source"] = source


def main():
    nb = load_notebook(INPUT_PATH)
    old_cells = nb["cells"]
    new_cells = []

    # =====================================================================
    # Pass 1: Build the new cell list with targeted modifications
    # =====================================================================

    # Track which cell index we're processing
    i = 0
    inserted_env_cell = False

    while i < len(old_cells):
        cell = copy.deepcopy(old_cells[i])
        src = get_source(cell)

        # -----------------------------------------------------------------
        # INSERTION: After the TOC (cell 0) and title (cell 1), insert the
        # environment-detection cell that configures all paths
        # -----------------------------------------------------------------
        if i == 1 and not inserted_env_cell:
            # First, add the original cell
            # Update title to vK.7.5
            src_updated = src.replace("vK.7", "vK.7.5")
            set_source(cell, src_updated)
            new_cells.append(cell)
            inserted_env_cell = True

            # Insert environment config markdown
            new_cells.append(make_md_cell(
                "## Runtime Environment Configuration\n"
                "\n"
                "This cell detects whether the notebook is running on Google Colab or locally,\n"
                "and configures dataset and working directory paths accordingly.\n"
                "\n"
                "- **Colab**: uses `/kaggle/input` and `/kaggle/working` (Kaggle-style emulation)\n"
                "- **Local**: resolves paths relative to the notebook directory"
            ))

            # Insert the environment detection code cell
            new_cells.append(make_code_cell(
                'import sys\n'
                'import platform\n'
                'from pathlib import Path\n'
                '\n'
                'IN_COLAB = "google.colab" in sys.modules\n'
                '\n'
                'if IN_COLAB:\n'
                '    KAGGLE_INPUT_DIR = Path("/kaggle/input")\n'
                '    KAGGLE_WORKING_DIR = Path("/kaggle/working")\n'
                'else:\n'
                '    # Local execution: resolve paths relative to the notebook location.\n'
                '    _NB_DIR = Path.cwd()\n'
                '    # Walk upward to find the project root (contains a kaggle/ directory or is the repo root).\n'
                '    _PROJECT_ROOT = _NB_DIR\n'
                '    for _p in [_NB_DIR] + list(_NB_DIR.parents):\n'
                '        if (_p / "kaggle" / "input").is_dir():\n'
                '            _PROJECT_ROOT = _p\n'
                '            break\n'
                '        if (_p / ".git").exists():\n'
                '            _PROJECT_ROOT = _p\n'
                '            break\n'
                '    KAGGLE_INPUT_DIR = _PROJECT_ROOT / "kaggle" / "input"\n'
                '    KAGGLE_WORKING_DIR = _PROJECT_ROOT / "kaggle" / "working"\n'
                '\n'
                '# On Windows, multiprocessing DataLoader workers can deadlock outside __main__.\n'
                '# Use 0 workers locally to avoid hangs; keep the original value on Colab (Linux).\n'
                'SAFE_NUM_WORKERS = 2 if IN_COLAB else 0\n'
                '\n'
                'KAGGLE_INPUT_DIR.mkdir(parents=True, exist_ok=True)\n'
                'KAGGLE_WORKING_DIR.mkdir(parents=True, exist_ok=True)\n'
                '\n'
                'print("IN_COLAB:", IN_COLAB)\n'
                'print("KAGGLE_INPUT_DIR:", KAGGLE_INPUT_DIR)\n'
                'print("KAGGLE_WORKING_DIR:", KAGGLE_WORKING_DIR)\n'
                'print("SAFE_NUM_WORKERS:", SAFE_NUM_WORKERS)'
            ))

            i += 1
            continue

        # -----------------------------------------------------------------
        # CELL 5 (original idx): Runtime config - rewrite to use env vars
        # -----------------------------------------------------------------
        if 'COLAB_KAGGLE_INPUT = Path("/kaggle/input")' in src:
            set_source(cell,
                'import subprocess\n'
                'import sys\n'
                'from pathlib import Path\n'
                '\n'
                'IN_COLAB = "google.colab" in sys.modules\n'
                'KAGGLE_DATASET_SLUG = "sagnikkayalcse52/casia-spicing-detection-localization"\n'
                'COLAB_KAGGLE_INPUT = KAGGLE_INPUT_DIR\n'
                'COLAB_KAGGLE_WORKING = KAGGLE_WORKING_DIR\n'
                '\n'
                'if IN_COLAB:\n'
                '    subprocess.check_call(\n'
                '        [\n'
                '            sys.executable,\n'
                '            "-m",\n'
                '            "pip",\n'
                '            "install",\n'
                '            "-q",\n'
                '            "albumentations==1.3.1",\n'
                '            "opencv-python-headless==4.10.0.84",\n'
                '            "kaggle",\n'
                '        ]\n'
                '    )\n'
                '\n'
                'COLAB_KAGGLE_INPUT.mkdir(parents=True, exist_ok=True)\n'
                'COLAB_KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)\n'
                '\n'
                'print("IN_COLAB:", IN_COLAB)\n'
                'print("COLAB_KAGGLE_INPUT:", COLAB_KAGGLE_INPUT)\n'
                'print("COLAB_KAGGLE_WORKING:", COLAB_KAGGLE_WORKING)\n'
                'print("KAGGLE_DATASET_SLUG:", KAGGLE_DATASET_SLUG)'
            )
            new_cells.append(cell)
            i += 1
            continue

        # -----------------------------------------------------------------
        # Dataset discovery helpers + drive mount cell:
        # The cell with "from google.colab import drive" - make drive mount
        # conditional and handle local dataset path
        # -----------------------------------------------------------------
        if "from google.colab import drive" in src and "dataset_dir = None" in src:
            set_source(cell,
                'dataset_dir = None\n'
                '\n'
                'if IN_COLAB:\n'
                '    try:\n'
                '        from google.colab import drive\n'
                '\n'
                '        # Prefer a mounted Google Drive copy so the notebook can reuse existing data without re-downloading.\n'
                '        drive.mount("/content/drive", force_remount=False)\n'
                '        dataset_dir = find_dataset_in_drive(DRIVE_SEARCH_ROOTS)\n'
                '        if dataset_dir is not None:\n'
                '            dataset_dir = normalize_dataset_dir(dataset_dir, TARGET_DATASET_DIR)\n'
                '            print(f"Using dataset from Google Drive: {dataset_dir}")\n'
                '    except Exception as exc:\n'
                '        print(f"Google Drive mount/search skipped: {exc}")\n'
                '\n'
                '    if dataset_dir is None:\n'
                '        # Fall back to the Kaggle API only when Drive-based discovery does not find a valid dataset root.\n'
                '        dataset_dir = ensure_dataset_from_kaggle(TARGET_DATASET_DIR)\n'
                '        if dataset_dir is not None:\n'
                '            print(f"Using dataset from Kaggle download: {dataset_dir}")\n'
                'else:\n'
                '    # Local execution: the dataset should already be at the target location (symlink or copy).\n'
                '    if has_image_and_mask_dirs(TARGET_DATASET_DIR):\n'
                '        dataset_dir = TARGET_DATASET_DIR\n'
                '        print(f"Using local dataset: {dataset_dir}")\n'
                '\n'
                'if dataset_dir is None or not has_image_and_mask_dirs(dataset_dir):\n'
                '    raise FileNotFoundError(\n'
                '        f"Could not prepare {TARGET_DATASET_DIR}. "\n'
                '        "Provide a Google Drive copy containing IMAGE and MASK folders or configure Kaggle API credentials for "\n'
                '        "sagnikkayalcse52/casia-spicing-detection-localization."\n'
                '    )\n'
                '\n'
                'print("Normalized dataset root:", dataset_dir)\n'
                'print("IMAGE dir exists:", (dataset_dir / "IMAGE").exists())\n'
                'print("MASK dir exists:", (dataset_dir / "MASK").exists())'
            )
            new_cells.append(cell)
            i += 1
            continue

        # -----------------------------------------------------------------
        # Cells with hardcoded /kaggle/input or /kaggle/working paths
        # Replace with the variables from the env cell
        # -----------------------------------------------------------------
        if cell["cell_type"] == "code":
            modified = False

            # Replace INPUT_ROOT = Path("/kaggle/input")
            if 'INPUT_ROOT = Path("/kaggle/input")' in src:
                src = src.replace(
                    'INPUT_ROOT = Path("/kaggle/input")',
                    'INPUT_ROOT = KAGGLE_INPUT_DIR'
                )
                modified = True

            # Replace DATASET_DIR with correct path construction
            if 'DATASET_DIR = INPUT_ROOT / "/kaggle/input/casia-spicing-detection-localization"' in src:
                src = src.replace(
                    'DATASET_DIR = INPUT_ROOT / "/kaggle/input/casia-spicing-detection-localization"',
                    'DATASET_DIR = KAGGLE_INPUT_DIR / "casia-spicing-detection-localization"'
                )
                modified = True

            # Replace output_csv paths
            if 'output_csv = "/kaggle/working/image_mask_metadata.csv"' in src:
                src = src.replace(
                    'output_csv = "/kaggle/working/image_mask_metadata.csv"',
                    'output_csv = str(KAGGLE_WORKING_DIR / "image_mask_metadata.csv")'
                )
                modified = True

            # Replace csv_path in split cell
            if 'csv_path = Path("/kaggle/working/image_mask_metadata.csv")' in src:
                src = src.replace(
                    'csv_path = Path("/kaggle/working/image_mask_metadata.csv")',
                    'csv_path = KAGGLE_WORKING_DIR / "image_mask_metadata.csv"'
                )
                modified = True

            # Replace output_dir in split cell
            if 'output_dir = Path("/kaggle/working")' in src:
                src = src.replace(
                    'output_dir = Path("/kaggle/working")',
                    'output_dir = KAGGLE_WORKING_DIR'
                )
                modified = True

            # Replace TRAIN/VAL/TEST CSV paths (prior experiment block)
            if 'TRAIN_CSV = "/kaggle/working/test_metadata.csv"' in src:
                src = src.replace(
                    'TRAIN_CSV = "/kaggle/working/test_metadata.csv"',
                    'TRAIN_CSV = str(KAGGLE_WORKING_DIR / "test_metadata.csv")'
                )
                src = src.replace(
                    'VAL_CSV   = "/kaggle/working/val_metadata.csv"',
                    'VAL_CSV   = str(KAGGLE_WORKING_DIR / "val_metadata.csv")'
                )
                src = src.replace(
                    'TEST_CSV  = "/kaggle/working/val_metadata.csv"',
                    'TEST_CSV  = str(KAGGLE_WORKING_DIR / "val_metadata.csv")'
                )
                modified = True

            # Replace TRAIN/VAL/TEST CSV paths (effective training block)
            if 'TRAIN_CSV = "/kaggle/working/train_metadata.csv"' in src:
                src = src.replace(
                    'TRAIN_CSV = "/kaggle/working/train_metadata.csv"',
                    'TRAIN_CSV = str(KAGGLE_WORKING_DIR / "train_metadata.csv")'
                )
                src = src.replace(
                    'VAL_CSV   = "/kaggle/working/val_metadata.csv"',
                    'VAL_CSV   = str(KAGGLE_WORKING_DIR / "val_metadata.csv")'
                )
                src = src.replace(
                    'TEST_CSV  = "/kaggle/working/test_metadata.csv"',
                    'TEST_CSV  = str(KAGGLE_WORKING_DIR / "test_metadata.csv")'
                )
                modified = True

            # Replace best_model_path
            if 'best_model_path = "/kaggle/working/best_model.pth"' in src:
                src = src.replace(
                    'best_model_path = "/kaggle/working/best_model.pth"',
                    'best_model_path = str(KAGGLE_WORKING_DIR / "best_model.pth")'
                )
                modified = True

            # Replace !pip install with conditional installs
            if "!pip install" in src:
                src = src.replace(
                    "!pip install -q albumentations==1.3.1 opencv-python-headless==4.10.0.84",
                    "import subprocess, sys\n"
                    "if IN_COLAB:\n"
                    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
                    "                           'albumentations==1.3.1', 'opencv-python-headless==4.10.0.84'])"
                )
                modified = True

            # Replace print("Available datasets under /kaggle/input:")
            if 'print("Available datasets under /kaggle/input:")' in src:
                src = src.replace(
                    'print("Available datasets under /kaggle/input:")',
                    'print(f"Available datasets under {KAGGLE_INPUT_DIR}:")'
                )
                modified = True

            # Replace hardcoded NUM_WORKERS = 2 with the safe environment value
            if 'NUM_WORKERS = 2' in src:
                src = src.replace(
                    'NUM_WORKERS = 2',
                    'NUM_WORKERS = SAFE_NUM_WORKERS'
                )
                modified = True

            # Fix albumentations 2.x API: JpegCompression -> ImageCompression
            if 'A.JpegCompression(quality_lower=50, quality_upper=90, p=0.25)' in src:
                src = src.replace(
                    'A.JpegCompression(quality_lower=50, quality_upper=90, p=0.25)',
                    'A.ImageCompression(quality_range=(50, 90), p=0.25)'
                )
                modified = True

            if modified:
                set_source(cell, src)

        # Also update TOC cell title and main title
        if cell["cell_type"] == "markdown":
            if "vK.7" in src:
                src = src.replace("vK.7", "vK.7.5")
                set_source(cell, src)

        new_cells.append(cell)
        i += 1

    # Build output
    out_nb = copy.deepcopy(nb)
    out_nb["cells"] = new_cells
    save_notebook(out_nb, OUTPUT_PATH)

    # -----------------------------------------------------------------
    # Verification: no leftover hardcoded /kaggle/ paths (except in
    # error messages and comments)
    # -----------------------------------------------------------------
    print("\n=== Verification: remaining /kaggle/ references ===")
    for ci, cell in enumerate(new_cells):
        if cell["cell_type"] != "code":
            continue
        src = get_source(cell)
        for j, line in enumerate(src.split("\n")):
            stripped = line.strip()
            if "/kaggle/" in stripped:
                # Skip comments and string error messages
                if stripped.startswith("#"):
                    continue
                if "Could not prepare" in stripped:
                    continue
                if "Keep the original" in stripped:
                    continue
                print(f"  Cell {ci}, L{j}: {stripped[:120]}")


if __name__ == "__main__":
    main()
