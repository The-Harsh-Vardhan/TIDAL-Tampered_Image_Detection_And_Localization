"""
Generate vK.7.1 from vK.7 by making dataset resolution Kaggle-first.

This is an isolated post-processor:
- input notebook remains unchanged
- cell order and overall structure are preserved
- only targeted markdown/code cells are rewritten
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = NOTEBOOKS_DIR / "source" / "vK.7 Image Detection and Localisation [Subsections by Opus].ipynb"
OUTPUT_PATH = NOTEBOOKS_DIR / "source" / "vK.7.1 Image Detection and Localisation.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(nb: dict, path: Path) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {path.name}")


def get_source(cell: dict) -> str:
    return "".join(cell["source"])


def to_source_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def set_source(cell: dict, text: str) -> None:
    cell["source"] = to_source_lines(text)


def replace_text(cell: dict, old: str, new: str) -> None:
    src = get_source(cell)
    if old not in src:
        raise ValueError(f"Expected text not found in cell: {old}")
    set_source(cell, src.replace(old, new))


def clear_outputs_if_needed(cell: dict) -> None:
    if cell["cell_type"] == "code" and cell.get("outputs"):
        cell["outputs"] = []
        cell["execution_count"] = None


def expect_contains(cell: dict, needle: str, label: str) -> None:
    if needle not in get_source(cell):
        raise ValueError(f"Notebook shape mismatch for {label}: missing {needle!r}")


def main() -> None:
    src_nb = load_notebook(INPUT_PATH)
    out_nb = copy.deepcopy(src_nb)
    cells = out_nb["cells"]

    if len(cells) != 91:
        raise ValueError(f"Unexpected cell count: {len(cells)}")

    # Safety checks so we only transform the expected vK.7 notebook.
    expect_contains(cells[0], "submission-notebook-vk7", "TOC/title anchor")
    expect_contains(cells[1], "Submission Notebook (vK.7)", "title")
    expect_contains(cells[5], 'COLAB_KAGGLE_INPUT = Path("/kaggle/input")', "runtime config code")
    expect_contains(cells[14], "from google.colab import drive", "dataset resolution code")
    expect_contains(cells[20], 'INPUT_ROOT = Path("/kaggle/input")', "dataset root discovery code")
    expect_contains(cells[49], '"runtime": "Google Colab"', "wandb config")
    expect_contains(cells[90], "one Google Colab notebook", "conclusion")

    modified_code_indices: set[int] = set()

    # Cell 1 (0-based 0): TOC
    replace_text(
        cells[0],
        "[Introduction](#tampered-image-detection-and-localization---submission-notebook-vk7)",
        "[Introduction](#tampered-image-detection-and-localization---submission-notebook-vk71)",
    )
    replace_text(
        cells[0],
        "Runtime Configuration and Kaggle-Style Directories",
        "Runtime Configuration and Kaggle Dataset Paths",
    )
    replace_text(
        cells[0],
        "Dataset Discovery Helpers",
        "Attached Kaggle Dataset and Fallback Helpers",
    )
    replace_text(
        cells[0],
        "Dataset Resolution from Google Drive or Kaggle",
        "Kaggle-First Dataset Resolution",
    )

    # Cell 2: title / intro
    title_text = """# Tampered Image Detection and Localization - Submission Notebook (vK.7.1)

This Kaggle-first notebook presents a complete assignment submission for tampered image detection and tampered region localization.
The workflow keeps the original implementation intact while improving readability, traceability, and presentation quality for final review.

**Notebook deliverables demonstrated here**
- image-level tamper detection through the classifier head
- pixel-level tampered region localization through the segmentation branch
- reproducible Kaggle-first execution using the preserved source notebook workflow
- qualitative visual evidence showing predicted masks and overlays

**Assignment alignment:** This notebook addresses the full assignment pipeline from dataset preparation to tamper detection, localization, evaluation, and qualitative reporting."""
    set_source(cells[1], title_text)

    # Cell 3: objectives table wording
    replace_text(cells[2], "Run in one Colab notebook", "Run in one Kaggle-first notebook")
    replace_text(
        cells[2],
        "Environment setup, dataset preparation, training, evaluation, and visualization remain in one notebook.",
        "Environment setup, Kaggle-first dataset preparation, training, evaluation, and visualization remain in one notebook.",
    )

    # Cell 4: section intro
    section1_text = """## 1. Environment Setup

The next cells configure the notebook runtime, use Kaggle-native dataset paths as the default source, and prepare fallback options only when the attached dataset is unavailable so the original code can run without redesigning the implementation.

**Section breakdown**
- runtime package installation and Kaggle dataset/working directories
- attached Kaggle dataset discovery with Drive/API fallback

**Assignment alignment:** This section supports the assignment requirement that the end-to-end tampering workflow runs in a single notebook environment."""
    set_source(cells[3], section1_text)

    # Cell 5: subsection markdown
    cell5_text = """### 1.1 Runtime Configuration and Kaggle Dataset Paths

Installs required packages for the notebook runtime and defines the Kaggle dataset and working paths
so later cells can default to the attached dataset under `/kaggle/input` without changing the downstream pipeline."""
    set_source(cells[4], cell5_text)

    # Cell 6: runtime config code
    cell6_code = """import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
KAGGLE_DATASET_SLUG = "sagnikkayalcse52/casia-spicing-detection-localization"
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")
ATTACHED_DATASET_DIR = KAGGLE_INPUT_DIR / "casia-spicing-detection-localization"
COLAB_KAGGLE_INPUT = KAGGLE_INPUT_DIR
COLAB_KAGGLE_WORKING = KAGGLE_WORKING_DIR

if IN_COLAB:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "albumentations==1.3.1",
            "opencv-python-headless==4.10.0.84",
            "kaggle",
        ]
    )

KAGGLE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_WORKING_DIR.mkdir(parents=True, exist_ok=True)

print("IN_COLAB:", IN_COLAB)
print("KAGGLE_INPUT_DIR:", KAGGLE_INPUT_DIR)
print("KAGGLE_WORKING_DIR:", KAGGLE_WORKING_DIR)
print("ATTACHED_DATASET_DIR:", ATTACHED_DATASET_DIR)
print("KAGGLE_DATASET_SLUG:", KAGGLE_DATASET_SLUG)"""
    set_source(cells[5], cell6_code)
    modified_code_indices.add(5)

    # Cell 7: dataset helper intro
    cell7_text = """### 1.2 Attached Kaggle Dataset and Fallback Helpers

These helpers validate dataset layout, search the attached Kaggle input area first, and retain Google Drive and Kaggle API paths only as fallback options."""
    set_source(cells[6], cell7_text)

    # Cell 8: helper subsection
    cell8_text = """#### 1.2.1 Layout Validation, Path Utilities, and Kaggle Input Search

These helpers check whether a directory contains the expected `IMAGE` and `MASK` subdirectories,
deduplicate discovered paths, and look for the attached Kaggle dataset before any fallback path is considered."""
    set_source(cells[7], cell8_text)

    # Cell 9: helper code
    cell9_code = '''import os
import shutil
from pathlib import Path


TARGET_DATASET_DIR = ATTACHED_DATASET_DIR
DRIVE_SEARCH_ROOTS = [
    Path("/content/drive/MyDrive"),
    Path("/content/drive/Shareddrives"),
]


def has_image_and_mask_dirs(path: Path) -> bool:
    """
    Purpose:
        Check whether a directory already matches the expected dataset layout.

    Inputs:
        path (Path): Candidate directory to validate.

    Returns:
        bool: True when the directory contains both IMAGE and MASK subdirectories.

    Notes:
        The notebook expects the CASIA-style folder structure before metadata generation can begin.
    """
    if path is None or not path.exists() or not path.is_dir():
        return False
    try:
        child_names = {child.name.lower() for child in path.iterdir() if child.is_dir()}
    except OSError:
        return False
    return "image" in child_names and "mask" in child_names


def sorted_unique_paths(paths):
    """
    Purpose:
        Deduplicate and sort discovered filesystem paths before selecting a dataset root.

    Inputs:
        paths (Iterable[Path]): Candidate paths collected during recursive search.

    Returns:
        list[Path]: Unique paths ordered to prefer shorter, cleaner directory matches.

    Notes:
        Sorting makes the automatic selection step deterministic across repeated notebook runs.
    """
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return sorted(unique, key=lambda p: (len(p.parts), str(p).lower()))


def find_dataset_in_input(search_root: Path) -> Path | None:
    """
    Purpose:
        Search the Kaggle input directory for an attached dataset containing IMAGE and MASK folders.

    Inputs:
        search_root (Path): Top-level Kaggle input directory to search.

    Returns:
        Path | None: The best matching dataset directory, or None if nothing suitable is attached.

    Notes:
        The search prefers the exact attached dataset slug before falling back to any compatible dataset layout.
    """
    if search_root is None or not search_root.exists() or not search_root.is_dir():
        return None

    preferred = []
    fallback = []

    exact_candidate = search_root / "casia-spicing-detection-localization"
    if has_image_and_mask_dirs(exact_candidate):
        preferred.append(exact_candidate)

    for candidate in search_root.iterdir():
        if candidate.is_dir() and has_image_and_mask_dirs(candidate):
            fallback.append(candidate)

    for candidate in search_root.rglob("*"):
        if candidate.is_dir() and has_image_and_mask_dirs(candidate):
            if candidate.name == "casia-spicing-detection-localization":
                preferred.append(candidate)
            else:
                fallback.append(candidate)

    candidates = sorted_unique_paths(preferred or fallback)
    return candidates[0] if candidates else None'''
    set_source(cells[8], cell9_code)
    modified_code_indices.add(8)

    # Cell 10: drive helper markdown
    cell10_text = """#### 1.2.2 Google Drive Fallback Search

Recursively searches mounted Google Drive directories for a dataset folder
only when the attached Kaggle dataset is unavailable and a Colab fallback path is needed."""
    set_source(cells[9], cell10_text)

    # Cell 11: drive helper code
    cell11_code = '''def find_dataset_in_drive(search_roots):
    """
    Purpose:
        Search mounted Google Drive locations for a fallback dataset directory containing IMAGE and MASK folders.

    Inputs:
        search_roots (Iterable[Path]): Top-level Drive directories to search recursively.

    Returns:
        Path | None: The best matching dataset directory, or None if nothing suitable is found.

    Notes:
        This helper is used only after the attached Kaggle dataset check has failed.
    """
    preferred = []
    fallback = []
    for root in search_roots:
        if not root.exists():
            continue

        if has_image_and_mask_dirs(root):
            preferred.append(root)

        for pattern in ["casia-spicing-detection-localization", "*casia*"]:
            for candidate in root.rglob(pattern):
                if candidate.is_dir() and has_image_and_mask_dirs(candidate):
                    preferred.append(candidate)

        if preferred:
            continue

        for candidate in root.rglob("*"):
            if candidate.is_dir() and has_image_and_mask_dirs(candidate):
                fallback.append(candidate)

    candidates = sorted_unique_paths(preferred or fallback)
    return candidates[0] if candidates else None'''
    set_source(cells[10], cell11_code)
    modified_code_indices.add(10)

    # Cell 12: normalization helper markdown
    cell12_text = """#### 1.2.3 Dataset Normalization and Kaggle API Fallback

`normalize_dataset_dir` exposes a fallback dataset through the attached Kaggle-style path.
`ensure_dataset_from_kaggle` remains available only when neither the attached dataset nor Google Drive fallback is usable."""
    set_source(cells[11], cell12_text)

    # Cell 13: normalization + API fallback code
    cell13_code = '''def normalize_dataset_dir(source_dir: Path, target_dir: Path) -> Path:
    """
    Purpose:
        Normalize a fallback dataset into the Kaggle-style input path expected by later notebook cells.

    Inputs:
        source_dir (Path): The dataset directory found in Drive or a download cache.
        target_dir (Path): The normalized Kaggle-style destination directory.

    Returns:
        Path: The directory that should be used as the canonical dataset root.

    Notes:
        The function first tries to create a symlink and falls back to copying when symlinks are unavailable.
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and has_image_and_mask_dirs(target_dir):
        return target_dir

    if target_dir.is_symlink():
        target_dir.unlink()
    elif target_dir.exists():
        shutil.rmtree(target_dir)

    try:
        os.symlink(source_dir, target_dir, target_is_directory=True)
        print(f"Symlinked dataset: {target_dir} -> {source_dir}")
    except OSError:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        print(f"Copied dataset into: {target_dir}")
    return target_dir


def ensure_dataset_from_kaggle(target_dir: Path) -> Path | None:
    """
    Purpose:
        Download the dataset through the Kaggle API when the attached dataset and Google Drive fallback are both unavailable.

    Inputs:
        target_dir (Path): Destination path where the normalized dataset should be exposed.

    Returns:
        Path | None: The normalized dataset directory if the download succeeds, otherwise None.

    Notes:
        This final fallback keeps the rest of the notebook unchanged by recreating the expected folder layout.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        print(f"Kaggle API import failed: {exc}")
        return None

    download_root = KAGGLE_INPUT_DIR / "_downloads"
    download_root.mkdir(parents=True, exist_ok=True)

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET_SLUG, path=str(download_root), unzip=True, quiet=False)
    except Exception as exc:
        print(f"Kaggle download failed: {exc}")
        return None

    candidates = []
    if has_image_and_mask_dirs(download_root):
        candidates.append(download_root)
    for candidate in download_root.rglob("*"):
        if candidate.is_dir() and has_image_and_mask_dirs(candidate):
            candidates.append(candidate)

    candidates = sorted_unique_paths(candidates)
    if not candidates:
        return None

    return normalize_dataset_dir(candidates[0], target_dir)'''
    set_source(cells[12], cell13_code)
    modified_code_indices.add(12)

    # Cell 14: resolution markdown
    cell14_text = """### 1.3 Kaggle-First Dataset Resolution

Checks for the attached Kaggle dataset first, uses Google Drive only as a Colab fallback,
and keeps the Kaggle API download as the final recovery path when no usable dataset is already available."""
    set_source(cells[13], cell14_text)

    # Cell 15: resolution code
    cell15_code = '''dataset_dir = find_dataset_in_input(KAGGLE_INPUT_DIR)

if dataset_dir is not None:
    if dataset_dir == ATTACHED_DATASET_DIR:
        dataset_dir = ATTACHED_DATASET_DIR
    else:
        dataset_dir = normalize_dataset_dir(dataset_dir, ATTACHED_DATASET_DIR)
    print(f"Using attached Kaggle dataset: {dataset_dir}")

if dataset_dir is None and IN_COLAB:
    try:
        from google.colab import drive

        # Use Google Drive only as a fallback when the attached Kaggle dataset is unavailable.
        drive.mount("/content/drive", force_remount=False)
        dataset_dir = find_dataset_in_drive(DRIVE_SEARCH_ROOTS)
        if dataset_dir is not None:
            dataset_dir = normalize_dataset_dir(dataset_dir, ATTACHED_DATASET_DIR)
            print(f"Using fallback dataset from Google Drive: {dataset_dir}")
    except Exception as exc:
        print(f"Google Drive fallback search skipped: {exc}")

if dataset_dir is None:
    # Fall back to the Kaggle API only when neither the attached dataset nor Google Drive provides a valid dataset root.
    dataset_dir = ensure_dataset_from_kaggle(ATTACHED_DATASET_DIR)
    if dataset_dir is not None:
        print(f"Using fallback dataset from Kaggle download: {dataset_dir}")

if dataset_dir is None or not has_image_and_mask_dirs(dataset_dir):
    raise FileNotFoundError(
        "Expected an attached Kaggle dataset at /kaggle/input/casia-spicing-detection-localization. "
        "If it is unavailable, provide a Google Drive copy in Colab or configure Kaggle API credentials for "
        "sagnikkayalcse52/casia-spicing-detection-localization."
    )

print("Normalized dataset root:", dataset_dir)
print("IMAGE dir exists:", (dataset_dir / "IMAGE").exists())
print("MASK dir exists:", (dataset_dir / "MASK").exists())'''
    set_source(cells[14], cell15_code)
    modified_code_indices.add(14)

    # Cell 21: dataset root discovery code
    cell21_code = '''import os
from pathlib import Path
import pandas as pd

# =========================
# 1) Discover the dataset root
# =========================

# Inspect the Kaggle input directory so the notebook can confirm the expected dataset is available.
INPUT_ROOT = KAGGLE_INPUT_DIR

print(f"Available datasets under {INPUT_ROOT}:")
for p in INPUT_ROOT.iterdir():
    if p.is_dir():
        print(" -", p.name)

# Use the Kaggle-first attached dataset path prepared in the environment setup cells.
DATASET_DIR = ATTACHED_DATASET_DIR

# Search recursively for IMAGE and MASK so the metadata build still works if the dataset is nested one level deeper.
IMAGE_DIR = None
MASK_DIR = None

for p in DATASET_DIR.rglob("*"):
    if p.is_dir() and p.name.lower() == "image":
        IMAGE_DIR = p
    if p.is_dir() and p.name.lower() == "mask":
        MASK_DIR = p

print("IMAGE_DIR:", IMAGE_DIR)
print("MASK_DIR:", MASK_DIR)

assert IMAGE_DIR is not None, "Could not find the IMAGE directory. Verify the dataset folder name and structure."
assert MASK_DIR is not None, "Could not find the MASK directory. Verify the dataset folder name and structure."'''
    set_source(cells[20], cell21_code)
    modified_code_indices.add(20)

    # Cell 25: metadata CSV output
    cell25_code = '''# =========================
# 3) Save the metadata CSV
# =========================

df = pd.DataFrame(rows)
print(df.head())

output_csv = KAGGLE_WORKING_DIR / "image_mask_metadata.csv"
df.to_csv(output_csv, index=False)
print("CSV saved to:", output_csv)

# Print a small sanity check so the class balance and missing-mask cases are visible before training starts.
print("\\nCounts per class_name:")
print(df["class_name"].value_counts())

print("\\nMissing masks:")
print(df[df["mask_exists"] == 0].head())'''
    set_source(cells[24], cell25_code)
    modified_code_indices.add(24)

    # Cell 27: metadata split code
    cell27_code = '''import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# 1) Load the metadata CSV generated in the previous step.
csv_path = KAGGLE_WORKING_DIR / "image_mask_metadata.csv"
df = pd.read_csv(csv_path)

print("Total samples:", len(df))
print(df["class_name"].value_counts())

# 2) Split the dataset into train, validation, and test subsets using stratified sampling.
# Stratification preserves the authentic vs tampered ratio across all three splits.
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)

print("Train size:", len(train_df))
print("Val size:  ", len(val_df))
print("Test size: ", len(test_df))

print("\\nTrain class distribution:")
print(train_df["class_name"].value_counts())

print("\\nVal class distribution:")
print(val_df["class_name"].value_counts())

print("\\nTest class distribution:")
print(test_df["class_name"].value_counts())

# 3) Save the split metadata files so the downstream training code can load them directly.
output_dir = KAGGLE_WORKING_DIR
output_dir.mkdir(parents=True, exist_ok=True)

train_csv = output_dir / "train_metadata.csv"
val_csv   = output_dir / "val_metadata.csv"
test_csv  = output_dir / "test_metadata.csv"

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("\\nSaved:")
print(" -", train_csv)
print(" -", val_csv)
print(" -", test_csv)'''
    set_source(cells[26], cell27_code)
    modified_code_indices.add(26)

    # Cell 50: W&B runtime strings and Kaggle secret-based login
    cell50_code = '''import importlib.util
import subprocess
import sys

WANDB_ACTIVE = False
WANDB_RUN = None
WANDB_MODE = "disabled"
WANDB_CONFIG = {
    "architecture": "UNetWithClassifier",
    "image_size": 256,
    "batch_size": 8,
    "num_workers": 2,
    "learning_rate": 1e-4,
    "epochs": 50,
    "classification_loss_weight": 1.5,
    "segmentation_loss_weight": 1.0,
    "scheduler": "CosineAnnealingLR(T_max=10)",
    "dropout": 0.5,
    "dataset": "CASIA tampered vs authentic with binary masks",
    "runtime": "Kaggle-first notebook",
}

try:
    # Install W&B only when it is missing so the notebook stays self-contained in a fresh notebook runtime.
    if importlib.util.find_spec("wandb") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb"])

    import wandb

    try:
        # Read the W&B API key from Kaggle notebook secrets before attempting online logging.
        from kaggle_secrets import UserSecretsClient

        wandb_api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError('Kaggle secret "WANDB_API_KEY" is empty or unavailable.')

        wandb.login(key=wandb_api_key)
        WANDB_RUN = wandb.init(
            project="tampered-image-detection-assignment",
            name="vk71-unetwithclassifier-kaggle-first",
            tags=["assignment", "kaggle-first", "tampering", "localization"],
            config=WANDB_CONFIG,
            reinit=True,
        )
        WANDB_MODE = "online"
    except Exception as auth_exc:
        # Fall back to offline logging so experiment tracking never blocks notebook execution.
        print(f"W&B online login unavailable, switching to offline mode: {auth_exc}")
        WANDB_RUN = wandb.init(
            project="vK.7.1 Image Detection and Localisation.ipynb",
            name="vk71-unetwithclassifier-kaggle-first-offline",
            tags=["assignment", "kaggle-first", "tampering", "localization", "offline"],
            config=WANDB_CONFIG,
            mode="offline",
            reinit=True,
        )
        WANDB_MODE = "offline"

    WANDB_ACTIVE = WANDB_RUN is not None
except Exception as wandb_exc:
    WANDB_ACTIVE = False
    WANDB_RUN = None
    print(f"W&B setup unavailable; continuing without active logging: {wandb_exc}")

print("W&B active:", WANDB_ACTIVE)
print("W&B mode:", WANDB_MODE)'''
    set_source(cells[49], cell50_code)
    modified_code_indices.add(49)

    # Cell 91: conclusion wording
    replace_text(cells[90], "one Colab notebook covers", "one Kaggle-first notebook covers")
    replace_text(cells[90], "one Google Colab notebook", "one Kaggle-first notebook")

    # Clear stale outputs only for modified code cells that already had outputs.
    for idx in modified_code_indices:
        clear_outputs_if_needed(cells[idx])

    # Static verification on the generated notebook.
    checks = {
        "title": "Submission Notebook (vK.7.1)",
        "toc wording": "Kaggle-First Dataset Resolution",
        "kaggle-first resolution": "dataset_dir = find_dataset_in_input(KAGGLE_INPUT_DIR)",
        "dataset path cleanup": 'DATASET_DIR = ATTACHED_DATASET_DIR',
        "working dir csv output": 'output_csv = KAGGLE_WORKING_DIR / "image_mask_metadata.csv"',
        "working dir split output": 'output_dir = KAGGLE_WORKING_DIR',
        "wandb secret": 'UserSecretsClient().get_secret("WANDB_API_KEY")',
    }
    for label, needle in checks.items():
        if needle not in "".join(get_source(cell) for cell in cells):
            raise ValueError(f"Verification failed for {label}: missing {needle!r}")

    if 'INPUT_ROOT / "/kaggle/input/casia-spicing-detection-localization"' in get_source(cells[20]):
        raise ValueError("Verification failed: cell 21 still contains the old dataset path expression.")

    save_notebook(out_nb, OUTPUT_PATH)

    print("\n=== Verification Summary ===")
    print(f"Input notebook:  {INPUT_PATH.name}")
    print(f"Output notebook: {OUTPUT_PATH.name}")
    print(f"Cell count preserved: {len(src_nb['cells']) == len(out_nb['cells'])} ({len(out_nb['cells'])} cells)")
    print("Kaggle-first dataset resolution inserted: True")
    print("KAGGLE_WORKING_DIR metadata paths inserted: True")


if __name__ == "__main__":
    main()
