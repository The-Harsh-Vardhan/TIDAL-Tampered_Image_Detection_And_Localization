"""
Generate vK.7 notebook from vK.3 by restructuring cells and adding markdown documentation.
No code is modified - only cell splitting and markdown insertion.
"""

import json
import copy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
INPUT_PATH = NOTEBOOKS_DIR / "source" / "vK.3 Image Detection and Localisation [Code Comments were changed.ipynb"
OUTPUT_PATH = NOTEBOOKS_DIR / "source" / "vK.7 Image Detection and Localisation.ipynb"


def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Saved: {path}")


def make_md_cell(text):
    """Create a markdown cell from a string."""
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


def make_code_cell(source_lines):
    """Create a code cell from a list of source line strings (already with \\n)."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def get_source(cell):
    """Get cell source as a single string."""
    return "".join(cell["source"])


def split_source_at_markers(source_str, markers):
    """
    Split source code string at comment-based section markers.
    Returns list of code chunks (strings).
    Each chunk starts at the marker line.
    """
    lines = source_str.split("\n")
    # Find line indices where markers appear
    split_indices = [0]
    for i, line in enumerate(lines):
        if i == 0:
            continue
        stripped = line.strip()
        for marker in markers:
            if marker in stripped:
                split_indices.append(i)
                break

    chunks = []
    for idx in range(len(split_indices)):
        start = split_indices[idx]
        end = split_indices[idx + 1] if idx + 1 < len(split_indices) else len(lines)
        chunk_lines = lines[start:end]
        # Rejoin
        chunk = "\n".join(chunk_lines)
        chunks.append(chunk)

    return chunks


def source_to_lines(text):
    """Convert a source string to notebook source list format."""
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result


def main():
    nb = load_notebook(INPUT_PATH)
    old_cells = nb["cells"]
    new_cells = []

    # ===============================================
    # CELL 0: Add Table of Contents BEFORE the existing title cell
    # ===============================================
    toc = make_md_cell(
        "# Table of Contents\n"
        "\n"
        "1. [Introduction](#tampered-image-detection-and-localization---submission-notebook-vk7)\n"
        "2. [Project Objectives](#project-objectives-fulfilled-vs-remaining)\n"
        "3. [Environment Setup](#1-environment-setup)\n"
        "    - 1.1 Runtime Configuration and Kaggle-Style Directories\n"
        "    - 1.2 Dataset Discovery Helpers\n"
        "    - 1.3 Dataset Resolution from Google Drive or Kaggle\n"
        "    - 1.4 Suppress libpng Warnings\n"
        "4. [Dataset Explanation](#2-dataset-explanation)\n"
        "    - 2.1 Discover Dataset Root and Build Metadata\n"
        "    - 2.2 Train / Validation / Test Split\n"
        "5. [Data Loading and Preprocessing](#3-data-loading-and-preprocessing)\n"
        "6. [Prior Experiment Block (Source-Preserved)](#source-preserved-prior-experiment-block)\n"
        "    - 3.1 Package Installation and Warning Suppression\n"
        "    - 3.2 Imports\n"
        "    - 3.3 Metadata Loading\n"
        "    - 3.4 Image and Mask Transforms\n"
        "    - 3.5 Dataset Class Definition\n"
        "    - 3.6 U-Net with Classification Head\n"
        "    - 3.7 DataLoader Construction\n"
        "    - 3.8 Training Loop and Validation\n"
        "7. [Model Architecture](#4-model-architecture)\n"
        "8. [Training Strategy](#5-training-strategy)\n"
        "9. [Hyperparameters](#6-hyperparameters)\n"
        "10. [Experiment Tracking (W&B)](#7-experiment-tracking)\n"
        "11. [Effective Submission Training Loop](#8-effective-submission-training-loop)\n"
        "    - 8.1 Dependencies and Imports\n"
        "    - 8.2 Metadata Loading\n"
        "    - 8.3 Image and Mask Transforms\n"
        "    - 8.4 Dataset Class\n"
        "    - 8.5 U-Net with Classifier Architecture\n"
        "    - 8.6 DataLoader Construction\n"
        "    - 8.7 Loss Functions and Reporting Metrics\n"
        "    - 8.8 Training History and Metric Helpers\n"
        "    - 8.9 Training and Evaluation Routines\n"
        "    - 8.10 Training Loop Execution\n"
        "    - 8.11 Final Test Evaluation\n"
        "12. [Training Curves](#81-training-curves-and-learning-behavior)\n"
        "13. [Evaluation Methodology](#9-evaluation-methodology)\n"
        "14. [Visualization of Predictions](#10-visualization-of-predictions)\n"
        "    - 10.1 Sample Collection for Qualitative Review\n"
        "    - 10.2 Submission-Ready Prediction Panels\n"
        "15. [Conclusion](#conclusion)"
    )
    new_cells.append(toc)

    # ===============================================
    # CELL 0 (original): Title - update version reference
    # ===============================================
    cell0 = copy.deepcopy(old_cells[0])
    # Update the title to reflect vK.7
    cell0["source"] = [s.replace("vK.3", "vK.7") for s in cell0["source"]]
    new_cells.append(cell0)

    # ===============================================
    # CELL 1 (original): Project Objectives table - keep as-is
    # ===============================================
    new_cells.append(copy.deepcopy(old_cells[1]))

    # ===============================================
    # CELL 2 (original): Section 1. Environment Setup markdown - keep as-is
    # ===============================================
    new_cells.append(copy.deepcopy(old_cells[2]))

    # ===============================================
    # CELL 3 (original): Runtime config code - add subsection header
    # ===============================================
    new_cells.append(make_md_cell(
        "### 1.1 Runtime Configuration and Kaggle-Style Directories\n"
        "\n"
        "Installs required packages for the Colab runtime and creates the Kaggle-style directory structure\n"
        "(`/kaggle/input`, `/kaggle/working`) so that source notebook path references work without modification."
    ))
    new_cells.append(copy.deepcopy(old_cells[3]))

    # ===============================================
    # CELL 4 (original): Dataset discovery helpers (~208 lines) - SPLIT
    # ===============================================
    cell4_src = get_source(old_cells[4])

    # Split cell 4 into logical chunks using the function boundaries
    # Chunk 1: imports + constants + has_image_and_mask_dirs + sorted_unique_paths
    # Chunk 2: find_dataset_in_drive
    # Chunk 3: normalize_dataset_dir + ensure_dataset_from_kaggle
    # Chunk 4: main execution logic (dataset_dir = None ... to end)

    c4_lines = cell4_src.split("\n")

    # Find function start lines
    func_starts = {}
    exec_start = None
    for i, line in enumerate(c4_lines):
        if line.startswith("def find_dataset_in_drive"):
            func_starts["find_dataset_in_drive"] = i
        elif line.startswith("def normalize_dataset_dir"):
            func_starts["normalize_dataset_dir"] = i
        elif line.startswith("dataset_dir = None"):
            exec_start = i

    # Chunk 1: From start to find_dataset_in_drive
    chunk1_lines = c4_lines[0:func_starts["find_dataset_in_drive"]]
    # Chunk 2: find_dataset_in_drive to normalize_dataset_dir
    chunk2_lines = c4_lines[func_starts["find_dataset_in_drive"]:func_starts["normalize_dataset_dir"]]
    # Chunk 3: normalize_dataset_dir to exec_start
    chunk3_lines = c4_lines[func_starts["normalize_dataset_dir"]:exec_start]
    # Chunk 4: exec logic
    chunk4_lines = c4_lines[exec_start:]

    new_cells.append(make_md_cell(
        "### 1.2 Dataset Discovery Helpers\n"
        "\n"
        "Utility functions that validate dataset layout, search Google Drive for existing copies,\n"
        "normalize paths into the Kaggle-style directory, and fall back to a Kaggle API download when necessary."
    ))

    new_cells.append(make_md_cell(
        "#### 1.2.1 Layout Validation and Path Utilities\n"
        "\n"
        "These helpers check whether a directory contains the expected `IMAGE` and `MASK` subdirectories\n"
        "and deduplicate discovered paths for deterministic selection."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(chunk1_lines))))

    new_cells.append(make_md_cell(
        "#### 1.2.2 Google Drive Dataset Search\n"
        "\n"
        "Recursively searches mounted Google Drive directories for a dataset folder\n"
        "that matches the CASIA naming convention and contains the expected layout."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(chunk2_lines))))

    new_cells.append(make_md_cell(
        "#### 1.2.3 Dataset Normalization and Kaggle API Fallback\n"
        "\n"
        "`normalize_dataset_dir` symlinks or copies the discovered dataset into the Kaggle-style input path.\n"
        "`ensure_dataset_from_kaggle` downloads the dataset through the Kaggle API when Drive is unavailable."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(chunk3_lines))))

    new_cells.append(make_md_cell(
        "### 1.3 Dataset Resolution from Google Drive or Kaggle\n"
        "\n"
        "Attempts to mount Google Drive and locate an existing dataset copy.\n"
        "Falls back to the Kaggle API download when Drive-based discovery fails.\n"
        "Raises an error if no valid dataset can be prepared."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(chunk4_lines))))

    # ===============================================
    # CELL 5 (original): Suppress libpng warnings
    # ===============================================
    new_cells.append(make_md_cell(
        "### 1.4 Suppress libpng Warnings\n"
        "\n"
        "Redirects C-level stderr to `/dev/null` so that repetitive libpng decoding warnings\n"
        "do not clutter notebook output during image loading."
    ))
    new_cells.append(copy.deepcopy(old_cells[5]))

    # ===============================================
    # CELL 6 (original): Section 2. Dataset Explanation markdown - keep as-is
    # ===============================================
    new_cells.append(copy.deepcopy(old_cells[6]))

    # ===============================================
    # CELL 7 (original): Metadata discovery + build (~103 lines) - SPLIT
    # ===============================================
    cell7_src = get_source(old_cells[7])
    c7_lines = cell7_src.split("\n")

    # Find section markers
    c7_sec2 = None
    c7_sec3 = None
    for i, line in enumerate(c7_lines):
        if "2) Build metadata" in line:
            c7_sec2 = i
        elif "3) Save the metadata" in line:
            c7_sec3 = i

    # Go back to the comment block start (the line with "# ====")
    while c7_sec2 > 0 and not c7_lines[c7_sec2 - 1].strip().startswith("# ===="):
        c7_sec2 -= 1
    if c7_sec2 > 0 and c7_lines[c7_sec2 - 1].strip().startswith("# ===="):
        c7_sec2 -= 1

    while c7_sec3 > 0 and not c7_lines[c7_sec3 - 1].strip().startswith("# ===="):
        c7_sec3 -= 1
    if c7_sec3 > 0 and c7_lines[c7_sec3 - 1].strip().startswith("# ===="):
        c7_sec3 -= 1

    chunk7_1 = c7_lines[0:c7_sec2]
    chunk7_2 = c7_lines[c7_sec2:c7_sec3]
    chunk7_3 = c7_lines[c7_sec3:]

    new_cells.append(make_md_cell(
        "### 2.1 Discover Dataset Root and Build Metadata\n"
        "\n"
        "This subsection inspects the Kaggle-style input directory to confirm the dataset is present,\n"
        "then iterates over `IMAGE/Au`, `IMAGE/Tp`, `MASK/Au`, and `MASK/Tp` subdirectories\n"
        "to build a metadata table of image-mask pairs."
    ))
    new_cells.append(make_md_cell(
        "#### 2.1.1 Locate IMAGE and MASK Directories\n"
        "\n"
        "Scans the normalized Kaggle-style input path and searches recursively for the `IMAGE` and `MASK` folders."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(chunk7_1))))

    new_cells.append(make_md_cell(
        "#### 2.1.2 Build Metadata Table from Image-Mask Pairs\n"
        "\n"
        "Iterates over authentic (`Au`) and tampered (`Tp`) subdirectories,\n"
        "pairing each image with its corresponding mask and recording the image-level label."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(chunk7_2))))

    new_cells.append(make_md_cell(
        "#### 2.1.3 Save Metadata CSV\n"
        "\n"
        "Persists the image-mask-label metadata to a CSV file for use by downstream data loading cells."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(chunk7_3))))

    # ===============================================
    # CELL 8 (original): Train/val/test split
    # ===============================================
    new_cells.append(make_md_cell(
        "### 2.2 Train / Validation / Test Split\n"
        "\n"
        "Loads the metadata CSV, applies a stratified 70/15/15 split to preserve class balance\n"
        "across train, validation, and test subsets, and saves each split to a separate CSV file."
    ))
    new_cells.append(copy.deepcopy(old_cells[8]))

    # ===============================================
    # CELL 9 (original): Section 3 Data Loading markdown - keep as-is
    # ===============================================
    new_cells.append(copy.deepcopy(old_cells[9]))

    # ===============================================
    # CELL 10 (original): Prior experiment block intro markdown - keep as-is
    # ===============================================
    new_cells.append(copy.deepcopy(old_cells[10]))

    # ===============================================
    # CELL 11 (original): Prior experiment block (~626 lines) - SPLIT
    # ===============================================
    cell11_src = get_source(old_cells[11])
    c11_lines = cell11_src.split("\n")

    # Find section markers by the ====== comment blocks
    c11_sections = {}
    for i, line in enumerate(c11_lines):
        stripped = line.strip()
        if "1) Suppress image-reading" in stripped:
            c11_sections["suppress"] = i
        elif "2) Imports" in stripped and "====" in c11_lines[max(0, i-1)]:
            c11_sections["imports"] = i
        elif "3) Load metadata CSV" in stripped:
            c11_sections["metadata"] = i
        elif "4) Define image and mask transforms" in stripped:
            c11_sections["transforms"] = i
        elif "5) Dataset definition" in stripped:
            c11_sections["dataset"] = i
        elif "6) Define the U-Net" in stripped:
            c11_sections["unet"] = i
        elif "7) Build dataloaders" in stripped:
            c11_sections["dataloaders"] = i
        elif "8) Train the earlier" in stripped:
            c11_sections["train"] = i

    # Helper: find the == banner start for each section
    def banner_start(idx):
        """Go back to the ===== line before the section header."""
        while idx > 0 and "=====" not in c11_lines[idx - 1]:
            idx -= 1
        if idx > 0 and "=====" in c11_lines[idx - 1]:
            idx -= 1
        return idx

    # Chunk boundaries
    # pip install line is the first line
    pip_end = 1  # line 0 is "!pip install ..."
    # Then blank line + import sys, os...
    suppress_start = banner_start(c11_sections["suppress"])
    imports_start = banner_start(c11_sections["imports"])
    metadata_start = banner_start(c11_sections["metadata"])
    transforms_start = banner_start(c11_sections["transforms"])
    dataset_start = banner_start(c11_sections["dataset"])
    unet_start = banner_start(c11_sections["unet"])
    dataloaders_start = banner_start(c11_sections["dataloaders"])
    train_start = banner_start(c11_sections["train"])

    # Split into chunks
    c11_chunk_pip = c11_lines[0:pip_end]
    # pip line is standalone, suppress includes "import sys, os, contextlib"
    c11_chunk_suppress = c11_lines[pip_end:imports_start]
    c11_chunk_imports = c11_lines[imports_start:metadata_start]
    c11_chunk_metadata = c11_lines[metadata_start:transforms_start]
    c11_chunk_transforms = c11_lines[transforms_start:dataset_start]
    c11_chunk_dataset = c11_lines[dataset_start:unet_start]
    c11_chunk_unet = c11_lines[unet_start:dataloaders_start]
    c11_chunk_dataloaders = c11_lines[dataloaders_start:train_start]
    c11_chunk_train = c11_lines[train_start:]

    # Add markdown + code cells for each chunk
    new_cells.append(make_md_cell(
        "#### 3.1 Package Installation\n"
        "\n"
        "Installs `albumentations` and `opencv-python-headless` for image augmentation and loading."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c11_chunk_pip))))

    new_cells.append(make_md_cell(
        "#### 3.2 Warning Suppression and Imports\n"
        "\n"
        "Defines a context manager that temporarily silences stderr to suppress repetitive\n"
        "libpng/OpenCV warnings during image loading, followed by all required imports."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c11_chunk_suppress + c11_chunk_imports))))

    new_cells.append(make_md_cell(
        "#### 3.3 Metadata Loading\n"
        "\n"
        "Reads the train, validation, and test CSV files produced by the dataset preparation step."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c11_chunk_metadata))))

    new_cells.append(make_md_cell(
        "#### 3.4 Image and Mask Transforms\n"
        "\n"
        "Defines Albumentations pipelines for the training split (with augmentation) and the\n"
        "validation/test splits (resize and normalize only)."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c11_chunk_transforms))))

    new_cells.append(make_md_cell(
        "#### 3.5 Dataset Class Definition\n"
        "\n"
        "The `ImageMaskDataset` class reads paired images and masks, applies transforms,\n"
        "and returns tensors suitable for classification and segmentation training."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c11_chunk_dataset))))

    new_cells.append(make_md_cell(
        "#### 3.6 U-Net with Classification Head\n"
        "\n"
        "Defines the `DoubleConv`, `Down`, `Up`, and `UNetWithClassifier` modules that form\n"
        "the shared encoder-decoder backbone and the bottleneck classification head."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c11_chunk_unet))))

    new_cells.append(make_md_cell(
        "#### 3.7 DataLoader Construction\n"
        "\n"
        "Creates PyTorch `DataLoader` instances for the train, validation, and test datasets."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c11_chunk_dataloaders))))

    new_cells.append(make_md_cell(
        "#### 3.8 Training Loop and Validation\n"
        "\n"
        "Runs the prior experiment configuration: trains for 30 epochs using Adam with ReduceLROnPlateau,\n"
        "checkpoints the best model by validation accuracy, and evaluates on the test split."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c11_chunk_train))))

    # ===============================================
    # CELL 12-17 (original): Model architecture + Training strategy + Hyperparams + W&B - keep as-is
    # ===============================================
    for ci in range(12, 18):
        new_cells.append(copy.deepcopy(old_cells[ci]))

    # ===============================================
    # CELL 18 (original): Effective training (~814 lines) - SPLIT
    # ===============================================
    cell18_src = get_source(old_cells[18])
    c18_lines = cell18_src.split("\n")

    # Find section markers
    c18_sections = {}
    for i, line in enumerate(c18_lines):
        stripped = line.strip()
        if "================== Imports ==================" in stripped:
            c18_sections["imports"] = i
        elif "================== 1) Load metadata" in stripped:
            c18_sections["metadata"] = i
        elif "================== 2) Define transforms" in stripped:
            c18_sections["transforms"] = i
        elif "================== 3) Define the dataset" in stripped:
            c18_sections["dataset"] = i
        elif "================== 4) Define the U-Net" in stripped:
            c18_sections["unet"] = i
        elif "================== 5) Build dataloaders" in stripped:
            c18_sections["dataloaders"] = i
        elif "================== 6) Loss functions" in stripped:
            c18_sections["loss"] = i
        elif "================== 7) Training history" in stripped:
            c18_sections["history"] = i
        elif "================== 8) Run the effective" in stripped:
            c18_sections["run"] = i
        elif "================== 9) Evaluate the best" in stripped:
            c18_sections["test_eval"] = i

    # Split at markers
    # Chunk: deps + silence (lines 0 to imports)
    c18_deps = c18_lines[0:c18_sections["imports"]]
    # Chunk: imports (to metadata)
    c18_imports = c18_lines[c18_sections["imports"]:c18_sections["metadata"]]
    # Chunk: metadata (to transforms)
    c18_metadata = c18_lines[c18_sections["metadata"]:c18_sections["transforms"]]
    # Chunk: transforms (to dataset)
    c18_transforms = c18_lines[c18_sections["transforms"]:c18_sections["dataset"]]
    # Chunk: dataset (to unet)
    c18_dataset = c18_lines[c18_sections["dataset"]:c18_sections["unet"]]
    # Chunk: unet (to dataloaders)
    c18_unet = c18_lines[c18_sections["unet"]:c18_sections["dataloaders"]]
    # Chunk: dataloaders (to loss)
    c18_dataloaders = c18_lines[c18_sections["dataloaders"]:c18_sections["loss"]]
    # Chunk: loss (to history)
    c18_loss = c18_lines[c18_sections["loss"]:c18_sections["history"]]
    # Chunk: history + metric helpers (to run)
    c18_history = c18_lines[c18_sections["history"]:c18_sections["run"]]

    # For the training loop section, we need to find train_one_epoch and evaluate functions
    # then the actual loop
    run_lines = c18_lines[c18_sections["run"]:]
    # Find "def train_one_epoch" and "def evaluate" within the history chunk
    # Actually, train_one_epoch and evaluate are in the history chunk (section 7)
    # Let me re-check: section 7 is "Training history and metrics", section 8 is "Run the effective training loop"
    # The functions train_one_epoch and evaluate are likely between 7 and 8

    # Let me find them precisely
    train_fn_start = None
    eval_fn_start = None
    for i, line in enumerate(c18_lines):
        if line.startswith("def train_one_epoch"):
            train_fn_start = i
        elif line.startswith("def evaluate"):
            eval_fn_start = i

    # Re-split: history+metrics goes from section 7 marker to train_one_epoch
    c18_history = c18_lines[c18_sections["history"]:train_fn_start]
    # Training and eval functions
    c18_train_fns = c18_lines[train_fn_start:c18_sections["run"]]
    # Run loop
    c18_run = c18_lines[c18_sections["run"]:c18_sections["test_eval"]]
    # Test eval
    c18_test = c18_lines[c18_sections["test_eval"]:]

    new_cells.append(make_md_cell(
        "### 8.1 Dependencies and Imports\n"
        "\n"
        "Installs required packages, silences libpng warnings, and imports all libraries\n"
        "needed for the effective submission run."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_deps + c18_imports))))

    new_cells.append(make_md_cell(
        "### 8.2 Metadata Loading\n"
        "\n"
        "Reads the train, validation, and test metadata CSV files generated during dataset preparation."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_metadata))))

    new_cells.append(make_md_cell(
        "### 8.3 Image and Mask Transforms\n"
        "\n"
        "Defines the Albumentations augmentation pipeline for training (with flip, brightness,\n"
        "noise, JPEG compression, and shift-scale-rotate) and the deterministic pipeline for validation/test."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_transforms))))

    new_cells.append(make_md_cell(
        "### 8.4 Dataset Class\n"
        "\n"
        "The `ImageMaskDataset` class loads image-mask pairs from metadata, applies shared transforms,\n"
        "and returns tensors suitable for joint classification and segmentation training."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_dataset))))

    new_cells.append(make_md_cell(
        "### 8.5 U-Net with Classifier Architecture\n"
        "\n"
        "Defines the encoder-decoder backbone (`DoubleConv`, `Down`, `Up`) and the full\n"
        "`UNetWithClassifier` model that produces both image-level classification logits\n"
        "and pixel-level segmentation logits from a shared representation."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_unet))))

    new_cells.append(make_md_cell(
        "### 8.6 DataLoader Construction\n"
        "\n"
        "Creates train, validation, and test `DataLoader` instances with batch size 8 and 2 workers."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_dataloaders))))

    new_cells.append(make_md_cell(
        "### 8.7 Loss Functions and Reporting Metrics\n"
        "\n"
        "Defines the focal-style classification loss, the combined BCE + Dice segmentation loss,\n"
        "and the Dice coefficient used for checkpoint-independent reporting.\n"
        "Configures the Adam optimizer with learning rate `1e-4` and CosineAnnealingLR scheduler."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_loss))))

    new_cells.append(make_md_cell(
        "### 8.8 Training History and Metric Helpers\n"
        "\n"
        "Initializes the training history dictionary and defines IoU and F1 scoring functions\n"
        "used for validation reporting (these do not affect training decisions)."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_history))))

    new_cells.append(make_md_cell(
        "### 8.9 Training and Evaluation Routines\n"
        "\n"
        "`train_one_epoch` trains the model for one pass over the training split using the combined\n"
        "classification and segmentation loss.\n\n"
        "`evaluate` computes loss, accuracy, Dice, IoU, and F1 on a given dataloader."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_train_fns))))

    new_cells.append(make_md_cell(
        "### 8.10 Training Loop Execution\n"
        "\n"
        "Runs the effective submission training loop for 50 epochs.\n"
        "Checkpoints the model whenever validation accuracy improves.\n"
        "Logs metrics to W&B when experiment tracking is active."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_run))))

    new_cells.append(make_md_cell(
        "### 8.11 Final Test Evaluation\n"
        "\n"
        "Loads the best checkpoint and evaluates it on the held-out test split.\n"
        "Stores the results in `TRAINING_HISTORY` and `FINAL_TEST_METRICS` for downstream reporting."
    ))
    new_cells.append(make_code_cell(source_to_lines("\n".join(c18_test))))

    # ===============================================
    # CELLS 19-36 (original): Keep as-is (already well-structured)
    # ===============================================
    for ci in range(19, len(old_cells)):
        new_cells.append(copy.deepcopy(old_cells[ci]))

    # Build the output notebook
    out_nb = copy.deepcopy(nb)
    out_nb["cells"] = new_cells

    save_notebook(out_nb, OUTPUT_PATH)

    # Print verification summary
    print("\n=== Verification Summary ===")
    print(f"Original cells: {len(old_cells)}")
    print(f"Output cells: {len(new_cells)}")

    # Extract all code from original
    orig_code = []
    for cell in old_cells:
        if cell["cell_type"] == "code":
            orig_code.append(get_source(cell))

    # Extract all code from output
    new_code = []
    for cell in new_cells:
        if cell["cell_type"] == "code":
            new_code.append(get_source(cell))

    # Concatenated code comparison
    orig_all = "\n".join(orig_code)
    new_all = "\n".join(new_code)

    if orig_all == new_all:
        print("CODE MATCH: All code is character-for-character identical.")
    else:
        # Find first difference
        for i, (a, b) in enumerate(zip(orig_all, new_all)):
            if a != b:
                print(f"CODE MISMATCH at char {i}:")
                print(f"  Original: ...{repr(orig_all[max(0,i-30):i+30])}...")
                print(f"  New:      ...{repr(new_all[max(0,i-30):i+30])}...")
                break
        if len(orig_all) != len(new_all):
            print(f"LENGTH DIFF: orig={len(orig_all)} new={len(new_all)}")


if __name__ == "__main__":
    main()
