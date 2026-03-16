"""
Build the vK.6 notebook from the vK.3 source notebook.

The transformation is source-preserving:
- all original code is copied verbatim
- only markdown cells and code-cell boundaries are added
- top-level notebook metadata is preserved exactly
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
SOURCE_NOTEBOOK = NOTEBOOKS_DIR / "source" / "vK.3 Image Detection and Localisation [Code Comments were changed.ipynb"
TARGET_NOTEBOOK = NOTEBOOKS_DIR / "source" / "vK.6 Image Detection and Localisation [Subsections by Codex].ipynb"


TOC_MARKDOWN = """## Table of Contents

1. Project Objectives: Fulfilled vs Remaining
2. Environment Setup
3. Dataset Explanation
4. Data Loading and Preprocessing
5. Source-Preserved Prior Experiment Block
6. Model Architecture
7. Training Strategy
8. Hyperparameters
9. Experiment Tracking
10. Effective Submission Training Loop
11. Evaluation Methodology
12. Visualization of Predictions
13. Conclusion
"""


INTRO_MARKDOWN = {
    4: """### Runtime Package Setup

This cell prepares the notebook runtime by detecting whether execution is happening in Google Colab, installing the required packages for that environment, and creating the Kaggle-style input and working directories expected later in the notebook.
""",
    6: """### Environment Verification

This short cell redirects low-level stderr noise away from the notebook display so the remaining setup, training, and visualization steps stay easier to read without changing how the underlying code executes.
""",
    17: """### W&B Initialization

This cell configures experiment tracking for the effective submission run. It keeps the original fallback behavior intact so execution can continue even when online authentication is unavailable.
""",
    21: """### Training History Plots

This cell converts the recorded training history into tables and figures so the reader can inspect optimization behavior, image-level accuracy, and segmentation quality over time.
""",
    23: """### Checkpoint Reload

This cell reloads the best saved model weights before the notebook transitions into post-training evaluation and qualitative review.
""",
    24: """### Image Denormalization Helper

This helper reverses the preprocessing normalization so images can be displayed in a human-readable RGB format during the visualization sections that follow.
""",
    27: """### Balanced Sample Collection

This helper walks through the test loader and collects a balanced set of authentic and tampered examples together with their predictions for later qualitative inspection.
""",
    28: """### Overlay Visualization

This cell defines an overlay-based display and immediately renders examples where predicted tampered pixels are highlighted on top of the original image.
""",
    29: """### Two-Row Image And Mask Grid

This helper presents each example as an image-and-mask pair so image-level predictions and localization masks can be reviewed side by side.
""",
    30: """### Five-Sample Grid Rendering

This cell uses the preceding helper to render the first balanced set of authentic and tampered examples in the two-row layout.
""",
    31: """### Expanded Sample Collection

This cell refreshes the qualitative sample pool with a larger balanced set so the notebook can show denser visualization grids in the next subsection.
""",
    32: """### Three-Per-Row Visualization Helper

This helper redefines the image-and-mask grid so a larger number of qualitative examples can be displayed in a more compact three-per-row format.
""",
    33: """### Ten-Sample Grid Rendering

This cell renders the larger authentic and tampered sample sets using the compact multi-row layout defined immediately above.
""",
    36: """### Four-Panel Submission Grid

This final visualization block assembles reviewer-friendly panels that align the original image, ground-truth mask, predicted mask, and overlay for quick qualitative assessment.
""",
}


SPLIT_MARKDOWN = {
    5: [
        (
            "### Colab Runtime Paths\n\n"
            "This fragment defines the preserved runtime flags and path constants that let the notebook mimic the Kaggle directory structure expected by the downstream dataset cells."
        ),
        (
            "### Dataset Discovery Helpers\n\n"
            "These helper functions search mounted Google Drive locations and identify candidate dataset roots that already contain the required `IMAGE` and `MASK` directories."
        ),
        (
            "### Dataset Normalization And Kaggle Fallback\n\n"
            "These helpers normalize a discovered dataset into the expected Kaggle-style path and fall back to a Kaggle API download only when Drive-based discovery does not succeed."
        ),
        (
            "### Dataset Resolution And Validation\n\n"
            "This final fragment executes the preserved discovery flow, applies the fallback logic when needed, and validates that the normalized dataset root is ready for the metadata-building steps that follow."
        ),
    ],
    8: [
        (
            "### Dataset Root Discovery\n\n"
            "This fragment inspects the Kaggle input area and resolves the concrete dataset folders used to locate images and masks without changing the original path assumptions."
        ),
        (
            "### Image And Mask Metadata Assembly\n\n"
            "This fragment iterates through the authentic and tampered subfolders, pairs images with masks, and builds the metadata rows used throughout the rest of the notebook."
        ),
        (
            "### Metadata Export And Sanity Checks\n\n"
            "This fragment materializes the metadata table as a CSV file and prints lightweight checks so class balance and missing-mask cases remain visible before model training begins."
        ),
    ],
    9: [
        (
            "### Metadata Split Loading\n\n"
            "This fragment loads the metadata CSV generated above and reports the overall dataset size before any partitioning takes place."
        ),
        (
            "### Stratified Train Validation Test Split\n\n"
            "This fragment applies the preserved stratified splitting logic so the authentic-versus-tampered class balance is maintained across train, validation, and test subsets."
        ),
        (
            "### Split Export Summary\n\n"
            "This fragment writes the three split CSV files used by the training code and prints their locations for traceability."
        ),
    ],
    12: [
        (
            "### Prior Experiment Dependencies And Warning Suppression\n\n"
            "This fragment preserves the earlier experiment's dependency installation and warning-suppression setup exactly as it appeared in the source notebook."
        ),
        (
            "### Prior Experiment Imports\n\n"
            "This fragment gathers the libraries used by the earlier experiment block without altering the original import list."
        ),
        (
            "### Prior Experiment Metadata Loading\n\n"
            "This fragment loads the preserved metadata CSV paths used by the earlier experiment and prints the split sizes for context."
        ),
        (
            "### Prior Experiment Training Transform\n\n"
            "This fragment defines the training-time augmentation pipeline used by the earlier experiment configuration."
        ),
        (
            "### Prior Experiment Validation Transform\n\n"
            "This fragment defines the deterministic preprocessing applied to validation and test samples in the earlier experiment block."
        ),
        (
            "### Prior Experiment Dataset Class\n\n"
            "This fragment contains the dataset implementation that reads paired images, binary masks, and image-level labels from the metadata table."
        ),
        (
            "### Prior Experiment DoubleConv Block\n\n"
            "This fragment introduces the reusable double-convolution building block used throughout the preserved U-Net backbone."
        ),
        (
            "### Prior Experiment Encoder Down Block\n\n"
            "This fragment defines the encoder downsampling block used in the earlier experiment architecture."
        ),
        (
            "### Prior Experiment Decoder Up Block\n\n"
            "This fragment defines the decoder upsampling block that fuses skip connections during localization."
        ),
        (
            "### Prior Experiment Shared U-Net Model\n\n"
            "This fragment preserves the combined segmentation-and-classification model definition used by the prior experiment block."
        ),
        (
            "### Prior Experiment Dataloaders\n\n"
            "This fragment instantiates the preserved datasets and dataloaders that feed the earlier experiment."
        ),
        (
            "### Prior Experiment Optimization Setup\n\n"
            "This fragment prepares the device, model instance, loss functions, optimizer, scheduler, and task-weighting constants used before training begins."
        ),
        (
            "### Prior Experiment Dice Metric\n\n"
            "This fragment defines the overlap metric used to summarize segmentation quality in the earlier experiment."
        ),
        (
            "### Prior Experiment Training Step\n\n"
            "This fragment contains the epoch-level training routine preserved from the source notebook."
        ),
        (
            "### Prior Experiment Validation Step\n\n"
            "This fragment contains the validation routine used to monitor the earlier experiment configuration."
        ),
        (
            "### Prior Experiment Epoch Loop And Test Check\n\n"
            "This fragment runs the preserved training loop, reloads the best checkpoint, and evaluates the earlier experiment on its held-out split."
        ),
    ],
    19: [
        (
            "### Submission Dependencies And Warning Suppression\n\n"
            "This fragment preserves the effective submission run's dependency installation and low-level warning redirection exactly as in the source notebook."
        ),
        (
            "### Submission Imports And Tracking Globals\n\n"
            "This fragment gathers the libraries and global W&B flags used by the main submission training block."
        ),
        (
            "### Submission Metadata Loading\n\n"
            "This fragment loads the train, validation, and test metadata files consumed by the effective submission run."
        ),
        (
            "### Submission Training Transform\n\n"
            "This fragment defines the training-time augmentation pipeline used by the final submission run."
        ),
        (
            "### Submission Validation Transform\n\n"
            "This fragment defines the deterministic preprocessing used for validation and test examples in the final run."
        ),
        (
            "### Submission Dataset Class\n\n"
            "This fragment preserves the dataset implementation that returns images, masks, and image-level labels for the final run."
        ),
        (
            "### Submission DoubleConv Block\n\n"
            "This fragment defines the shared double-convolution block used by the final model."
        ),
        (
            "### Submission Encoder Down Block\n\n"
            "This fragment defines the encoder downsampling block used in the effective submission architecture."
        ),
        (
            "### Submission Decoder Up Block\n\n"
            "This fragment defines the decoder upsampling block used in the effective submission architecture."
        ),
        (
            "### Submission Shared U-Net Model\n\n"
            "This fragment preserves the joint classification-and-localization model used for the main assignment run."
        ),
        (
            "### Submission Dataloaders\n\n"
            "This fragment instantiates the dataloaders that feed the final training, validation, and test loops."
        ),
        (
            "### Submission Device And Class-Weight Setup\n\n"
            "This fragment moves the model onto the active device and prepares the class-weight information used by the image-level classification loss."
        ),
        (
            "### Submission Focal Loss Definition\n\n"
            "This fragment defines the focal-style classification loss preserved in the final submission run."
        ),
        (
            "### Submission Dice Loss Definition\n\n"
            "This fragment defines the soft Dice loss used as part of the segmentation objective."
        ),
        (
            "### Submission Dice Metric Definition\n\n"
            "This fragment defines the Dice reporting metric used during evaluation."
        ),
        (
            "### Submission Loss Objects And Optimizer Setup\n\n"
            "This fragment instantiates the preserved loss objects, optimizer, scheduler, and task-weighting constants used by the effective training loop."
        ),
        (
            "### Submission Training History Setup\n\n"
            "This fragment initializes the history containers used to track the main run across epochs."
        ),
        (
            "### Submission IoU Metric Definition\n\n"
            "This fragment defines the Intersection over Union metric used for segmentation reporting."
        ),
        (
            "### Submission F1 Metric Definition\n\n"
            "This fragment defines the F1 reporting metric for thresholded segmentation outputs."
        ),
        (
            "### Submission Training Step\n\n"
            "This fragment contains the epoch-level training routine used by the effective submission run."
        ),
        (
            "### Submission Evaluation Step\n\n"
            "This fragment contains the validation and test evaluation routine used by the effective submission run."
        ),
        (
            "### Submission Epoch Loop And Checkpointing\n\n"
            "This fragment runs the main training loop, logs metrics, and saves the best checkpoint according to the preserved validation-accuracy rule."
        ),
        (
            "### Submission Final Test Evaluation\n\n"
            "This fragment reloads the best checkpoint, evaluates it on the test split, and exposes the final history and metric objects for the downstream reporting cells."
        ),
    ],
}


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def to_source_list(source: list[str] | str) -> list[str]:
    if isinstance(source, list):
        return "".join(source).splitlines(keepends=True)
    return source.splitlines(keepends=True)


def text_to_source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def stripped(line: str) -> str:
    return line.rstrip("\r\n")


def find_exact(lines: list[str], text: str, start: int = 0) -> int:
    for idx in range(start, len(lines)):
        if stripped(lines[idx]) == text:
            return idx
    raise ValueError(f"Could not find line: {text}")


def find_sequence(lines: list[str], sequence: list[str], start: int = 0) -> int:
    limit = len(lines) - len(sequence) + 1
    for idx in range(start, limit):
        if all(stripped(lines[idx + offset]) == expected for offset, expected in enumerate(sequence)):
            return idx
    raise ValueError(f"Could not find sequence: {sequence}")


def make_new_id(seed: str, used_ids: set[str]) -> str:
    attempt = 0
    while True:
        suffix = f"{seed}-{attempt}" if attempt else seed
        candidate = hashlib.sha1(suffix.encode("utf-8")).hexdigest()[:8]
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        attempt += 1


def markdown_cell(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text_to_source(text),
        "id": cell_id,
    }


def code_fragment(
    original_cell: dict,
    lines: list[str],
    *,
    cell_id: str,
    preserve_original_state: bool,
) -> dict:
    return {
        "cell_type": "code",
        "metadata": copy.deepcopy(original_cell.get("metadata", {})) if preserve_original_state else {},
        "source": list(lines),
        "execution_count": original_cell.get("execution_count") if preserve_original_state else None,
        "outputs": copy.deepcopy(original_cell.get("outputs", [])) if preserve_original_state else [],
        "id": cell_id,
    }


def split_ranges_for_cell(cell_index: int, lines: list[str]) -> list[tuple[int, int]]:
    if cell_index == 5:
        idx_has = find_exact(lines, "def has_image_and_mask_dirs(path: Path) -> bool:")
        idx_normalize = find_exact(lines, "def normalize_dataset_dir(source_dir: Path, target_dir: Path) -> Path:")
        idx_dataset = find_exact(lines, "dataset_dir = None")
        return [
            (0, idx_has),
            (idx_has, idx_normalize),
            (idx_normalize, idx_dataset),
            (idx_dataset, len(lines)),
        ]

    if cell_index == 8:
        idx_section_2 = find_sequence(
            lines,
            [
                "# =========================",
                "# 2) Build metadata for images and masks",
                "# =========================",
            ],
        )
        idx_section_3 = find_sequence(
            lines,
            [
                "# =========================",
                "# 3) Save the metadata CSV",
                "# =========================",
            ],
            start=idx_section_2 + 1,
        )
        return [
            (0, idx_section_2),
            (idx_section_2, idx_section_3),
            (idx_section_3, len(lines)),
        ]

    if cell_index == 9:
        idx_section_2 = find_exact(
            lines,
            "# 2) Split the dataset into train, validation, and test subsets using stratified sampling.",
        )
        idx_section_3 = find_exact(
            lines,
            "# 3) Save the split metadata files so the downstream training code can load them directly.",
            start=idx_section_2 + 1,
        )
        return [
            (0, idx_section_2),
            (idx_section_2, idx_section_3),
            (idx_section_3, len(lines)),
        ]

    if cell_index == 12:
        idx_section_2 = find_sequence(
            lines,
            [
                "# =====================================================",
                "# 2) Imports",
                "# =====================================================",
            ],
        )
        idx_section_3 = find_sequence(
            lines,
            [
                "# =====================================================",
                "# 3) Load metadata CSV files",
                "# =====================================================",
            ],
            start=idx_section_2 + 1,
        )
        idx_section_4 = find_sequence(
            lines,
            [
                "# =====================================================",
                "# 4) Define image and mask transforms",
                "# =====================================================",
            ],
            start=idx_section_3 + 1,
        )
        idx_valid_transform = find_exact(lines, "def get_valid_transform():", start=idx_section_4 + 1)
        idx_section_5 = find_sequence(
            lines,
            [
                "# =====================================================",
                "# 5) Dataset definition",
                "# =====================================================",
            ],
            start=idx_valid_transform + 1,
        )
        idx_section_6 = find_sequence(
            lines,
            [
                "# =====================================================",
                "# 6) Define the U-Net with a classification head",
                "# =====================================================",
            ],
            start=idx_section_5 + 1,
        )
        idx_down = find_exact(lines, "class Down(nn.Module):", start=idx_section_6 + 1)
        idx_up = find_exact(lines, "class Up(nn.Module):", start=idx_down + 1)
        idx_unet = find_exact(lines, "class UNetWithClassifier(nn.Module):", start=idx_up + 1)
        idx_section_7 = find_sequence(
            lines,
            [
                "# =====================================================",
                "# 7) Build dataloaders",
                "# =====================================================",
            ],
            start=idx_unet + 1,
        )
        idx_section_8 = find_sequence(
            lines,
            [
                "# =====================================================",
                "# 8) Train the earlier experiment configuration",
                "# =====================================================",
            ],
            start=idx_section_7 + 1,
        )
        idx_dice = find_exact(lines, "def dice_coef(preds, targets, eps=1e-7):", start=idx_section_8 + 1)
        idx_train = find_exact(lines, "def train_one_epoch(epoch):", start=idx_dice + 1)
        idx_validate = find_exact(lines, 'def validate(epoch, loader, name="Val"):', start=idx_train + 1)
        idx_loop = find_exact(lines, "NUM_EPOCHS = 30", start=idx_validate + 1)
        return [
            (0, idx_section_2),
            (idx_section_2, idx_section_3),
            (idx_section_3, idx_section_4),
            (idx_section_4, idx_valid_transform),
            (idx_valid_transform, idx_section_5),
            (idx_section_5, idx_section_6),
            (idx_section_6, idx_down),
            (idx_down, idx_up),
            (idx_up, idx_unet),
            (idx_unet, idx_section_7),
            (idx_section_7, idx_section_8),
            (idx_section_8, idx_dice),
            (idx_dice, idx_train),
            (idx_train, idx_validate),
            (idx_validate, idx_loop),
            (idx_loop, len(lines)),
        ]

    if cell_index == 19:
        idx_imports = find_exact(lines, "# ================== Imports ==================")
        idx_load = find_exact(lines, "# ================== 1) Load metadata files ==================", start=idx_imports + 1)
        idx_transforms = find_exact(lines, "# ================== 2) Define transforms ==================", start=idx_load + 1)
        idx_valid_transform = find_exact(lines, "def get_valid_transform():", start=idx_transforms + 1)
        idx_dataset = find_exact(lines, "# ================== 3) Define the dataset ==================", start=idx_valid_transform + 1)
        idx_model = find_exact(
            lines,
            "# ================== 4) Define the U-Net with classifier head ==================",
            start=idx_dataset + 1,
        )
        idx_down = find_exact(lines, "class Down(nn.Module):", start=idx_model + 1)
        idx_up = find_exact(lines, "class Up(nn.Module):", start=idx_down + 1)
        idx_unet = find_exact(lines, "class UNetWithClassifier(nn.Module):", start=idx_up + 1)
        idx_loaders = find_exact(lines, "# ================== 5) Build dataloaders ==================", start=idx_unet + 1)
        idx_loss = find_exact(
            lines,
            "# ================== 6) Loss functions and reporting metrics ==================",
            start=idx_loaders + 1,
        )
        idx_focal = find_exact(lines, "class FocalLoss(nn.Module):", start=idx_loss + 1)
        idx_dice_loss = find_exact(lines, "def dice_loss(pred, target, eps=1e-7):", start=idx_focal + 1)
        idx_dice_coef = find_exact(lines, "def dice_coef(pred, target, eps=1e-7):", start=idx_dice_loss + 1)
        idx_loss_setup = find_exact(lines, "criterion_cls = FocalLoss(alpha=class_weights)", start=idx_dice_coef + 1)
        idx_history = find_exact(lines, "# ================== 7) Training history and metrics ==================", start=idx_loss_setup + 1)
        idx_iou = find_exact(lines, "def iou_coef(pred, target, eps=1e-7):", start=idx_history + 1)
        idx_f1 = find_exact(lines, "def f1_coef(pred, target, eps=1e-7):", start=idx_iou + 1)
        idx_train = find_exact(lines, "def train_one_epoch(epoch):", start=idx_f1 + 1)
        idx_evaluate = find_exact(
            lines,
            'def evaluate(epoch, loader, name="Val", return_details=False):',
            start=idx_train + 1,
        )
        idx_loop = find_exact(lines, "# ================== 8) Run the effective training loop ==================", start=idx_evaluate + 1)
        idx_test = find_exact(
            lines,
            "# ================== 9) Evaluate the best checkpoint on the test split ==================",
            start=idx_loop + 1,
        )
        return [
            (0, idx_imports),
            (idx_imports, idx_load),
            (idx_load, idx_transforms),
            (idx_transforms, idx_valid_transform),
            (idx_valid_transform, idx_dataset),
            (idx_dataset, idx_model),
            (idx_model, idx_down),
            (idx_down, idx_up),
            (idx_up, idx_unet),
            (idx_unet, idx_loaders),
            (idx_loaders, idx_loss),
            (idx_loss, idx_focal),
            (idx_focal, idx_dice_loss),
            (idx_dice_loss, idx_dice_coef),
            (idx_dice_coef, idx_loss_setup),
            (idx_loss_setup, idx_history),
            (idx_history, idx_iou),
            (idx_iou, idx_f1),
            (idx_f1, idx_train),
            (idx_train, idx_evaluate),
            (idx_evaluate, idx_loop),
            (idx_loop, idx_test),
            (idx_test, len(lines)),
        ]

    raise ValueError(f"Unsupported split cell index: {cell_index}")


def build_notebook(source_nb: dict) -> tuple[dict, dict[int, list[dict]]]:
    used_ids = {
        cell["id"]
        for cell in source_nb["cells"]
        if isinstance(cell, dict) and isinstance(cell.get("id"), str) and cell["id"]
    }

    new_nb = copy.deepcopy(source_nb)
    new_cells: list[dict] = []
    grouped_code_cells: dict[int, list[dict]] = {}

    for cell_index, source_cell in enumerate(source_nb["cells"], start=1):
        cell_type = source_cell["cell_type"]

        if cell_index == 1:
            new_cells.append(copy.deepcopy(source_cell))
            new_cells.append(markdown_cell(TOC_MARKDOWN, make_new_id("vk6-toc", used_ids)))
            continue

        if cell_index in INTRO_MARKDOWN:
            new_cells.append(
                markdown_cell(
                    INTRO_MARKDOWN[cell_index],
                    make_new_id(f"vk6-intro-{cell_index}", used_ids),
                )
            )

        if cell_type == "code" and cell_index in SPLIT_MARKDOWN:
            source_lines = to_source_list(source_cell["source"])
            ranges = split_ranges_for_cell(cell_index, source_lines)
            markdown_entries = SPLIT_MARKDOWN[cell_index]
            if len(ranges) != len(markdown_entries):
                raise ValueError(f"Split spec mismatch for cell {cell_index}")

            grouped_code_cells[cell_index] = []
            original_id = source_cell.get("id")

            for fragment_index, ((start, end), markdown_text) in enumerate(zip(ranges, markdown_entries), start=1):
                fragment_lines = source_lines[start:end]
                if not fragment_lines:
                    raise ValueError(f"Empty fragment produced for cell {cell_index}, fragment {fragment_index}")

                new_cells.append(
                    markdown_cell(
                        markdown_text,
                        make_new_id(f"vk6-md-{cell_index}-{fragment_index}", used_ids),
                    )
                )

                preserve_original_state = fragment_index == len(ranges)
                fragment_id = (
                    original_id
                    if preserve_original_state and isinstance(original_id, str) and original_id
                    else make_new_id(f"vk6-code-{cell_index}-{fragment_index}", used_ids)
                )
                fragment_cell = code_fragment(
                    source_cell,
                    fragment_lines,
                    cell_id=fragment_id,
                    preserve_original_state=preserve_original_state,
                )
                new_cells.append(fragment_cell)
                grouped_code_cells[cell_index].append(fragment_cell)
            continue

        copied_cell = copy.deepcopy(source_cell)
        new_cells.append(copied_cell)
        if cell_type == "code":
            grouped_code_cells[cell_index] = [copied_cell]

    new_nb["cells"] = new_cells
    return new_nb, grouped_code_cells


def verify(source_nb: dict, built_nb: dict, grouped_code_cells: dict[int, list[dict]]) -> dict:
    if source_nb["metadata"] != built_nb["metadata"]:
        raise AssertionError("Top-level notebook metadata changed.")

    original_code_text = []
    rebuilt_code_text = []

    split_cell_indices = set(SPLIT_MARKDOWN)

    for cell_index, source_cell in enumerate(source_nb["cells"], start=1):
        if source_cell["cell_type"] != "code":
            continue

        if cell_index not in grouped_code_cells:
            raise AssertionError(f"Missing rebuilt code group for source cell {cell_index}.")

        rebuilt_group = grouped_code_cells[cell_index]
        source_text = "".join(to_source_list(source_cell["source"]))
        rebuilt_text = "".join("".join(to_source_list(cell["source"])) for cell in rebuilt_group)

        if source_text != rebuilt_text:
            raise AssertionError(f"Code mismatch for source cell {cell_index}.")

        original_code_text.append(source_text)
        rebuilt_code_text.append(rebuilt_text)

        if cell_index in split_cell_indices:
            terminal = rebuilt_group[-1]
            if terminal.get("metadata", {}) != source_cell.get("metadata", {}):
                raise AssertionError(f"Terminal metadata mismatch for split cell {cell_index}.")
            if terminal.get("execution_count") != source_cell.get("execution_count"):
                raise AssertionError(f"Terminal execution_count mismatch for split cell {cell_index}.")
            if terminal.get("outputs", []) != source_cell.get("outputs", []):
                raise AssertionError(f"Terminal outputs mismatch for split cell {cell_index}.")
            source_id = source_cell.get("id")
            if isinstance(source_id, str) and source_id and source_id != terminal.get("id"):
                raise AssertionError(f"Terminal cell id mismatch for split cell {cell_index}.")

            for fragment in rebuilt_group[:-1]:
                if fragment.get("metadata", {}) != {}:
                    raise AssertionError(f"Non-terminal fragment metadata was not cleared for cell {cell_index}.")
                if fragment.get("execution_count") is not None:
                    raise AssertionError(f"Non-terminal fragment execution_count was not cleared for cell {cell_index}.")
                if fragment.get("outputs", []) != []:
                    raise AssertionError(f"Non-terminal fragment outputs were not cleared for cell {cell_index}.")
        else:
            if rebuilt_group[0] != source_cell:
                raise AssertionError(f"Untouched code cell {cell_index} changed.")

    if "".join(original_code_text) != "".join(rebuilt_code_text):
        raise AssertionError("Full concatenated code stream changed.")

    ids = [cell.get("id") for cell in built_nb["cells"] if isinstance(cell.get("id"), str) and cell["id"]]
    if len(ids) != len(set(ids)):
        raise AssertionError("Notebook contains duplicate cell ids.")

    return {
        "source_cells": len(source_nb["cells"]),
        "target_cells": len(built_nb["cells"]),
        "source_code_cells": sum(1 for cell in source_nb["cells"] if cell["cell_type"] == "code"),
        "target_code_cells": sum(1 for cell in built_nb["cells"] if cell["cell_type"] == "code"),
        "source_markdown_cells": sum(1 for cell in source_nb["cells"] if cell["cell_type"] == "markdown"),
        "target_markdown_cells": sum(1 for cell in built_nb["cells"] if cell["cell_type"] == "markdown"),
        "split_cells": sorted(split_cell_indices),
    }


def main() -> None:
    source_nb = load_notebook(SOURCE_NOTEBOOK)
    built_nb, grouped_code_cells = build_notebook(source_nb)
    summary = verify(source_nb, built_nb, grouped_code_cells)
    save_notebook(TARGET_NOTEBOOK, built_nb)

    print(f"Generated: {TARGET_NOTEBOOK.name}")
    print(f"Source cells: {summary['source_cells']} -> Target cells: {summary['target_cells']}")
    print(
        "Code cells: "
        f"{summary['source_code_cells']} -> {summary['target_code_cells']} | "
        f"Markdown cells: {summary['source_markdown_cells']} -> {summary['target_markdown_cells']}"
    )
    print(f"Split source cells: {summary['split_cells']}")
    print("Verification passed: code stream and top-level metadata are unchanged.")


if __name__ == "__main__":
    main()
