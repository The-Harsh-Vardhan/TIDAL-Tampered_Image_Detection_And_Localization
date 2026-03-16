#!/usr/bin/env python3
"""
generate_vk114.py — Constructive generator for vK.11.4

Reads vK.11.3, preserves ALL existing cells unchanged, and prepends a
Project Executive Summary section at the very top of the notebook.

Output: vK.11.4 Image Detection and Localisation.ipynb
"""

import json
import sys
import copy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
INPUT = NOTEBOOKS_DIR / "source" / "vK.11.3 Image Detection and Localisation.ipynb"
OUTPUT = NOTEBOOKS_DIR / "source" / "vK.11.4 Image Detection and Localisation.ipynb"

# ── Helpers ─────────────────────────────────────────────────────────────

def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Saved: {path}  ({path.stat().st_size:,} bytes)")


def to_source_lines(text):
    """Convert multiline string to list of lines ending with \\n (Jupyter format)."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def make_md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source_lines(text),
    }


def make_code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_lines(text),
    }


# ── Executive Summary Section Cells ───────────────────────────────────

def build_executive_summary_cells():
    """Build the Project Executive Summary section cells."""
    cells = []

    # ── Title + summary header ──
    cells.append(make_md_cell(
        "# Tampered Image Detection and Localization (vK.11.4)\n"
        "\n"
        "**Big Vision Internship Assignment** — Image Forgery Detection and Localization"
    ))

    # ── Project Executive Summary header ──
    cells.append(make_md_cell(
        "## Project Executive Summary\n"
        "\n"
        "This section provides a high-level overview of the project for reviewers.\n"
        "It covers the problem definition, dataset, model, training strategy, and\n"
        "key results. Detailed implementation follows in the sections below."
    ))

    # ── Problem Statement ──
    cells.append(make_md_cell(
        "### Problem Statement\n"
        "\n"
        "Image tampering — including copy-move forgery and region splicing — is\n"
        "increasingly difficult to detect by visual inspection alone. This project\n"
        "develops a deep learning system that:\n"
        "\n"
        "1. **Detects** whether an image has been tampered with (image-level binary classification)\n"
        "2. **Localizes** the exact tampered pixels by producing a binary segmentation mask (pixel-level localization)\n"
        "\n"
        "The system uses a dual-head architecture that jointly performs both tasks,\n"
        "combining semantic segmentation with image-level classification in a single\n"
        "forward pass.\n"
        "\n"
        "This work is developed for the **Big Vision Internship Assignment** and is\n"
        "the result of iterating across 13 experiment runs spanning two architecture\n"
        "tracks (custom UNet from scratch and pretrained encoder via SMP)."
    ))

    # ── Dataset Overview ──
    cells.append(make_md_cell(
        "### Dataset Overview\n"
        "\n"
        "| Property | Value |\n"
        "|---|---|\n"
        "| **Dataset** | CASIA v2.0 (publicly available via Kaggle) |\n"
        "| **Authentic images** | ~5,123 (unmanipulated, all-zero masks) |\n"
        "| **Tampered images** | ~3,706 (with binary ground truth masks) |\n"
        "| **Total samples** | ~8,829 |\n"
        "| **Forgery types** | Copy-move (`Tp_D_*`) and Splicing (`Tp_S_*`) |\n"
        "| **Ground truth** | Binary masks marking tampered pixel regions |\n"
        "\n"
        "**Dataset split** (stratified by class label to preserve class ratios):\n"
        "\n"
        "| Split | Proportion | Purpose |\n"
        "|---|---|---|\n"
        "| Train | 70% (~6,180) | Model training with augmentation |\n"
        "| Validation | 15% (~1,324) | Early stopping, threshold tuning, LR scheduling |\n"
        "| Test | 15% (~1,325) | Final held-out evaluation (no tuning on this split) |\n"
        "\n"
        "Data leakage between splits is verified at the image path level (Section 4.5)."
    ))

    # ── Model Architecture Overview ──
    cells.append(make_md_cell(
        "### Model Architecture Overview\n"
        "\n"
        "The model (`TamperDetector`) is a dual-head encoder-decoder architecture:\n"
        "\n"
        "```\n"
        "Input (256x256x4: RGB + ELA)\n"
        "        |\n"
        "   ResNet34 Encoder (pretrained on ImageNet)\n"
        "        |\n"
        "   +----+----+\n"
        "   |         |\n"
        "   v         v\n"
        "UNet       Classification\n"
        "Decoder    Head (FC layers)\n"
        "   |         |\n"
        "   v         v\n"
        "Segmentation   Authentic/Tampered\n"
        "Mask (256x256) Label (2 classes)\n"
        "```\n"
        "\n"
        "**Key design choices:**\n"
        "\n"
        "| Feature | Choice | Rationale |\n"
        "|---|---|---|\n"
        "| Encoder | ResNet34 (ImageNet pretrained) | Transfer learning provides robust low/mid-level features; proven Tam-F1=0.41 in v6.5 |\n"
        "| Decoder | UNet (via SMP library) | Skip connections preserve spatial detail critical for precise mask boundaries |\n"
        "| Input | 4 channels (RGB + ELA) | Error Level Analysis highlights JPEG compression inconsistencies from tampering |\n"
        "| Dual heads | Segmentation + Classification | Joint training enables pixel-level localization and image-level detection |\n"
        "| Loss | Focal + BCE + Dice + Edge (Sobel) | Multi-term loss balances classification, segmentation overlap, and boundary precision |"
    ))

    # ── Training Strategy ──
    cells.append(make_md_cell(
        "### Training Strategy\n"
        "\n"
        "| Component | Configuration |\n"
        "|---|---|\n"
        "| **Preprocessing** | ELA computation (JPEG QF=90), resize to 256x256, ImageNet normalization |\n"
        "| **Augmentation** | HorizontalFlip, VerticalFlip, BrightnessContrast, GaussNoise, ImageCompression, Affine |\n"
        "| **Optimizer** | AdamW with differential LR (encoder: 1e-4, decoder: 1e-3) |\n"
        "| **Scheduler** | ReduceLROnPlateau (patience=3, monitors val tampered F1) |\n"
        "| **Batch size** | 8 per GPU, effective 32 via gradient accumulation (4 steps) |\n"
        "| **Mixed precision** | AMP enabled for faster training and reduced VRAM |\n"
        "| **Encoder freeze** | First 2 epochs frozen to protect pretrained BatchNorm statistics |\n"
        "| **Early stopping** | Patience=10 on validation tampered Dice |\n"
        "| **Checkpointing** | Best model, last epoch, and periodic snapshots |\n"
        "| **GPU** | Kaggle T4 (15 GB) / 2x T4 with DataParallel |\n"
        "| **Reproducibility** | Seed=42 across Python, NumPy, PyTorch, CUDA, cuDNN |"
    ))

    # ── Final Performance Metrics ──
    cells.append(make_code_cell(
        "# ================== Executive Summary: Final Performance Metrics ==================\n"
        "# This cell displays the final test metrics if they have been computed.\n"
        "# On first run (top-to-bottom), this will show a placeholder message.\n"
        "# After training + evaluation completes, re-run this cell to see results.\n"
        "\n"
        "try:\n"
        "    m = FINAL_TEST_METRICS\n"
        "    print('=' * 55)\n"
        "    print('     FINAL TEST PERFORMANCE — EXECUTIVE SUMMARY')\n"
        "    print('=' * 55)\n"
        "    print()\n"
        "    print(f\"{'Metric':<35} {'Score':>10}\")\n"
        "    print('-' * 47)\n"
        "    print(f\"{'Image-Level Accuracy':<35} {m.get('acc', 0):.4f}\")\n"
        "    print(f\"{'Image-Level ROC-AUC':<35} {m.get('roc_auc', 0):.4f}\")\n"
        "    print(f\"{'Dice (tampered only)':<35} {m.get('tampered_dice', 0):.4f}\")\n"
        "    print(f\"{'IoU  (tampered only)':<35} {m.get('tampered_iou', 0):.4f}\")\n"
        "    print(f\"{'F1   (tampered only)':<35} {m.get('tampered_f1', 0):.4f}\")\n"
        "    print('-' * 47)\n"
        "    print()\n"
        "    print('Note: Tampered-only metrics are the primary evaluation criterion.')\n"
        "    print('See Section 11 for full evaluation details.')\n"
        "except NameError:\n"
        "    print('Final test metrics have not been computed yet.')\n"
        "    print('Run the full notebook (Sections 1-11) to see results here.')\n"
        "    print()\n"
        "    print('Expected metrics (from Section 11 after training):')\n"
        "    print('  - Image-Level Accuracy')\n"
        "    print('  - Image-Level ROC-AUC')\n"
        "    print('  - Dice / IoU / F1 (tampered-only)')"
    ))

    # ── Example Localization Result ──
    cells.append(make_md_cell(
        "### Example Localization Result\n"
        "\n"
        "After training, the model produces pixel-level tamper localization masks.\n"
        "The visualization sections of this notebook provide multiple qualitative\n"
        "comparisons:\n"
        "\n"
        "| Visualization | Section | Description |\n"
        "|---|---|---|\n"
        "| Prediction panels | Section 12 | Side-by-side: Original, Ground Truth, Predicted Mask, Overlay |\n"
        "| ELA visualization | Section 12.2 | RGB image alongside its ELA forensic signal map |\n"
        "| Failure cases | Section 13 | 10 worst-performing tampered images with analysis |\n"
        "| Grad-CAM heatmaps | Section 14 | Encoder attention maps showing which regions drive classification |\n"
        "| Quick inference demo | Section 19 | End-to-end single-image inference with 3-panel result |\n"
        "\n"
        "The overlay visualizations use color coding: **green** = true positive (correctly detected),\n"
        "**red** = false positive (incorrectly flagged), **blue** = false negative (missed tampered region).\n"
        "\n"
        "---\n"
        "\n"
        "*The full implementation follows below, starting with environment setup.*"
    ))

    return cells


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"Reading:  {INPUT}")
    nb = load_notebook(INPUT)
    src_cells = nb["cells"]
    print(f"Source cells: {len(src_cells)}")

    new_cells = []

    # 1. Insert Executive Summary cells at the top
    summary_cells = build_executive_summary_cells()
    new_cells.extend(summary_cells)
    print(f"Inserted {len(summary_cells)} Executive Summary cells at top")

    # 2. Copy the TOC cell (old cell 0) — updated with summary link + version
    toc_cell = copy.deepcopy(src_cells[0])
    toc_text = "".join(toc_cell["source"])
    # Add Executive Summary to top of TOC
    toc_text = toc_text.replace(
        "# Table of Contents\n",
        "# Table of Contents\n"
        "\n"
        "0. [Project Executive Summary](#project-executive-summary)\n"
    )
    toc_text = toc_text.replace("vK.11.3", "vK.11.4")
    toc_cell["source"] = to_source_lines(toc_text)
    new_cells.append(toc_cell)

    # 3. Skip old cell 0 (TOC, already handled) and old cell 1 (old title, replaced by summary title)
    #    Copy old cell 2 onward (Project Objectives, Section 1, etc.) unchanged
    for i in range(2, len(src_cells)):
        cell = copy.deepcopy(src_cells[i])

        # Update Conclusion version string
        text = "".join(cell["source"])
        if "## 20. Conclusion" in text and cell["cell_type"] == "markdown":
            text = text.replace("vK.11.3", "vK.11.4")
            cell["source"] = to_source_lines(text)

        new_cells.append(cell)

    # ── Assemble notebook ──
    nb_out = copy.deepcopy(nb)
    nb_out["cells"] = new_cells

    if "language_info" in nb_out.get("metadata", {}):
        nb_out["metadata"]["language_info"]["name"] = "python"

    save_notebook(nb_out, OUTPUT)

    # ── Summary ──
    code_cells = sum(1 for c in new_cells if c["cell_type"] == "code")
    md_cells = sum(1 for c in new_cells if c["cell_type"] == "markdown")
    print(f"Total cells: {len(new_cells)} ({code_cells} code, {md_cells} markdown)")
    print(f"Summary cells added: {len(summary_cells)}")
    print(f"Old title cell (vK.11.3) replaced by new Executive Summary title")
    print("Done.")


if __name__ == "__main__":
    main()
