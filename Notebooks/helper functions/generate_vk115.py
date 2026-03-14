#!/usr/bin/env python3
"""
generate_vk115.py — Constructive generator for vK.11.5

Reads vK.11.4, preserves ALL existing cells unchanged, and inserts a
Results Dashboard section immediately after the Project Executive Summary.

Output: vK.11.5 Image Detection and Localisation.ipynb
"""

import json
import sys
import copy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
INPUT = NOTEBOOKS_DIR / "vK.11.4 Image Detection and Localisation.ipynb"
OUTPUT = NOTEBOOKS_DIR / "vK.11.5 Image Detection and Localisation.ipynb"

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


# ── Results Dashboard Section Cells ────────────────────────────────────

def build_dashboard_cells():
    """Build the Results Dashboard section cells."""
    cells = []

    # ── D0: Section header ──
    cells.append(make_md_cell(
        "## Results Dashboard\n"
        "\n"
        "This dashboard provides a condensed overview of the trained model's\n"
        "performance. It is designed to let reviewers understand the project\n"
        "quality within a few seconds.\n"
        "\n"
        "**Note:** On a fresh top-to-bottom run, these cells display placeholder\n"
        "messages until the training and evaluation sections (Sections 10–11)\n"
        "have completed. After that, re-running just this section will show live\n"
        "results."
    ))

    # ── D1: Metrics Summary Table header ──
    cells.append(make_md_cell(
        "### Metrics Summary Table\n"
        "\n"
        "The following table shows the final held-out **test set** evaluation\n"
        "metrics. Tampered-only metrics are the primary evaluation criterion\n"
        "because they isolate the model's ability to localize actual forgeries,\n"
        "excluding the trivial contribution of authentic (all-zero-mask) images."
    ))

    # ── D2: Metrics Summary Table (code) ──
    cells.append(make_code_cell(
        "# ================== Results Dashboard: Metrics Summary ==================\n"
        "try:\n"
        "    m = FINAL_TEST_METRICS\n"
        "    print('=' * 55)\n"
        "    print('     FINAL TEST METRICS — RESULTS DASHBOARD')\n"
        "    print('=' * 55)\n"
        "    print()\n"
        "    print(f\"{'Metric':<35} {'Score':>10}\")\n"
        "    print('-' * 47)\n"
        "    print(f\"{'Image-Level Accuracy':<35} {m.get('acc', 0):>10.4f}\")\n"
        "    print(f\"{'Image-Level ROC-AUC':<35} {m.get('roc_auc', 0):>10.4f}\")\n"
        "    print('-' * 47)\n"
        "    print(f\"{'Dice (tampered only)':<35} {m.get('tampered_dice', 0):>10.4f}\")\n"
        "    print(f\"{'IoU  (tampered only)':<35} {m.get('tampered_iou', 0):>10.4f}\")\n"
        "    print(f\"{'F1   (tampered only)':<35} {m.get('tampered_f1', 0):>10.4f}\")\n"
        "    print('-' * 47)\n"
        "    print()\n"
        "    # Provide a quick pass/fail summary\n"
        "    dice = m.get('tampered_dice', 0)\n"
        "    auc  = m.get('roc_auc', 0)\n"
        "    if dice > 0.30 and auc > 0.85:\n"
        "        print('Assessment: STRONG — model produces meaningful localization.')\n"
        "    elif dice > 0.15 and auc > 0.75:\n"
        "        print('Assessment: MODERATE — model detects tampering but localization needs improvement.')\n"
        "    else:\n"
        "        print('Assessment: WEAK — model struggles with tamper localization.')\n"
        "except NameError:\n"
        "    print('Final test metrics have not been computed yet.')\n"
        "    print('Run the full notebook (Sections 1-11) to populate this dashboard.')\n"
        "    print()\n"
        "    print('Expected metrics after evaluation:')\n"
        "    print('  - Image-Level Accuracy and ROC-AUC')\n"
        "    print('  - Tampered-only Dice / IoU / F1')"
    ))

    # ── D3: Training Curve header ──
    cells.append(make_md_cell(
        "### Training Curve Visualization\n"
        "\n"
        "The plots below show how the model trained over epochs:\n"
        "\n"
        "| Panel | Description |\n"
        "|---|---|\n"
        "| **Loss** | Training and validation loss — should decrease and converge |\n"
        "| **Tampered Dice** | Validation tampered-only Dice score — the primary metric for early stopping |\n"
        "\n"
        "A vertical dashed line marks the epoch with the best validation tampered Dice\n"
        "(the checkpoint that was saved and used for final evaluation)."
    ))

    # ── D4: Training Curve Plot (code) ──
    cells.append(make_code_cell(
        "# ================== Results Dashboard: Training Curves ==================\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "try:\n"
        "    if history.get('train_loss') and len(history['train_loss']) > 0:\n"
        "        epochs_range = range(1, len(history['train_loss']) + 1)\n"
        "\n"
        "        fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "\n"
        "        # Panel 1: Loss\n"
        "        axes[0].plot(epochs_range, history['train_loss'], 'b-', lw=1.5, label='Train Loss')\n"
        "        if history.get('val_loss'):\n"
        "            axes[0].plot(epochs_range, history['val_loss'], 'r-', lw=1.5, label='Val Loss')\n"
        "        axes[0].set_xlabel('Epoch')\n"
        "        axes[0].set_ylabel('Loss')\n"
        "        axes[0].set_title('Training & Validation Loss')\n"
        "        axes[0].legend()\n"
        "        axes[0].grid(True, alpha=0.3)\n"
        "\n"
        "        # Panel 2: Tampered Dice\n"
        "        if history.get('val_tampered_dice'):\n"
        "            axes[1].plot(epochs_range, history['val_tampered_dice'], 'g-', lw=1.5,\n"
        "                         label='Val Tampered Dice')\n"
        "            best_ep = max(range(len(history['val_tampered_dice'])),\n"
        "                          key=lambda i: history['val_tampered_dice'][i])\n"
        "            best_val = history['val_tampered_dice'][best_ep]\n"
        "            axes[1].axvline(x=best_ep + 1, color='gray', ls='--', alpha=0.6,\n"
        "                            label=f'Best epoch {best_ep + 1} ({best_val:.4f})')\n"
        "            axes[1].scatter([best_ep + 1], [best_val], color='red', zorder=5, s=60)\n"
        "        axes[1].set_xlabel('Epoch')\n"
        "        axes[1].set_ylabel('Dice')\n"
        "        axes[1].set_title('Validation Tampered Dice')\n"
        "        axes[1].legend()\n"
        "        axes[1].grid(True, alpha=0.3)\n"
        "\n"
        "        plt.suptitle('Results Dashboard — Training Progress', fontsize=14, fontweight='bold')\n"
        "        plt.tight_layout()\n"
        "        plt.show()\n"
        "    else:\n"
        "        print('No training history available yet.')\n"
        "        print('Run the Training Loop (Section 10) first.')\n"
        "except NameError:\n"
        "    print('Training history not yet available.')\n"
        "    print('Run Sections 1-10 to populate the training curves.')"
    ))

    # ── D5: Example Localization header ──
    cells.append(make_md_cell(
        "### Example Localization Result\n"
        "\n"
        "The four-panel visualization below demonstrates the model's tamper\n"
        "localization capability on a representative test image:\n"
        "\n"
        "| Panel | Description |\n"
        "|---|---|\n"
        "| **Original** | The input RGB image |\n"
        "| **Ground Truth** | The binary mask marking the actual tampered region |\n"
        "| **Predicted Mask** | The model's predicted segmentation mask at the optimal threshold |\n"
        "| **Overlay** | Color-coded overlay — **green** = true positive, **red** = false positive, **blue** = false negative |"
    ))

    # ── D6: Example Localization Plot (code) ──
    cells.append(make_code_cell(
        "# ================== Results Dashboard: Example Localization ==================\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import cv2\n"
        "\n"
        "try:\n"
        "    tampered_test = test_df[test_df['label'] == 1].reset_index(drop=True)\n"
        "    _dashboard_ready = (\n"
        "        len(tampered_test) > 0\n"
        "        and 'predict_single_image' in dir()\n"
        "        and model is not None\n"
        "    )\n"
        "except NameError:\n"
        "    _dashboard_ready = False\n"
        "\n"
        "if _dashboard_ready:\n"
        "    _idx = min(3, len(tampered_test) - 1)  # 4th tampered image (different from other demos)\n"
        "    _img_path  = tampered_test.iloc[_idx]['image_path']\n"
        "    _mask_path = tampered_test.iloc[_idx]['mask_path']\n"
        "\n"
        "    _result = predict_single_image(_img_path, model, device)\n"
        "\n"
        "    _sz = CONFIG['image_size']\n"
        "    _display = cv2.cvtColor(cv2.imread(_img_path), cv2.COLOR_BGR2RGB)\n"
        "    _display = cv2.resize(_display, (_sz, _sz))\n"
        "\n"
        "    _gt_raw = cv2.imread(_mask_path, cv2.IMREAD_GRAYSCALE)\n"
        "    if _gt_raw is not None:\n"
        "        _gt = cv2.resize(_gt_raw, (_sz, _sz))\n"
        "        _gt_bin = (_gt > 127).astype(np.uint8)\n"
        "    else:\n"
        "        _gt_bin = np.zeros((_sz, _sz), dtype=np.uint8)\n"
        "\n"
        "    _pred = _result['seg_mask']\n"
        "\n"
        "    # Build color-coded overlay: green=TP, red=FP, blue=FN\n"
        "    _overlay = _display.copy()\n"
        "    _tp = (_pred == 1) & (_gt_bin == 1)\n"
        "    _fp = (_pred == 1) & (_gt_bin == 0)\n"
        "    _fn = (_pred == 0) & (_gt_bin == 1)\n"
        "    _color_mask = np.zeros_like(_overlay)\n"
        "    _color_mask[_tp] = [0, 200, 0]     # green\n"
        "    _color_mask[_fp] = [200, 0, 0]      # red\n"
        "    _color_mask[_fn] = [0, 0, 200]      # blue\n"
        "    _overlay = cv2.addWeighted(_overlay, 0.6, _color_mask, 0.4, 0)\n"
        "\n"
        "    _cls = 'TAMPERED' if _result['is_tampered'] else 'AUTHENTIC'\n"
        "    _conf = _result['cls_probs'][_result['is_tampered']]\n"
        "\n"
        "    fig, axes = plt.subplots(1, 4, figsize=(22, 5))\n"
        "    axes[0].imshow(_display)\n"
        "    axes[0].set_title('Original Image', fontsize=12)\n"
        "    axes[0].axis('off')\n"
        "\n"
        "    axes[1].imshow(_gt_bin, cmap='gray', vmin=0, vmax=1)\n"
        "    axes[1].set_title('Ground Truth Mask', fontsize=12)\n"
        "    axes[1].axis('off')\n"
        "\n"
        "    axes[2].imshow(_pred, cmap='gray', vmin=0, vmax=1)\n"
        "    axes[2].set_title(f'Predicted Mask (thr={_result[\"threshold\"]:.2f})', fontsize=12)\n"
        "    axes[2].axis('off')\n"
        "\n"
        "    axes[3].imshow(_overlay)\n"
        "    axes[3].set_title('Overlay (G=TP, R=FP, B=FN)', fontsize=12)\n"
        "    axes[3].axis('off')\n"
        "\n"
        "    plt.suptitle(\n"
        "        f'Results Dashboard — {_cls} ({_conf:.1%} confidence)',\n"
        "        fontsize=14, fontweight='bold'\n"
        "    )\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "\n"
        "    # Clean up temporary variables\n"
        "    del _idx, _img_path, _mask_path, _result, _sz, _display\n"
        "    del _gt_raw, _gt_bin, _pred, _overlay, _tp, _fp, _fn, _color_mask, _cls, _conf\n"
        "    del _dashboard_ready\n"
        "else:\n"
        "    print('Example localization not available yet.')\n"
        "    print('Run Sections 1-16 to load data, train, and define inference functions.')\n"
        "    print()\n"
        "    print('After execution, this cell will display a 4-panel visualization:')\n"
        "    print('  Original | Ground Truth | Predicted Mask | Color Overlay')"
    ))

    # ── D7: Interpretation ──
    cells.append(make_md_cell(
        "### Interpretation\n"
        "\n"
        "The Results Dashboard above shows three key aspects of the model's performance:\n"
        "\n"
        "1. **Metrics Summary** — The tampered-only Dice, IoU, and F1 scores measure how\n"
        "   accurately the model localizes forged pixels, excluding the trivial contribution\n"
        "   of authentic images. Image-level accuracy and AUC-ROC measure detection\n"
        "   (whether the image is tampered at all).\n"
        "\n"
        "2. **Training Curves** — The loss curves confirm that training converged without\n"
        "   instability. The tampered Dice curve shows when the best checkpoint was saved\n"
        "   (used for all subsequent evaluation).\n"
        "\n"
        "3. **Example Localization** — The color-coded overlay provides an intuitive quality\n"
        "   check: green pixels are correctly detected tampered regions, red pixels are false\n"
        "   alarms, and blue pixels are missed tampered regions.\n"
        "\n"
        "For detailed analysis — including per-forgery-type breakdown, mask-size stratification,\n"
        "robustness testing, Grad-CAM explainability, and failure case analysis — see\n"
        "Sections 11–15 below.\n"
        "\n"
        "---\n"
        "\n"
        "*The full implementation follows below, starting with the Table of Contents.*"
    ))

    return cells


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"Reading:  {INPUT}")
    nb = load_notebook(INPUT)
    src_cells = nb["cells"]
    print(f"Source cells: {len(src_cells)}")

    new_cells = []

    # 1. Copy Executive Summary cells (0–7), update title version
    for i in range(8):
        cell = copy.deepcopy(src_cells[i])
        if i == 0:
            text = "".join(cell["source"])
            text = text.replace("vK.11.4", "vK.11.5")
            cell["source"] = to_source_lines(text)
        new_cells.append(cell)

    print(f"Copied {8} Executive Summary cells (title updated to vK.11.5)")

    # 2. Insert Results Dashboard cells
    dashboard_cells = build_dashboard_cells()
    new_cells.extend(dashboard_cells)
    print(f"Inserted {len(dashboard_cells)} Results Dashboard cells")

    # 3. Copy TOC cell (cell 8) — add dashboard link + update version
    toc_cell = copy.deepcopy(src_cells[8])
    toc_text = "".join(toc_cell["source"])
    toc_text = toc_text.replace(
        "0. [Project Executive Summary](#project-executive-summary)\n",
        "0. [Project Executive Summary](#project-executive-summary)\n"
        "    - [Results Dashboard](#results-dashboard)\n"
    )
    toc_text = toc_text.replace("vK.11.4", "vK.11.5")
    toc_cell["source"] = to_source_lines(toc_text)
    new_cells.append(toc_cell)

    # 4. Copy remaining cells (9–end) unchanged, except Conclusion version
    for i in range(9, len(src_cells)):
        cell = copy.deepcopy(src_cells[i])
        text = "".join(cell["source"])
        if "## 20. Conclusion" in text and cell["cell_type"] == "markdown":
            text = text.replace("vK.11.4", "vK.11.5")
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
    print(f"Dashboard cells added: {len(dashboard_cells)}")
    print("Done.")


if __name__ == "__main__":
    main()
