#!/usr/bin/env python3
"""
generate_vk113.py — Constructive generator for vK.11.3

Reads vK.11.2, preserves ALL existing cells unchanged, and inserts a
Quick Inference Demo section before the Conclusion.

Output: vK.11.3 Image Detection and Localisation.ipynb
"""

import json
import sys
import copy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
INPUT = NOTEBOOKS_DIR / "source" / "vK.11.2 Image Detection and Localisation.ipynb"
OUTPUT = NOTEBOOKS_DIR / "source" / "vK.11.3 Image Detection and Localisation.ipynb"

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


# ── Quick Inference Demo Section Cells ─────────────────────────────────

def build_inference_demo_cells():
    """Build the Quick Inference Demo section cells."""
    cells = []

    # ── Section header ──
    cells.append(make_md_cell(
        "## 19. Quick Inference Demo\n"
        "\n"
        "This section demonstrates how to use the trained TamperDetector model\n"
        "for tamper detection on a single image. The following cells show:\n"
        "\n"
        "1. **Loading the trained model** from the best checkpoint\n"
        "2. **Selecting a sample image** from the test dataset\n"
        "3. **Running tamper detection inference** with ELA preprocessing\n"
        "4. **Visualizing the results** — original image, predicted mask, and overlay\n"
        "\n"
        "The inference pipeline reuses the `predict_single_image()` function\n"
        "defined in Section 16, which handles ELA computation, preprocessing\n"
        "transforms, and model forward pass internally."
    ))

    # ── 19.1 Load Best Model Checkpoint ──
    cells.append(make_md_cell(
        "### 19.1 Load Best Model Checkpoint\n"
        "\n"
        "Load the best model weights from the checkpoint saved during training.\n"
        "The checkpoint contains the model state at the epoch with the highest\n"
        "validation tampered Dice score. The model is set to evaluation mode\n"
        "to disable dropout and freeze batch normalization statistics."
    ))

    cells.append(make_code_cell(
        "# ================== 19.1 Load Best Checkpoint for Demo ==================\n"
        "import os\n"
        "import torch\n"
        "\n"
        "demo_ckpt_path = os.path.join(str(CHECKPOINT_DIR), 'best_model.pt')\n"
        "\n"
        "if os.path.exists(demo_ckpt_path):\n"
        "    demo_ckpt = torch.load(demo_ckpt_path, map_location=device, weights_only=False)\n"
        "    get_base_model(model).load_state_dict(demo_ckpt['model_state_dict'])\n"
        "    print(f'Model loaded from best checkpoint (epoch {demo_ckpt.get(\"best_epoch\", \"?\")}, '\n"
        "          f'Tam Dice: {demo_ckpt.get(\"best_metric\", \"?\"):.4f})')\n"
        "    del demo_ckpt  # free memory\n"
        "else:\n"
        "    print(f'WARNING: Best checkpoint not found at {demo_ckpt_path}')\n"
        "    print('Using current model weights (may not be the best checkpoint).')\n"
        "\n"
        "model.eval()\n"
        "print(f'Device: {device}')\n"
        "print('Model is in evaluation mode.')"
    ))

    # ── 19.2 Select and Load Example Image ──
    cells.append(make_md_cell(
        "### 19.2 Select and Load Example Image\n"
        "\n"
        "A tampered image is selected from the test split to demonstrate\n"
        "the inference pipeline. The image is displayed alongside its\n"
        "Error Level Analysis (ELA) map, which highlights JPEG compression\n"
        "inconsistencies in tampered regions."
    ))

    cells.append(make_code_cell(
        "# ================== 19.2 Load Example Image ==================\n"
        "import cv2\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "# Select a tampered image from the test set\n"
        "tampered_test = test_df[test_df['label'] == 1].reset_index(drop=True)\n"
        "if len(tampered_test) > 0:\n"
        "    sample_idx = min(5, len(tampered_test) - 1)  # pick 6th tampered image\n"
        "    demo_image_path = tampered_test.iloc[sample_idx]['image_path']\n"
        "    demo_mask_path  = tampered_test.iloc[sample_idx]['mask_path']\n"
        "    print(f'Selected image: {Path(demo_image_path).name}')\n"
        "    print(f'Image path:     {demo_image_path}')\n"
        "\n"
        "    # Load and display image + ELA\n"
        "    demo_bgr = cv2.imread(demo_image_path)\n"
        "    demo_rgb = cv2.cvtColor(demo_bgr, cv2.COLOR_BGR2RGB)\n"
        "    demo_ela = compute_ela(demo_bgr, quality=CONFIG['ela_quality'])\n"
        "    demo_gt  = cv2.imread(demo_mask_path, cv2.IMREAD_GRAYSCALE)\n"
        "\n"
        "    fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
        "    axes[0].imshow(demo_rgb)\n"
        "    axes[0].set_title('Original Image')\n"
        "    axes[0].axis('off')\n"
        "    axes[1].imshow(demo_ela, cmap='hot')\n"
        "    axes[1].set_title('ELA Map (Forensic Signal)')\n"
        "    axes[1].axis('off')\n"
        "    if demo_gt is not None:\n"
        "        axes[2].imshow(demo_gt, cmap='gray')\n"
        "        axes[2].set_title('Ground Truth Mask')\n"
        "    else:\n"
        "        axes[2].set_title('Ground Truth (N/A)')\n"
        "    axes[2].axis('off')\n"
        "    plt.suptitle('Demo Input: Image, ELA, and Ground Truth', fontsize=14)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No tampered images found in test set.')\n"
        "    demo_image_path = None"
    ))

    # ── 19.3 Run Inference and Visualize ──
    cells.append(make_md_cell(
        "### 19.3 Run Inference and Visualize Results\n"
        "\n"
        "The `predict_single_image()` function (Section 16) handles the complete\n"
        "inference pipeline: ELA computation, validation transforms, model forward\n"
        "pass with AMP, and mask thresholding. The results are displayed as a\n"
        "three-panel visualization:\n"
        "\n"
        "| Panel | Description |\n"
        "|---|---|\n"
        "| **Input Image** | Original RGB image |\n"
        "| **Predicted Mask** | Binary segmentation output at the optimal threshold |\n"
        "| **Overlay** | Predicted mask overlaid on the original image with semi-transparent red highlighting |"
    ))

    cells.append(make_code_cell(
        "# ================== 19.3 Inference and Visualization ==================\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import cv2\n"
        "\n"
        "if demo_image_path is not None:\n"
        "    # Run inference\n"
        "    result = predict_single_image(demo_image_path, model, device)\n"
        "\n"
        "    cls_label = 'TAMPERED' if result['is_tampered'] else 'AUTHENTIC'\n"
        "    cls_conf  = result['cls_probs'][result['is_tampered']]\n"
        "    threshold = result['threshold']\n"
        "\n"
        "    print(f'Classification: {cls_label} (confidence: {cls_conf:.3f})')\n"
        "    print(f'Segmentation threshold: {threshold:.3f}')\n"
        "    print(f'Tampered pixels: {result[\"seg_mask\"].sum()} / {result[\"seg_mask\"].size} '\n"
        "          f'({100 * result[\"seg_mask\"].mean():.2f}%)')\n"
        "\n"
        "    # Load original image for display\n"
        "    display_img = cv2.cvtColor(cv2.imread(demo_image_path), cv2.COLOR_BGR2RGB)\n"
        "    display_img = cv2.resize(display_img, (CONFIG['image_size'], CONFIG['image_size']))\n"
        "\n"
        "    # Create overlay: red highlight on predicted tampered regions\n"
        "    overlay = display_img.copy()\n"
        "    mask_rgb = np.zeros_like(overlay)\n"
        "    mask_rgb[result['seg_mask'] == 1] = [255, 0, 0]  # red\n"
        "    overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)\n"
        "\n"
        "    # Three-panel visualization\n"
        "    fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
        "\n"
        "    axes[0].imshow(display_img)\n"
        "    axes[0].set_title('Input Image', fontsize=12)\n"
        "    axes[0].axis('off')\n"
        "\n"
        "    axes[1].imshow(result['seg_mask'], cmap='gray', vmin=0, vmax=1)\n"
        "    axes[1].set_title(f'Predicted Mask (thr={threshold:.2f})', fontsize=12)\n"
        "    axes[1].axis('off')\n"
        "\n"
        "    axes[2].imshow(overlay)\n"
        "    axes[2].set_title('Overlay (red = predicted tamper)', fontsize=12)\n"
        "    axes[2].axis('off')\n"
        "\n"
        "    plt.suptitle(f'Quick Inference Demo — {cls_label} ({cls_conf:.1%} confidence)',\n"
        "                 fontsize=14, fontweight='bold')\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(os.path.join(str(RESULTS_DIR), 'quick_inference_demo.png'),\n"
        "                dpi=150, bbox_inches='tight')\n"
        "    plt.show()\n"
        "    print(f'\\nVisualization saved to {RESULTS_DIR}/quick_inference_demo.png')\n"
        "else:\n"
        "    print('Skipping inference demo — no demo image available.')"
    ))

    return cells


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"Reading:  {INPUT}")
    nb = load_notebook(INPUT)
    src_cells = nb["cells"]
    print(f"Source cells: {len(src_cells)}")

    new_cells = []

    # ── Find Conclusion cell ──
    conclusion_idx = None
    for i, c in enumerate(src_cells):
        if c["cell_type"] == "markdown":
            text = "".join(c["source"])
            if "## 19. Conclusion" in text:
                conclusion_idx = i
                break

    if conclusion_idx is None:
        print("ERROR: Could not find Conclusion cell (## 19. Conclusion)")
        sys.exit(1)

    print(f"Conclusion cell found at index: {conclusion_idx}")

    # ── Build output cells ──
    # 1. Copy cells 0 through conclusion_idx-1
    for i in range(conclusion_idx):
        cell = copy.deepcopy(src_cells[i])

        # Update TOC
        if i == 0:
            text = "".join(cell["source"])
            text = text.replace(
                "19. [Conclusion](#19-conclusion)",
                "19. [Quick Inference Demo](#19-quick-inference-demo)\n"
                "20. [Conclusion](#20-conclusion)"
            )
            text = text.replace("vK.11.2", "vK.11.3")
            cell["source"] = to_source_lines(text)

        # Update title cell
        elif i == 1:
            text = "".join(cell["source"])
            text = text.replace("vK.11.2", "vK.11.3")
            cell["source"] = to_source_lines(text)

        new_cells.append(cell)

    # 2. Insert Quick Inference Demo section
    demo_cells = build_inference_demo_cells()
    new_cells.extend(demo_cells)
    print(f"Inserted {len(demo_cells)} Quick Inference Demo cells")

    # 3. Copy Conclusion and remaining cells, renumbered
    for i in range(conclusion_idx, len(src_cells)):
        cell = copy.deepcopy(src_cells[i])

        # Renumber Conclusion from 19 → 20
        if i == conclusion_idx:
            text = "".join(cell["source"])
            text = text.replace("## 19. Conclusion", "## 20. Conclusion")
            text = text.replace("vK.11.2", "vK.11.3")
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
    print(f"Demo cells added: {len(demo_cells)}")
    print("Done.")


if __name__ == "__main__":
    main()
