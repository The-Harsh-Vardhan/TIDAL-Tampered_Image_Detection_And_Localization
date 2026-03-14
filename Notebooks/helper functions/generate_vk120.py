#!/usr/bin/env python3
"""
generate_vk120.py — Constructive generator for vK.12.0

Reads vK.11.5, preserves ALL existing cells unchanged, and inserts 16 new
cells at 8 insertion points + replaces 1 cell (robustness) to add:
  - Mask coverage histogram (Section 6.4 enhancement)
  - Architecture diagram + model complexity (new sections 7.1, 7.2)
  - Training strategy explanation (new section 9.5)
  - Extended localization metrics (Section 11.3.1)
  - Experiment comparison table (new section 11.9)
  - Enhanced visualizations with difference map (Section 12.3)
  - FP/FN error analysis (Section 13.2)
  - Enhanced robustness testing (Section 15 - replacement)
  - Inference speed test (Section 16.1)

Output: vK.12.0 Image Detection and Localisation.ipynb
"""

import json
import sys
import copy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
INPUT = NOTEBOOKS_DIR / "vK.11.5 Image Detection and Localisation.ipynb"
OUTPUT = NOTEBOOKS_DIR / "vK.12.0 Image Detection and Localisation.ipynb"

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


# ── Builder Functions ──────────────────────────────────────────────────

def build_mask_coverage_cells():
    """Mask coverage distribution histogram — insert after cell 50."""
    return [make_code_cell(
        "# ================== Mask Coverage Distribution (Section 6.4 Enhancement) ==================\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import cv2\n"
        "\n"
        "# Compute mask coverage for all tampered images in training set\n"
        "tampered_train = train_df[train_df['label'] == 1].reset_index(drop=True)\n"
        "coverages = []\n"
        "for idx in range(len(tampered_train)):\n"
        "    mask_path = tampered_train.iloc[idx]['mask_path']\n"
        "    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)\n"
        "    if mask is not None:\n"
        "        coverage = (mask > 127).sum() / mask.size * 100.0\n"
        "        coverages.append(coverage)\n"
        "coverages = np.array(coverages)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "\n"
        "# Histogram\n"
        "axes[0].hist(coverages, bins=50, color='steelblue', edgecolor='black', alpha=0.7)\n"
        "axes[0].set_xlabel('Mask Coverage (%)')\n"
        "axes[0].set_ylabel('Number of Images')\n"
        "axes[0].set_title('Distribution of Tampered Region Size (Train Set)')\n"
        "axes[0].axvline(np.median(coverages), color='red', linestyle='--',\n"
        "                label=f'Median: {np.median(coverages):.1f}%')\n"
        "axes[0].axvline(np.mean(coverages), color='orange', linestyle='--',\n"
        "                label=f'Mean: {np.mean(coverages):.1f}%')\n"
        "axes[0].legend()\n"
        "axes[0].grid(True, alpha=0.3)\n"
        "\n"
        "# CDF\n"
        "sorted_cov = np.sort(coverages)\n"
        "cdf = np.arange(1, len(sorted_cov) + 1) / len(sorted_cov)\n"
        "axes[1].plot(sorted_cov, cdf, 'b-', linewidth=1.5)\n"
        "axes[1].set_xlabel('Mask Coverage (%)')\n"
        "axes[1].set_ylabel('Cumulative Proportion')\n"
        "axes[1].set_title('CDF of Tampered Region Size')\n"
        "axes[1].grid(True, alpha=0.3)\n"
        "\n"
        "plt.suptitle('Mask Coverage Analysis \\u2014 How Much of Each Image Is Tampered?', fontsize=13)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print(f'Tampered images analyzed: {len(coverages)}')\n"
        "print(f'Coverage range: {coverages.min():.1f}% \\u2014 {coverages.max():.1f}%')\n"
        "print(f'Mean coverage:  {coverages.mean():.1f}%')\n"
        "print(f'Median coverage: {np.median(coverages):.1f}%')\n"
        "print(f'Images with <2% coverage: {(coverages < 2).sum()} ({100*(coverages < 2).mean():.1f}%)')\n"
        "print(f'Images with >15% coverage: {(coverages > 15).sum()} ({100*(coverages > 15).mean():.1f}%)')\n"
        "\n"
        "if WANDB_ACTIVE:\n"
        "    wandb.log({'data/mask_coverage_histogram': wandb.Image(fig)})"
    )]


def build_architecture_diagram_cells():
    """Architecture diagram — insert after cell 53 (part 1)."""
    cells = []

    cells.append(make_md_cell(
        "### 7.1 Architecture Diagram\n"
        "\n"
        "The following diagram illustrates the TamperDetector data flow.\n"
        "The 4-channel input (RGB + ELA) is processed by a pretrained ResNet34\n"
        "encoder, then split into two heads: a UNet decoder for pixel-level\n"
        "segmentation and a classification head for image-level detection."
    ))

    cells.append(make_code_cell(
        "# ================== Architecture Diagram (Matplotlib) ==================\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.patches as mpatches\n"
        "\n"
        "fig, ax = plt.subplots(1, 1, figsize=(12, 7))\n"
        "ax.set_xlim(0, 12)\n"
        "ax.set_ylim(0, 10)\n"
        "ax.axis('off')\n"
        "\n"
        "box_kw = dict(boxstyle='round,pad=0.4', facecolor='#E8F0FE', edgecolor='#4285F4', linewidth=2)\n"
        "head_kw = dict(boxstyle='round,pad=0.4', facecolor='#E6F4EA', edgecolor='#34A853', linewidth=2)\n"
        "out_kw = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#FB8C00', linewidth=2)\n"
        "arrow_kw = dict(arrowstyle='->', color='#333333', linewidth=2)\n"
        "\n"
        "# Input\n"
        "ax.text(6, 9.2, 'Input (256x256x4)\\nRGB + ELA Channel', ha='center', va='center',\n"
        "        fontsize=11, fontweight='bold', bbox=box_kw)\n"
        "\n"
        "# Encoder\n"
        "ax.annotate('', xy=(6, 7.8), xytext=(6, 8.6), arrowprops=arrow_kw)\n"
        "ax.text(6, 7.2, 'ResNet34 Encoder\\n(ImageNet Pretrained)', ha='center', va='center',\n"
        "        fontsize=11, fontweight='bold', bbox=box_kw)\n"
        "\n"
        "# Split arrows\n"
        "ax.annotate('', xy=(3.5, 5.6), xytext=(6, 6.6), arrowprops=arrow_kw)\n"
        "ax.annotate('', xy=(8.5, 5.6), xytext=(6, 6.6), arrowprops=arrow_kw)\n"
        "\n"
        "# UNet Decoder branch\n"
        "ax.text(3.5, 5.0, 'UNet Decoder\\n(Skip Connections)', ha='center', va='center',\n"
        "        fontsize=10, fontweight='bold', bbox=head_kw)\n"
        "ax.annotate('', xy=(3.5, 3.6), xytext=(3.5, 4.4), arrowprops=arrow_kw)\n"
        "ax.text(3.5, 3.0, 'Segmentation Mask\\n(256x256x1)', ha='center', va='center',\n"
        "        fontsize=10, fontweight='bold', bbox=out_kw)\n"
        "\n"
        "# Classification branch\n"
        "ax.text(8.5, 5.0, 'Classification Head\\n(GAP + FC Layers)', ha='center', va='center',\n"
        "        fontsize=10, fontweight='bold', bbox=head_kw)\n"
        "ax.annotate('', xy=(8.5, 3.6), xytext=(8.5, 4.4), arrowprops=arrow_kw)\n"
        "ax.text(8.5, 3.0, 'Class Label\\n(Authentic / Tampered)', ha='center', va='center',\n"
        "        fontsize=10, fontweight='bold', bbox=out_kw)\n"
        "\n"
        "# Loss labels\n"
        "ax.text(3.5, 1.8, 'Focal + Dice + Edge Loss', ha='center', va='center',\n"
        "        fontsize=9, fontstyle='italic', color='#666666')\n"
        "ax.text(8.5, 1.8, 'BCE Loss', ha='center', va='center',\n"
        "        fontsize=9, fontstyle='italic', color='#666666')\n"
        "\n"
        "ax.set_title('TamperDetector Architecture \\u2014 Dual-Head Design', fontsize=14, fontweight='bold', pad=15)\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(str(PLOTS_DIR), 'architecture_diagram.png'), dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "if WANDB_ACTIVE:\n"
        "    wandb.log({'model/architecture_diagram': wandb.Image(fig)})"
    ))
    return cells


def build_model_complexity_cells():
    """Model complexity analysis — insert after cell 53 (part 2, after arch diagram)."""
    cells = []

    cells.append(make_md_cell(
        "### 7.2 Model Complexity Analysis\n"
        "\n"
        "Quantitative analysis of model size, parameter counts, and estimated\n"
        "memory usage. Uses `torchinfo` for a detailed layer-by-layer summary."
    ))

    cells.append(make_code_cell(
        "# ================== Model Complexity Analysis ==================\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "try:\n"
        "    from torchinfo import summary\n"
        "except ImportError:\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torchinfo'])\n"
        "    from torchinfo import summary\n"
        "\n"
        "# Detailed summary\n"
        "base_model = get_base_model(model)\n"
        "model_summary = summary(\n"
        "    base_model,\n"
        "    input_size=(1, CONFIG.get('in_channels', 4), CONFIG['image_size'], CONFIG['image_size']),\n"
        "    col_names=['input_size', 'output_size', 'num_params', 'trainable'],\n"
        "    depth=3,\n"
        "    verbose=0,\n"
        ")\n"
        "print(model_summary)\n"
        "\n"
        "# Manual parameter counting\n"
        "total_params = sum(p.numel() for p in base_model.parameters())\n"
        "trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)\n"
        "frozen_params = total_params - trainable_params\n"
        "\n"
        "# Estimate model size (assuming float32)\n"
        "model_size_mb = total_params * 4 / (1024 ** 2)\n"
        "\n"
        "# Estimate VRAM usage (model + gradients + optimizer states ~ 4x model size for AdamW)\n"
        "est_vram_training_mb = model_size_mb * 4  # model + gradients + 2 optimizer states\n"
        "batch_activation_mb = CONFIG['batch_size'] * CONFIG['image_size']**2 * 4 * 64 / (1024**2)  # rough estimate\n"
        "\n"
        "print()\n"
        "print('=' * 55)\n"
        "print('     MODEL COMPLEXITY SUMMARY')\n"
        "print('=' * 55)\n"
        "print(f\"{'Total parameters':<35} {total_params:>15,}\")\n"
        "print(f\"{'Trainable parameters':<35} {trainable_params:>15,}\")\n"
        "print(f\"{'Frozen parameters':<35} {frozen_params:>15,}\")\n"
        "print(f\"{'Model size (FP32)':<35} {model_size_mb:>14.1f} MB\")\n"
        "print(f\"{'Est. training VRAM (model only)':<35} {est_vram_training_mb:>14.1f} MB\")\n"
        "print()\n"
        "\n"
        "if WANDB_ACTIVE:\n"
        "    wandb.log({\n"
        "        'model/total_params': total_params,\n"
        "        'model/trainable_params': trainable_params,\n"
        "        'model/model_size_mb': model_size_mb,\n"
        "    })"
    ))
    return cells


def build_training_strategy_cells():
    """Training strategy rationale — insert after cell 61."""
    return [make_md_cell(
        "### 9.5 Training Strategy Rationale\n"
        "\n"
        "This section explains the reasoning behind key training decisions.\n"
        "\n"
        "**Architecture Choice (ResNet34 + UNet via SMP):**\n"
        "ResNet34 provides a strong feature extraction backbone pretrained on ImageNet,\n"
        "offering robust low-level edge and texture features critical for detecting\n"
        "manipulation artifacts. The UNet decoder with skip connections preserves\n"
        "spatial resolution needed for precise mask boundaries. SMP provides a\n"
        "production-quality implementation with proper weight initialization.\n"
        "\n"
        "**Loss Function Design (Focal + BCE + Dice + Edge):**\n"
        "- *Focal Loss (segmentation)*: Addresses class imbalance at the pixel level \\u2014\n"
        "  most pixels are authentic even in tampered images.\n"
        "- *BCE Loss (classification)*: Standard binary cross-entropy for the image-level\n"
        "  detection head, with class weights inversely proportional to frequency.\n"
        "- *Dice Loss (segmentation)*: Directly optimizes the overlap metric, complementing\n"
        "  focal loss which optimizes pixel-wise accuracy.\n"
        "- *Edge Loss (Sobel)*: Encourages sharp mask boundaries by penalizing edge\n"
        "  prediction errors, computed via Sobel filters on predicted vs ground truth masks.\n"
        "\n"
        "**Augmentation Strategy:**\n"
        "The augmentation pipeline (Section 6.1) applies geometric transforms\n"
        "(flips, affine) and photometric transforms (brightness/contrast, noise,\n"
        "JPEG compression) to improve generalization. Crucially, augmentations are\n"
        "applied consistently to both image and mask to maintain spatial correspondence.\n"
        "\n"
        "**Optimizer and Scheduler:**\n"
        "- *AdamW* with differential learning rates: encoder at 1e-4 (fine-tuning\n"
        "  pretrained weights gently) and decoder at 1e-3 (training from scratch faster).\n"
        "- *ReduceLROnPlateau* scheduler monitors validation tampered F1 with patience=3,\n"
        "  reducing LR by 0.5x when the metric stalls.\n"
        "\n"
        "**Batch Size and Gradient Accumulation:**\n"
        "Physical batch size of 8 fits in a single T4 GPU (15 GB VRAM) with AMP enabled.\n"
        "Gradient accumulation over 4 steps yields an effective batch size of 32, which\n"
        "provides more stable gradient estimates without exceeding memory limits.\n"
        "\n"
        "**Encoder Freeze Warmup:**\n"
        "The encoder is frozen for the first 2 epochs to protect pretrained BatchNorm\n"
        "statistics from being destroyed by randomly initialized decoder gradients.\n"
        "This is critical for transfer learning stability."
    )]


def build_extended_metrics_cells():
    """Extended localization metrics — insert after cell 73."""
    cells = []

    cells.append(make_md_cell(
        "### 11.3.1 Extended Localization Metrics\n"
        "\n"
        "Beyond Dice, IoU, and F1, the following additional metrics provide a more\n"
        "complete picture of segmentation quality:\n"
        "\n"
        "| Metric | Description |\n"
        "|---|---|\n"
        "| **Precision** | Of all pixels predicted as tampered, how many actually are |\n"
        "| **Recall** | Of all truly tampered pixels, how many were detected |\n"
        "| **Pixel Accuracy** | Overall pixel-level accuracy (TP + TN) / total |\n"
        "\n"
        "These are computed on **tampered images only** to avoid metric inflation from\n"
        "authentic images (which have all-zero masks)."
    ))

    cells.append(make_code_cell(
        "# ================== Extended Localization Metrics ==================\n"
        "import torch\n"
        "\n"
        "thr = OPTIMAL_THRESHOLD if 'OPTIMAL_THRESHOLD' in dir() else CONFIG['seg_threshold']\n"
        "tam_idx = (test_preds['labels'] == 1).nonzero(as_tuple=True)[0]\n"
        "seg_probs = torch.sigmoid(test_preds['seg_logits'][tam_idx])\n"
        "pred_bin = (seg_probs > thr).float()\n"
        "gt = test_preds['masks'][tam_idx]\n"
        "eps = 1e-7\n"
        "\n"
        "tp = (pred_bin * gt).sum(dim=(1,2,3))\n"
        "fp = (pred_bin * (1 - gt)).sum(dim=(1,2,3))\n"
        "fn = ((1 - pred_bin) * gt).sum(dim=(1,2,3))\n"
        "tn = ((1 - pred_bin) * (1 - gt)).sum(dim=(1,2,3))\n"
        "\n"
        "precision = ((tp + eps) / (tp + fp + eps)).mean().item()\n"
        "recall = ((tp + eps) / (tp + fn + eps)).mean().item()\n"
        "pixel_accuracy = ((tp + tn) / (tp + tn + fp + fn + eps)).mean().item()\n"
        "\n"
        "# Also retrieve existing metrics\n"
        "try:\n"
        "    m = FINAL_TEST_METRICS\n"
        "    dice = m.get('tampered_dice', 0)\n"
        "    iou = m.get('tampered_iou', 0)\n"
        "    f1 = m.get('tampered_f1', 0)\n"
        "except NameError:\n"
        "    dice = test_opt.get('tampered_dice', 0)\n"
        "    iou = test_opt.get('tampered_iou', 0)\n"
        "    f1 = test_opt.get('tampered_f1', 0)\n"
        "\n"
        "print('=' * 60)\n"
        "print('     COMPLETE LOCALIZATION METRICS (Tampered-Only, Test Set)')\n"
        "print('=' * 60)\n"
        "print()\n"
        "print(f\"{'Metric':<35} {'Score':>10}\")\n"
        "print('-' * 47)\n"
        "print(f\"{'Dice Coefficient':<35} {dice:>10.4f}\")\n"
        "print(f\"{'IoU (Jaccard Index)':<35} {iou:>10.4f}\")\n"
        "print(f\"{'F1 Score':<35} {f1:>10.4f}\")\n"
        "print(f\"{'Precision':<35} {precision:>10.4f}\")\n"
        "print(f\"{'Recall':<35} {recall:>10.4f}\")\n"
        "print(f\"{'Pixel Accuracy':<35} {pixel_accuracy:>10.4f}\")\n"
        "print('-' * 47)\n"
        "print(f\"{'Threshold used':<35} {thr:>10.4f}\")\n"
        "print(f\"{'Tampered test images':<35} {len(tam_idx):>10}\")\n"
        "print()\n"
        "\n"
        "# Interpretation\n"
        "if precision > recall + 0.1:\n"
        "    print('Note: Precision >> Recall \\u2014 model is conservative (few false positives, may miss tampering).')\n"
        "elif recall > precision + 0.1:\n"
        "    print('Note: Recall >> Precision \\u2014 model is aggressive (catches most tampering, but with some false positives).')\n"
        "else:\n"
        "    print('Note: Precision and Recall are balanced.')\n"
        "\n"
        "if WANDB_ACTIVE:\n"
        "    wandb.log({\n"
        "        'evaluation/ext_precision': precision,\n"
        "        'evaluation/ext_recall': recall,\n"
        "        'evaluation/ext_pixel_accuracy': pixel_accuracy,\n"
        "    })"
    ))
    return cells


def build_experiment_comparison_cells():
    """Experiment version comparison — insert after cell 83."""
    cells = []

    cells.append(make_md_cell(
        "### 11.9 Experiment Version Comparison\n"
        "\n"
        "The following table compares key metrics across major experiment versions.\n"
        "This contextualizes vK.12.0 within the broader experiment history and\n"
        "shows the progression of improvements.\n"
        "\n"
        "| Version | Architecture | Tam Dice | Tam F1 | Accuracy | AUC | Key Change |\n"
        "|---|---|---|---|---|---|---|\n"
        "| v7 | Custom UNet (scratch) | ~0.05 | ~0.05 | ~0.58 | ~0.60 | Baseline custom encoder |\n"
        "| v10 | SMP UNet + ResNet34 | ~0.20 | ~0.22 | ~0.80 | ~0.85 | Pretrained encoder, ELA |\n"
        "| v11.x | SMP UNet + ResNet34 | ~0.35 | ~0.38 | ~0.85 | ~0.90 | Edge loss, grad accum, freeze |\n"
        "| **v12.0** | SMP UNet + ResNet34 | *live* | *live* | *live* | *live* | Extended eval, robustness, complexity |\n"
        "\n"
        "*Note: Historic values are approximate. vK.12.0 values are computed live below.*"
    ))

    cells.append(make_code_cell(
        "# ================== Experiment Comparison (Live Metrics) ==================\n"
        "try:\n"
        "    m = FINAL_TEST_METRICS\n"
        "    print('=' * 65)\n"
        "    print('     vK.12.0 METRICS vs HISTORIC BASELINES')\n"
        "    print('=' * 65)\n"
        "    print()\n"
        "    print(f\"{'Version':<10} {'Tam Dice':>10} {'Tam F1':>10} {'Accuracy':>10} {'AUC':>10}\")\n"
        "    print('-' * 55)\n"
        "    print(f\"{'v7':<10} {'~0.050':>10} {'~0.050':>10} {'~0.580':>10} {'~0.600':>10}\")\n"
        "    print(f\"{'v10':<10} {'~0.200':>10} {'~0.220':>10} {'~0.800':>10} {'~0.850':>10}\")\n"
        "    print(f\"{'v11.x':<10} {'~0.350':>10} {'~0.380':>10} {'~0.850':>10} {'~0.900':>10}\")\n"
        "    print(f\"{'v12.0':<10} {m.get('tampered_dice', 0):>10.4f} {m.get('tampered_f1', 0):>10.4f} \"\n"
        "          f\"{m.get('acc', 0):>10.4f} {m.get('roc_auc', 0):>10.4f}\")\n"
        "    print('-' * 55)\n"
        "    print()\n"
        "    # Compute improvement over v11.x baseline\n"
        "    v11_dice = 0.35\n"
        "    delta = m.get('tampered_dice', 0) - v11_dice\n"
        "    print(f'Delta from v11.x baseline (Tam Dice): {delta:+.4f}')\n"
        "except NameError:\n"
        "    print('Live metrics not yet available. Run Sections 10-11 first.')\n"
        "\n"
        "if WANDB_ACTIVE:\n"
        "    try:\n"
        "        comparison_table = wandb.Table(\n"
        "            columns=['Version', 'Tam_Dice', 'Tam_F1', 'Accuracy', 'AUC'],\n"
        "            data=[\n"
        "                ['v7', 0.05, 0.05, 0.58, 0.60],\n"
        "                ['v10', 0.20, 0.22, 0.80, 0.85],\n"
        "                ['v11.x', 0.35, 0.38, 0.85, 0.90],\n"
        "                ['v12.0', m.get('tampered_dice', 0), m.get('tampered_f1', 0),\n"
        "                 m.get('acc', 0), m.get('roc_auc', 0)],\n"
        "            ]\n"
        "        )\n"
        "        wandb.log({'evaluation/experiment_comparison': comparison_table})\n"
        "    except Exception:\n"
        "        pass"
    ))
    return cells


def build_enhanced_viz_cells():
    """Enhanced 6-panel visualization — insert after cell 96."""
    cells = []

    cells.append(make_md_cell(
        "### 12.3 Difference Map and Contour Overlay\n"
        "\n"
        "This visualization adds two additional panels beyond the standard 4-panel view:\n"
        "\n"
        "| Panel | Description |\n"
        "|---|---|\n"
        "| **Difference Map** | Absolute difference between predicted and ground-truth masks \\u2014 highlights disagreement regions |\n"
        "| **Contour Overlay** | Ground truth contour (green) and predicted contour (red) overlaid on the original image |"
    ))

    cells.append(make_code_cell(
        "# ================== Enhanced Visualization: Difference Map + Contour Overlay ==================\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import cv2\n"
        "\n"
        "def show_enhanced_viz(samples, title, max_items=4):\n"
        "    \"\"\"6-panel visualization: Original, GT, Predicted, Overlay, Diff Map, Contour.\"\"\"\n"
        "    items = [s for s in samples if s.get('label', 0) == 1][:max_items]\n"
        "    if not items:\n"
        "        items = samples[:max_items]\n"
        "    n = len(items)\n"
        "    if n == 0:\n"
        "        print('No samples to visualize.')\n"
        "        return None\n"
        "\n"
        "    fig, axes = plt.subplots(n, 6, figsize=(24, 4 * n))\n"
        "    if n == 1:\n"
        "        axes = axes[np.newaxis, :]\n"
        "\n"
        "    for i, s in enumerate(items):\n"
        "        img = denormalize(s['image'])\n"
        "        gt_mask = s['true_mask'].squeeze().numpy() if hasattr(s['true_mask'], 'numpy') else s['true_mask'].squeeze()\n"
        "        gt_bin = (gt_mask > 0.5).astype(np.uint8)\n"
        "\n"
        "        seg_logit = s.get('seg_logit', s.get('seg_logits', None))\n"
        "        if seg_logit is not None:\n"
        "            import torch\n"
        "            if hasattr(seg_logit, 'sigmoid'):\n"
        "                pred_prob = torch.sigmoid(seg_logit).squeeze().numpy()\n"
        "            else:\n"
        "                pred_prob = seg_logit.squeeze()\n"
        "        else:\n"
        "            pred_prob = s.get('pred_mask', np.zeros_like(gt_bin)).squeeze()\n"
        "            if hasattr(pred_prob, 'numpy'):\n"
        "                pred_prob = pred_prob.numpy()\n"
        "\n"
        "        thr = OPTIMAL_THRESHOLD if 'OPTIMAL_THRESHOLD' in dir() else CONFIG['seg_threshold']\n"
        "        pred_bin = (pred_prob > thr).astype(np.uint8)\n"
        "\n"
        "        # Difference map\n"
        "        diff_map = np.abs(pred_bin.astype(float) - gt_bin.astype(float))\n"
        "\n"
        "        # Contour overlay\n"
        "        img_display = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)\n"
        "        contour_img = img_display.copy()\n"
        "        gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n"
        "        pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n"
        "        cv2.drawContours(contour_img, gt_contours, -1, (0, 255, 0), 2)  # green = GT\n"
        "        cv2.drawContours(contour_img, pred_contours, -1, (255, 0, 0), 2)  # red = pred\n"
        "\n"
        "        # Standard overlay (TP=green, FP=red, FN=blue)\n"
        "        overlay = img_display.copy()\n"
        "        color_mask = np.zeros_like(overlay)\n"
        "        color_mask[(pred_bin == 1) & (gt_bin == 1)] = [0, 200, 0]\n"
        "        color_mask[(pred_bin == 1) & (gt_bin == 0)] = [200, 0, 0]\n"
        "        color_mask[(pred_bin == 0) & (gt_bin == 1)] = [0, 0, 200]\n"
        "        overlay = cv2.addWeighted(overlay, 0.6, color_mask, 0.4, 0)\n"
        "\n"
        "        panels = [\n"
        "            (img, 'Original'),\n"
        "            (gt_bin, 'Ground Truth'),\n"
        "            (pred_bin, 'Predicted'),\n"
        "            (overlay, 'Overlay (G=TP,R=FP,B=FN)'),\n"
        "            (diff_map, 'Difference Map'),\n"
        "            (contour_img, 'Contour (G=GT, R=Pred)'),\n"
        "        ]\n"
        "        for j, (data, ttl) in enumerate(panels):\n"
        "            ax = axes[i, j]\n"
        "            if data.ndim == 2:\n"
        "                ax.imshow(data, cmap='hot' if 'Diff' in ttl else 'gray', vmin=0, vmax=1)\n"
        "            else:\n"
        "                ax.imshow(data)\n"
        "            ax.set_title(ttl, fontsize=9)\n"
        "            ax.axis('off')\n"
        "\n"
        "    plt.suptitle(title, fontsize=14, fontweight='bold')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    return fig\n"
        "\n"
        "enhanced_fig = show_enhanced_viz(fake_samples, 'Enhanced Visualization: 6-Panel Analysis (Tampered)', max_items=4)\n"
        "\n"
        "if WANDB_ACTIVE and enhanced_fig is not None:\n"
        "    wandb.log({'visualization/enhanced_6panel': wandb.Image(enhanced_fig)})"
    ))
    return cells


def build_fpfn_analysis_cells():
    """FP/FN error analysis — insert after cell 98."""
    cells = []

    cells.append(make_md_cell(
        "### 13.2 False Positive / False Negative Analysis\n"
        "\n"
        "Beyond the worst-Dice analysis above, this section separately examines:\n"
        "\n"
        "- **False Positives (FP):** Authentic images incorrectly classified as tampered\n"
        "- **False Negatives (FN):** Tampered images incorrectly classified as authentic\n"
        "\n"
        "This helps identify systematic failure patterns \\u2014 for example, whether\n"
        "certain image types consistently fool the classifier."
    ))

    cells.append(make_code_cell(
        "# ================== FP/FN Error Analysis ==================\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import torch\n"
        "import cv2\n"
        "\n"
        "thr_cls = 0.5  # classification threshold\n"
        "cls_probs = torch.softmax(test_preds['cls_logits'], dim=1)\n"
        "pred_labels = (cls_probs[:, 1] > thr_cls).long()\n"
        "true_labels = test_preds['labels']\n"
        "\n"
        "# False Positives: authentic (label=0) predicted as tampered (pred=1)\n"
        "fp_mask = (true_labels == 0) & (pred_labels == 1)\n"
        "fp_indices = fp_mask.nonzero(as_tuple=True)[0]\n"
        "\n"
        "# False Negatives: tampered (label=1) predicted as authentic (pred=0)\n"
        "fn_mask = (true_labels == 1) & (pred_labels == 0)\n"
        "fn_indices = fn_mask.nonzero(as_tuple=True)[0]\n"
        "\n"
        "print(f'Classification Error Summary (test set):')\n"
        "print(f'  Total test images: {len(true_labels)}')\n"
        "print(f'  False Positives (authentic -> tampered): {len(fp_indices)}')\n"
        "print(f'  False Negatives (tampered -> authentic): {len(fn_indices)}')\n"
        "print()\n"
        "\n"
        "# Show up to 5 FP examples\n"
        "n_show = min(5, len(fp_indices))\n"
        "if n_show > 0:\n"
        "    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3.5 * n_show))\n"
        "    if n_show == 1:\n"
        "        axes = axes[np.newaxis, :]\n"
        "    fig.suptitle('False Positives: Authentic Images Predicted as Tampered', fontsize=14, y=1.01)\n"
        "    for row in range(n_show):\n"
        "        gi = fp_indices[row].item()\n"
        "        img_path = test_df.iloc[gi]['image_path']\n"
        "        conf = cls_probs[gi, 1].item()\n"
        "        orig = cv2.imread(img_path)\n"
        "        if orig is not None:\n"
        "            orig = cv2.cvtColor(cv2.resize(orig, (256, 256)), cv2.COLOR_BGR2RGB)\n"
        "        else:\n"
        "            orig = np.zeros((256, 256, 3), dtype=np.uint8)\n"
        "        seg_prob = torch.sigmoid(test_preds['seg_logits'][gi, 0]).numpy()\n"
        "        seg_mask = (seg_prob > OPTIMAL_THRESHOLD).astype(np.uint8)\n"
        "        axes[row, 0].imshow(orig)\n"
        "        axes[row, 0].set_title(f'Authentic Image (conf={conf:.2f})', fontsize=9)\n"
        "        axes[row, 1].imshow(seg_prob, cmap='hot', vmin=0, vmax=1)\n"
        "        axes[row, 1].set_title('Predicted Seg Prob', fontsize=9)\n"
        "        axes[row, 2].imshow(seg_mask, cmap='gray', vmin=0, vmax=1)\n"
        "        axes[row, 2].set_title('Predicted Mask (should be blank)', fontsize=9)\n"
        "        for ax in axes[row]:\n"
        "            ax.axis('off')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    if WANDB_ACTIVE:\n"
        "        wandb.log({'evaluation/false_positives': wandb.Image(fig)})\n"
        "else:\n"
        "    print('No false positives found \\u2014 all authentic images correctly classified.')\n"
        "\n"
        "# Show up to 5 FN examples\n"
        "n_show_fn = min(5, len(fn_indices))\n"
        "if n_show_fn > 0:\n"
        "    fig2, axes2 = plt.subplots(n_show_fn, 4, figsize=(16, 3.5 * n_show_fn))\n"
        "    if n_show_fn == 1:\n"
        "        axes2 = axes2[np.newaxis, :]\n"
        "    fig2.suptitle('False Negatives: Tampered Images Predicted as Authentic', fontsize=14, y=1.01)\n"
        "    for row in range(n_show_fn):\n"
        "        gi = fn_indices[row].item()\n"
        "        img_path = test_df.iloc[gi]['image_path']\n"
        "        fname = os.path.basename(img_path)\n"
        "        ftype = 'copy-move' if fname.startswith('Tp_D') else 'splicing' if fname.startswith('Tp_S') else 'unknown'\n"
        "        conf = cls_probs[gi, 1].item()\n"
        "        orig = cv2.imread(img_path)\n"
        "        if orig is not None:\n"
        "            orig = cv2.cvtColor(cv2.resize(orig, (256, 256)), cv2.COLOR_BGR2RGB)\n"
        "        else:\n"
        "            orig = np.zeros((256, 256, 3), dtype=np.uint8)\n"
        "        gt = test_preds['masks'][gi, 0].numpy()\n"
        "        seg_prob = torch.sigmoid(test_preds['seg_logits'][gi, 0]).numpy()\n"
        "        axes2[row, 0].imshow(orig)\n"
        "        axes2[row, 0].set_title(f'{ftype} (conf={conf:.2f})', fontsize=9)\n"
        "        axes2[row, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)\n"
        "        axes2[row, 1].set_title('Ground Truth', fontsize=9)\n"
        "        axes2[row, 2].imshow(seg_prob, cmap='hot', vmin=0, vmax=1)\n"
        "        axes2[row, 2].set_title('Predicted Seg Prob', fontsize=9)\n"
        "        mask_area = gt.sum() / gt.size * 100\n"
        "        axes2[row, 3].imshow((seg_prob > OPTIMAL_THRESHOLD).astype(np.uint8), cmap='gray', vmin=0, vmax=1)\n"
        "        axes2[row, 3].set_title(f'Pred Mask (GT area={mask_area:.1f}%)', fontsize=9)\n"
        "        for ax in axes2[row]:\n"
        "            ax.axis('off')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    if WANDB_ACTIVE:\n"
        "        wandb.log({'evaluation/false_negatives': wandb.Image(fig2)})\n"
        "else:\n"
        "    print('No false negatives found \\u2014 all tampered images correctly classified.')\n"
        "\n"
        "# Summary commentary\n"
        "print()\n"
        "print('Common Failure Patterns:')\n"
        "if len(fp_indices) > 0:\n"
        "    print(f'  FP: {len(fp_indices)} authentic images triggered false tamper detection.')\n"
        "    print('       Possible causes: JPEG artifacts, complex textures, or high-frequency noise.')\n"
        "if len(fn_indices) > 0:\n"
        "    # Check if FNs tend to have small masks\n"
        "    fn_mask_areas = []\n"
        "    for gi in fn_indices:\n"
        "        area = test_preds['masks'][gi.item()].sum().item() / test_preds['masks'][gi.item()].numel() * 100\n"
        "        fn_mask_areas.append(area)\n"
        "    fn_mask_areas = np.array(fn_mask_areas)\n"
        "    print(f'  FN: {len(fn_indices)} tampered images were missed by the classifier.')\n"
        "    print(f'       Mean tampered area: {fn_mask_areas.mean():.1f}% (small tampering is harder to detect).')"
    ))
    return cells


def build_enhanced_robustness_cell():
    """Enhanced robustness cell — replaces cell 102."""
    return make_code_cell(
        "# ================== Enhanced Robustness Testing Suite ==================\n"
        "NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))\n"
        "\n"
        "robustness_transforms = {\n"
        "    'clean': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), NORMALIZE, ToTensorV2()],\n"
        "                       additional_targets={'ela': 'image'}),\n"
        "    'jpeg_qf70': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n"
        "                            A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),\n"
        "                            NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),\n"
        "    'jpeg_qf50': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n"
        "                            A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),\n"
        "                            NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),\n"
        "    'noise_s10': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n"
        "                            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),\n"
        "                            NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),\n"
        "    'noise_s25': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n"
        "                            A.GaussNoise(var_limit=(100.0, 100.0), p=1.0),\n"
        "                            NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),\n"
        "    'blur_k3': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n"
        "                          A.GaussianBlur(blur_limit=(3, 3), p=1.0),\n"
        "                          NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),\n"
        "    'blur_k5': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n"
        "                          A.GaussianBlur(blur_limit=(5, 5), p=1.0),\n"
        "                          NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),\n"
        "    'resize_0.75': A.Compose([A.Resize(int(IMAGE_SIZE*0.75), int(IMAGE_SIZE*0.75)),\n"
        "                              A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n"
        "                              NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),\n"
        "}\n"
        "\n"
        "@torch.no_grad()\n"
        "def run_robustness_eval_enhanced(condition_name, transform, threshold):\n"
        "    \"\"\"Evaluate model under a degradation condition \\u2014 returns Dice, IoU, F1.\"\"\"\n"
        "    model.eval()\n"
        "    robust_dataset = ELAImageMaskDataset(\n"
        "        test_df[test_df['label'] == 1],\n"
        "        transform=transform,\n"
        "        ela_quality=CONFIG['ela_quality'],\n"
        "    )\n"
        "    if len(robust_dataset) == 0:\n"
        "        return {'dice': 0.0, 'iou': 0.0, 'f1': 0.0}\n"
        "    robust_loader = DataLoader(robust_dataset, batch_size=CONFIG['batch_size'],\n"
        "                               shuffle=False, num_workers=CONFIG['num_workers'],\n"
        "                               pin_memory=True)\n"
        "    all_preds, all_masks = [], []\n"
        "    for images, masks, labels in robust_loader:\n"
        "        images = images.to(device)\n"
        "        with autocast('cuda', enabled=CONFIG['use_amp']):\n"
        "            _, seg_logits = model(images)\n"
        "        pred_bin = (torch.sigmoid(seg_logits).cpu() > threshold).float()\n"
        "        all_preds.append(pred_bin)\n"
        "        all_masks.append(masks)\n"
        "    preds = torch.cat(all_preds)\n"
        "    masks_cat = torch.cat(all_masks)\n"
        "    eps = 1e-7\n"
        "    tp = (preds * masks_cat).sum(dim=(1,2,3))\n"
        "    fp = (preds * (1 - masks_cat)).sum(dim=(1,2,3))\n"
        "    fn = ((1 - preds) * masks_cat).sum(dim=(1,2,3))\n"
        "    inter = tp\n"
        "    union = preds.sum(dim=(1,2,3)) + masks_cat.sum(dim=(1,2,3))\n"
        "    dice = ((2 * inter + eps) / (union + eps)).mean().item()\n"
        "    iou = ((inter + eps) / (union - inter + eps)).mean().item()\n"
        "    prec = (tp + eps) / (tp + fp + eps)\n"
        "    rec = (tp + eps) / (tp + fn + eps)\n"
        "    f1 = ((2 * prec * rec) / (prec + rec + eps)).mean().item()\n"
        "    return {'dice': dice, 'iou': iou, 'f1': f1}\n"
        "\n"
        "threshold = OPTIMAL_THRESHOLD if 'OPTIMAL_THRESHOLD' in dir() else CONFIG['seg_threshold']\n"
        "print(f'Enhanced robustness evaluation using threshold={threshold:.4f}')\n"
        "print()\n"
        "\n"
        "robustness_results = {}\n"
        "for name, transform in tqdm(robustness_transforms.items(), desc='Robustness tests'):\n"
        "    metrics = run_robustness_eval_enhanced(name, transform, threshold)\n"
        "    robustness_results[name] = metrics\n"
        "    print(f'  {name:20s}: Dice={metrics[\"dice\"]:.4f}  IoU={metrics[\"iou\"]:.4f}  F1={metrics[\"f1\"]:.4f}')\n"
        "\n"
        "# Summary table\n"
        "clean = robustness_results.get('clean', {'dice': 0, 'iou': 0, 'f1': 0})\n"
        "print(f'\\nDeltas from clean baseline:')\n"
        "print(f\"{'Condition':<20} {'dDice':>8} {'dIoU':>8} {'dF1':>8}\")\n"
        "print('-' * 46)\n"
        "for name, m in robustness_results.items():\n"
        "    if name != 'clean':\n"
        "        print(f\"{name:<20} {m['dice']-clean['dice']:>+8.4f} {m['iou']-clean['iou']:>+8.4f} {m['f1']-clean['f1']:>+8.4f}\")\n"
        "\n"
        "# Bar chart with all three metrics\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "names = list(robustness_results.keys())\n"
        "for ax_idx, metric_name in enumerate(['dice', 'iou', 'f1']):\n"
        "    vals = [robustness_results[n][metric_name] for n in names]\n"
        "    colors = ['green' if n == 'clean' else 'steelblue' for n in names]\n"
        "    axes[ax_idx].bar(names, vals, color=colors)\n"
        "    axes[ax_idx].axhline(clean[metric_name], color='green', linestyle='--', alpha=0.5,\n"
        "                         label=f'Clean = {clean[metric_name]:.4f}')\n"
        "    axes[ax_idx].set_ylabel(f'Tampered-Only {metric_name.upper()}')\n"
        "    axes[ax_idx].set_title(f'Robustness: {metric_name.upper()}')\n"
        "    axes[ax_idx].legend()\n"
        "    axes[ax_idx].set_xticklabels(names, rotation=30, ha='right')\n"
        "plt.suptitle('Enhanced Robustness Testing: Dice / IoU / F1 Under Degradation', fontsize=14)\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(PLOTS_DIR, 'robustness_results_enhanced.png'), dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "# Degradation visualization: side-by-side original vs degraded\n"
        "sample_row = test_df[test_df['label'] == 1].iloc[0]\n"
        "sample_img = cv2.cvtColor(cv2.imread(sample_row['image_path']), cv2.COLOR_BGR2RGB)\n"
        "sample_img = cv2.resize(sample_img, (IMAGE_SIZE, IMAGE_SIZE))\n"
        "demo_conditions = ['clean', 'jpeg_qf50', 'noise_s25', 'blur_k5', 'resize_0.75']\n"
        "fig2, axes2 = plt.subplots(1, len(demo_conditions), figsize=(4 * len(demo_conditions), 4))\n"
        "for j, cond in enumerate(demo_conditions):\n"
        "    tf = robustness_transforms[cond]\n"
        "    ela_raw = compute_ela(cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR), quality=CONFIG['ela_quality'])\n"
        "    ela_3ch = np.stack([ela_raw]*3, axis=-1)\n"
        "    aug = tf(image=sample_img, ela=ela_3ch)\n"
        "    # Reverse normalization for display\n"
        "    disp = aug['image'][:3].permute(1,2,0).numpy()\n"
        "    disp = disp * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])\n"
        "    disp = np.clip(disp, 0, 1)\n"
        "    axes2[j].imshow(disp)\n"
        "    axes2[j].set_title(cond, fontsize=10)\n"
        "    axes2[j].axis('off')\n"
        "fig2.suptitle('Degradation Visualization: Same Image Under Different Conditions', fontsize=13)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "if WANDB_ACTIVE:\n"
        "    wandb.log({'evaluation/robustness_enhanced': wandb.Image(fig)})\n"
        "    wandb.log({'evaluation/degradation_visualization': wandb.Image(fig2)})\n"
        "    try:\n"
        "        _rob_table = wandb.Table(\n"
        "            columns=['Condition', 'Dice', 'IoU', 'F1', 'dDice', 'dIoU', 'dF1'],\n"
        "            data=[[n, m['dice'], m['iou'], m['f1'],\n"
        "                   m['dice']-clean['dice'], m['iou']-clean['iou'], m['f1']-clean['f1']]\n"
        "                  for n, m in robustness_results.items()]\n"
        "        )\n"
        "        wandb.log({'evaluation/robustness_table_enhanced': _rob_table})\n"
        "    except Exception:\n"
        "        pass"
    )


def build_inference_speed_cells():
    """Inference speed benchmark — insert after cell 104."""
    cells = []

    cells.append(make_md_cell(
        "### 16.1 Inference Speed Benchmark\n"
        "\n"
        "Measures inference latency and throughput on test images.\n"
        "Uses CUDA events for accurate GPU timing (avoids CPU-GPU sync overhead in\n"
        "wall-clock measurements)."
    ))

    cells.append(make_code_cell(
        "# ================== Inference Speed Benchmark ==================\n"
        "import time\n"
        "import torch\n"
        "import numpy as np\n"
        "\n"
        "N_BENCH = min(50, len(test_df))\n"
        "bench_paths = test_df['image_path'].tolist()[:N_BENCH]\n"
        "model.eval()\n"
        "\n"
        "def benchmark_inference(use_amp=True, n_warmup=5):\n"
        "    \"\"\"Benchmark inference speed with optional AMP.\"\"\"\n"
        "    latencies = []\n"
        "\n"
        "    # Warmup\n"
        "    for i in range(min(n_warmup, len(bench_paths))):\n"
        "        _ = predict_single_image(bench_paths[i], model, device)\n"
        "\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.synchronize()\n"
        "\n"
        "    for path in bench_paths:\n"
        "        if torch.cuda.is_available():\n"
        "            start_event = torch.cuda.Event(enable_timing=True)\n"
        "            end_event = torch.cuda.Event(enable_timing=True)\n"
        "            start_event.record()\n"
        "            _ = predict_single_image(path, model, device)\n"
        "            end_event.record()\n"
        "            torch.cuda.synchronize()\n"
        "            latencies.append(start_event.elapsed_time(end_event))  # ms\n"
        "        else:\n"
        "            t0 = time.perf_counter()\n"
        "            _ = predict_single_image(path, model, device)\n"
        "            latencies.append((time.perf_counter() - t0) * 1000)  # ms\n"
        "\n"
        "    latencies = np.array(latencies)\n"
        "    return {\n"
        "        'mean_ms': latencies.mean(),\n"
        "        'std_ms': latencies.std(),\n"
        "        'median_ms': np.median(latencies),\n"
        "        'min_ms': latencies.min(),\n"
        "        'max_ms': latencies.max(),\n"
        "        'fps': 1000.0 / latencies.mean() if latencies.mean() > 0 else 0,\n"
        "        'n_images': len(latencies),\n"
        "    }\n"
        "\n"
        "# Benchmark with AMP (default)\n"
        "print(f'Benchmarking inference on {N_BENCH} test images...')\n"
        "print(f'Device: {device}')\n"
        "print()\n"
        "\n"
        "results_amp = benchmark_inference(use_amp=True)\n"
        "\n"
        "print('=' * 55)\n"
        "print('     INFERENCE SPEED BENCHMARK')\n"
        "print('=' * 55)\n"
        "print()\n"
        "print(f\"{'Metric':<35} {'Value':>15}\")\n"
        "print('-' * 52)\n"
        "print(f\"{'Images benchmarked':<35} {results_amp['n_images']:>15}\")\n"
        "print(f\"{'Mean latency':<35} {results_amp['mean_ms']:>12.1f} ms\")\n"
        "print(f\"{'Std latency':<35} {results_amp['std_ms']:>12.1f} ms\")\n"
        "print(f\"{'Median latency':<35} {results_amp['median_ms']:>12.1f} ms\")\n"
        "print(f\"{'Min latency':<35} {results_amp['min_ms']:>12.1f} ms\")\n"
        "print(f\"{'Max latency':<35} {results_amp['max_ms']:>12.1f} ms\")\n"
        "print(f\"{'Throughput (FPS)':<35} {results_amp['fps']:>12.1f} fps\")\n"
        "print('-' * 52)\n"
        "print()\n"
        "print('Note: Latency includes ELA computation, preprocessing, and model forward pass.')\n"
        "print('Batch size = 1 (single-image inference pipeline).')\n"
        "\n"
        "if WANDB_ACTIVE:\n"
        "    wandb.log({\n"
        "        'inference/mean_latency_ms': results_amp['mean_ms'],\n"
        "        'inference/std_latency_ms': results_amp['std_ms'],\n"
        "        'inference/fps': results_amp['fps'],\n"
        "    })"
    ))
    return cells


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print(f"Reading:  {INPUT}")
    nb = load_notebook(INPUT)
    src_cells = nb["cells"]
    print(f"Source cells: {len(src_cells)}")

    new_cells = []
    cells_inserted = 0

    for i in range(len(src_cells)):
        cell = copy.deepcopy(src_cells[i])
        text = "".join(cell["source"])

        # ── Title cell (cell 0): update version ──
        if i == 0:
            text = text.replace("vK.11.5", "vK.12.0")
            cell["source"] = to_source_lines(text)

        # ── TOC cell (cell 16): add new subsection entries + update version ──
        elif i == 16:
            text = text.replace("vK.11.5", "vK.12.0")
            # Add new subsections to TOC
            text = text.replace(
                "7. [Model Architecture](#7-model-architecture)\n",
                "7. [Model Architecture](#7-model-architecture)\n"
                "    - [7.1 Architecture Diagram](#71-architecture-diagram)\n"
                "    - [7.2 Model Complexity Analysis](#72-model-complexity-analysis)\n"
            )
            text = text.replace(
                "9. [Training Utilities](#9-training-utilities)\n",
                "9. [Training Utilities](#9-training-utilities)\n"
                "    - [9.5 Training Strategy Rationale](#95-training-strategy-rationale)\n"
            )
            text = text.replace(
                "11. [Evaluation](#11-evaluation) (includes threshold sweep, pixel AUC, confusion matrix, forgery-type, mask-size, shortcut checks)\n",
                "11. [Evaluation](#11-evaluation) (includes threshold sweep, pixel AUC, confusion matrix, forgery-type, mask-size, shortcut checks)\n"
                "    - [11.3.1 Extended Localization Metrics](#1131-extended-localization-metrics)\n"
                "    - [11.9 Experiment Version Comparison](#119-experiment-version-comparison)\n"
            )
            text = text.replace(
                "12. [Visualization of Predictions](#12-visualization-of-predictions) (includes ELA visualization)\n",
                "12. [Visualization of Predictions](#12-visualization-of-predictions) (includes ELA visualization)\n"
                "    - [12.3 Difference Map and Contour Overlay](#123-difference-map-and-contour-overlay)\n"
            )
            text = text.replace(
                "13. [Advanced Analysis](#13-advanced-analysis) (failure cases)\n",
                "13. [Advanced Analysis](#13-advanced-analysis) (failure cases)\n"
                "    - [13.2 FP/FN Error Analysis](#132-fpfn-error-analysis)\n"
            )
            text = text.replace(
                "16. [Inference Examples](#16-inference-examples)\n",
                "16. [Inference Examples](#16-inference-examples)\n"
                "    - [16.1 Inference Speed Benchmark](#161-inference-speed-benchmark)\n"
            )
            cell["source"] = to_source_lines(text)

        # ── W&B init (cell 55): update version strings ──
        elif i == 55:
            text = text.replace("vK.11.5", "vK.12.0")
            text = text.replace("vk11.5", "vk12.0")
            cell["source"] = to_source_lines(text)

        # ── Robustness cell (cell 102): REPLACE with enhanced version ──
        elif i == 102:
            cell = build_enhanced_robustness_cell()

        # ── Conclusion cell (cell 133): update version ──
        elif i == 133 and cell["cell_type"] == "markdown" and "## 20. Conclusion" in text:
            text = text.replace("vK.11.5", "vK.12.0")
            cell["source"] = to_source_lines(text)

        new_cells.append(cell)

        # ── INSERTIONS (after specific cells) ──

        # After cell 50: Mask coverage histogram
        if i == 50:
            added = build_mask_coverage_cells()
            new_cells.extend(added)
            cells_inserted += len(added)
            print(f"  Inserted {len(added)} cell(s): Mask Coverage Histogram (after cell 50)")

        # After cell 53: Architecture diagram + Model complexity
        elif i == 53:
            added1 = build_architecture_diagram_cells()
            added2 = build_model_complexity_cells()
            new_cells.extend(added1)
            new_cells.extend(added2)
            cells_inserted += len(added1) + len(added2)
            print(f"  Inserted {len(added1)+len(added2)} cell(s): Arch Diagram + Model Complexity (after cell 53)")

        # After cell 61: Training strategy rationale
        elif i == 61:
            added = build_training_strategy_cells()
            new_cells.extend(added)
            cells_inserted += len(added)
            print(f"  Inserted {len(added)} cell(s): Training Strategy Rationale (after cell 61)")

        # After cell 73: Extended localization metrics
        elif i == 73:
            added = build_extended_metrics_cells()
            new_cells.extend(added)
            cells_inserted += len(added)
            print(f"  Inserted {len(added)} cell(s): Extended Localization Metrics (after cell 73)")

        # After cell 83: Experiment comparison
        elif i == 83:
            added = build_experiment_comparison_cells()
            new_cells.extend(added)
            cells_inserted += len(added)
            print(f"  Inserted {len(added)} cell(s): Experiment Comparison (after cell 83)")

        # After cell 96: Enhanced visualizations
        elif i == 96:
            added = build_enhanced_viz_cells()
            new_cells.extend(added)
            cells_inserted += len(added)
            print(f"  Inserted {len(added)} cell(s): Enhanced 6-Panel Viz (after cell 96)")

        # After cell 98: FP/FN analysis
        elif i == 98:
            added = build_fpfn_analysis_cells()
            new_cells.extend(added)
            cells_inserted += len(added)
            print(f"  Inserted {len(added)} cell(s): FP/FN Error Analysis (after cell 98)")

        # After cell 104: Inference speed
        elif i == 104:
            added = build_inference_speed_cells()
            new_cells.extend(added)
            cells_inserted += len(added)
            print(f"  Inserted {len(added)} cell(s): Inference Speed Benchmark (after cell 104)")

    # ── Assemble notebook ──
    nb_out = copy.deepcopy(nb)
    nb_out["cells"] = new_cells

    if "language_info" in nb_out.get("metadata", {}):
        nb_out["metadata"]["language_info"]["name"] = "python"

    save_notebook(nb_out, OUTPUT)

    # ── Summary ──
    code_cells = sum(1 for c in new_cells if c["cell_type"] == "code")
    md_cells = sum(1 for c in new_cells if c["cell_type"] == "markdown")
    print(f"\nTotal cells: {len(new_cells)} ({code_cells} code, {md_cells} markdown)")
    print(f"New cells inserted: {cells_inserted}")
    print(f"Cells replaced: 1 (robustness)")
    print(f"Expected: {135 + cells_inserted} cells")
    print("Done.")


if __name__ == "__main__":
    main()
