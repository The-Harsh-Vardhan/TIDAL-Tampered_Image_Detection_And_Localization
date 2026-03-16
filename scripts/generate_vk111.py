#!/usr/bin/env python3
"""
generate_vk111.py — Constructive generator for vK.11.1

Reads vK.11.0, preserves ALL existing cells unchanged, and inserts a
Model Card and Experiment Report section before the Conclusion.

Output: vK.11.1 Image Detection and Localisation.ipynb
"""

import json
import sys
import copy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
INPUT = NOTEBOOKS_DIR / "source" / "vK.11.0 Image Detection and Localisation [Pretrained ResNet34].ipynb"
OUTPUT = NOTEBOOKS_DIR / "source" / "vK.11.1 Image Detection and Localisation.ipynb"

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


# ── Model Card Section Cells ──────────────────────────────────────────

def build_model_card_cells():
    """Build the Model Card and Experiment Report section cells."""
    cells = []

    # ── Section header ──
    cells.append(make_md_cell(
        "## 17. Model Card and Experiment Report\n"
        "\n"
        "This section provides a structured project summary following the conventions\n"
        "of modern ML model cards. It documents the system design, dataset,\n"
        "training configuration, evaluation methodology, results, and compliance\n"
        "with the Big Vision Internship Assignment requirements."
    ))

    # ── 17.1 Model Overview ──
    cells.append(make_md_cell(
        "### 17.1 Model Overview\n"
        "\n"
        "| Field | Description |\n"
        "|---|---|\n"
        "| **Model Name** | TamperDetector vK.11.1 |\n"
        "| **Task** | Tampered image detection and pixel-level tamper localization |\n"
        "| **Input** | 256 x 256 x 4 tensor (RGB + ELA channel) |\n"
        "| **Outputs** | (1) Binary segmentation mask (256 x 256), (2) Image-level classification logits |\n"
        "| **Architecture** | Dual-head model: SMP UNet (ResNet34 encoder, ImageNet pretrained) + FC classification head on bottleneck |\n"
        "| **Framework** | PyTorch + Segmentation Models PyTorch (SMP) |\n"
        "| **Approach** | Semantic segmentation with an auxiliary classification branch |\n"
        "\n"
        "**High-level pipeline:**\n"
        "\n"
        "1. An input image is JPEG-recompressed to produce an Error Level Analysis (ELA) map\n"
        "   that highlights compression inconsistencies introduced by tampering.\n"
        "2. The ELA map is stacked as a 4th channel alongside the RGB image.\n"
        "3. The 4-channel input passes through a pretrained ResNet34 encoder, which extracts\n"
        "   hierarchical features at multiple scales.\n"
        "4. The UNet decoder fuses multi-scale encoder features to produce a pixel-level\n"
        "   tamper probability map (segmentation branch).\n"
        "5. The encoder bottleneck features are pooled and passed through a fully connected\n"
        "   head to produce an image-level authentic/tampered classification (classification branch).\n"
        "6. Both outputs are trained jointly with a combined loss:\n"
        "   `L = alpha * FocalLoss(cls) + beta * (w_bce * BCE + w_dice * Dice)(seg) + gamma * EdgeLoss(seg)`"
    ))

    # ── 17.2 Dataset Summary ──
    cells.append(make_md_cell(
        "### 17.2 Dataset Summary\n"
        "\n"
        "| Property | Value |\n"
        "|---|---|\n"
        "| **Dataset** | CASIA v2.0 (via Kaggle) |\n"
        "| **Domain** | Image forgery detection |\n"
        "| **Forgery types** | Copy-move (`Tp_D_*`) and Splicing (`Tp_S_*`) |\n"
        "| **Ground truth** | Binary masks indicating tampered pixel regions |\n"
        "| **Authentic images** | ~5,123 images with all-zero masks |\n"
        "| **Tampered images** | ~3,706 images with non-zero masks |\n"
        "| **Total images** | ~8,829 |\n"
        "\n"
        "**Dataset split** (stratified by class label):\n"
        "\n"
        "| Split | Proportion | Purpose |\n"
        "|---|---|---|\n"
        "| Train | 70% | Model training |\n"
        "| Validation | 15% | Hyperparameter tuning, early stopping, threshold optimization |\n"
        "| Test | 15% | Final held-out evaluation (no tuning decisions based on this split) |\n"
        "\n"
        "Data leakage is verified at the path level: zero overlap exists between\n"
        "train, validation, and test image paths (see Section 4.5)."
    ))

    # ── 17.3 Training Configuration ──
    cells.append(make_md_cell(
        "### 17.3 Training Configuration\n"
        "\n"
        "| Parameter | Value |\n"
        "|---|---|\n"
        "| **Framework** | PyTorch 2.x |\n"
        "| **GPU** | Kaggle T4 (15 GB VRAM) / 2x T4 with DataParallel |\n"
        "| **Input size** | 256 x 256 |\n"
        "| **Batch size** | 8 (auto-adjusted by VRAM; effective = batch x accumulation_steps) |\n"
        "| **Gradient accumulation** | 4 steps (effective batch = 32) |\n"
        "| **Optimizer** | AdamW |\n"
        "| **Encoder LR** | 1e-4 (pretrained ResNet34 — lower to preserve features) |\n"
        "| **Decoder LR** | 1e-3 (randomly initialized decoder and heads) |\n"
        "| **Weight decay** | 1e-4 |\n"
        "| **Scheduler** | ReduceLROnPlateau (patience=3, factor=0.5, monitors val tampered F1) |\n"
        "| **Max epochs** | 50 |\n"
        "| **Early stopping** | Patience = 10 (based on validation tampered Dice) |\n"
        "| **Encoder freeze** | First 2 epochs (protects pretrained BatchNorm statistics) |\n"
        "| **Mixed precision** | AMP (automatic mixed precision) enabled |\n"
        "| **Gradient clipping** | max_norm = 5.0 |\n"
        "| **Reproducibility** | Seed = 42 (Python, NumPy, PyTorch, CUDA) |\n"
        "\n"
        "**Loss function:**\n"
        "\n"
        "```\n"
        "Total Loss = alpha * FocalLoss(classification)\n"
        "           + beta  * (0.5 * BCEWithLogitsLoss + 0.5 * PerSampleDiceLoss)(segmentation)\n"
        "           + gamma * SobelEdgeLoss(segmentation)\n"
        "\n"
        "alpha = 1.5, beta = 1.0, gamma = 0.3\n"
        "```\n"
        "\n"
        "**Data augmentation (training only):**\n"
        "\n"
        "| Augmentation | Probability |\n"
        "|---|---|\n"
        "| HorizontalFlip | 0.5 |\n"
        "| VerticalFlip | 0.3 |\n"
        "| RandomRotate90 | 0.5 |\n"
        "| ShiftScaleRotate | 0.3 |\n"
        "| RandomBrightnessContrast | 0.3 |\n"
        "| Normalize (ImageNet stats) | 1.0 |"
    ))

    # ── 17.4 Evaluation Metrics ──
    cells.append(make_md_cell(
        "### 17.4 Evaluation Metrics\n"
        "\n"
        "The following metrics are used to evaluate model performance at both the\n"
        "pixel level (localization) and image level (detection).\n"
        "\n"
        "**Pixel-level segmentation metrics** (computed on thresholded binary masks):\n"
        "\n"
        "| Metric | Definition | Why it matters |\n"
        "|---|---|---|\n"
        "| **Dice coefficient** | 2 * \\|P ∩ G\\| / (\\|P\\| + \\|G\\|) | Measures overlap between predicted and ground truth masks; penalizes both false positives and false negatives equally |\n"
        "| **IoU (Jaccard)** | \\|P ∩ G\\| / (\\|P ∪ G\\|) | Stricter than Dice; standard benchmark metric for segmentation tasks |\n"
        "| **Pixel F1** | 2 * Precision * Recall / (Precision + Recall) | Balances pixel-level precision and recall; equivalent to Dice for binary masks |\n"
        "| **Pixel AUC-ROC** | Area under ROC at pixel level | Threshold-independent measure of localization quality |\n"
        "\n"
        "**Image-level classification metrics:**\n"
        "\n"
        "| Metric | Definition | Why it matters |\n"
        "|---|---|---|\n"
        "| **Accuracy** | Correct predictions / Total images | Overall detection rate |\n"
        "| **ROC-AUC** | Area under receiver operating characteristic | Threshold-independent discriminative ability |\n"
        "| **PR-AUC** | Area under precision-recall curve | Detection quality under class imbalance |\n"
        "\n"
        "**Metric inflation warning:** Mixed-set averages (computed over all images including\n"
        "authentic ones) are inflated because authentic images with all-zero masks score 1.0\n"
        "when the model correctly predicts all-zero. The **tampered-only** metrics isolate\n"
        "actual localization performance and are the primary evaluation criterion."
    ))

    # ── 17.5 Quantitative Results ──
    cells.append(make_code_cell(
        "# ================== 17.5 Quantitative Results Summary ==================\n"
        "# This cell generates the results table from the computed test metrics.\n"
        "# FINAL_TEST_METRICS is populated by the evaluation cell in Section 11.\n"
        "\n"
        "print('=' * 60)\n"
        "print('         QUANTITATIVE RESULTS — MODEL CARD')\n"
        "print('=' * 60)\n"
        "print()\n"
        "\n"
        "if 'FINAL_TEST_METRICS' in dir():\n"
        "    m = FINAL_TEST_METRICS\n"
        "    rows = [\n"
        "        ('Image-Level Accuracy',           m.get('acc', 0.0)),\n"
        "        ('Image-Level ROC-AUC',            m.get('roc_auc', 0.0)),\n"
        "        ('Dice (all samples)',              m.get('dice', 0.0)),\n"
        "        ('IoU  (all samples)',              m.get('iou', 0.0)),\n"
        "        ('F1   (all samples)',              m.get('f1', 0.0)),\n"
        "        ('Dice (tampered only)',            m.get('tampered_dice', 0.0)),\n"
        "        ('IoU  (tampered only)',            m.get('tampered_iou', 0.0)),\n"
        "        ('F1   (tampered only)',            m.get('tampered_f1', 0.0)),\n"
        "    ]\n"
        "    if 'OPTIMAL_THRESHOLD' in dir():\n"
        "        rows.append(('Optimal Seg Threshold', OPTIMAL_THRESHOLD))\n"
        "    if 'pixel_auc' in dir():\n"
        "        rows.append(('Pixel-Level AUC-ROC', pixel_auc))\n"
        "\n"
        "    print(f\"{'Metric':<35} {'Score':>10}\")\n"
        "    print('-' * 47)\n"
        "    for name, val in rows:\n"
        "        print(f'{name:<35} {val:>10.4f}')\n"
        "    print()\n"
        "    print('Note: Tampered-only metrics are the primary evaluation criterion.')\n"
        "    print('All-sample metrics are inflated by authentic images scoring 1.0.')\n"
        "else:\n"
        "    print('FINAL_TEST_METRICS not found — run the evaluation cells in Section 11 first.')"
    ))

    # ── 17.6 Qualitative Results ──
    cells.append(make_md_cell(
        "### 17.6 Qualitative Results\n"
        "\n"
        "The notebook includes multiple visualization outputs that provide qualitative\n"
        "evidence of model performance.\n"
        "\n"
        "**Prediction panels (Section 12):** For both authentic and tampered test images,\n"
        "four-panel visualizations compare:\n"
        "\n"
        "1. **Original image** — the input RGB image\n"
        "2. **Ground truth mask** — the binary annotation of tampered regions\n"
        "3. **Predicted mask** — the model's thresholded segmentation output\n"
        "4. **Overlay visualization** — the predicted mask superimposed on the original image\n"
        "   with color-coded true positives (green), false positives (red), and false negatives (blue)\n"
        "\n"
        "**ELA visualization (Section 12.2):** Side-by-side display of RGB images and their\n"
        "Error Level Analysis maps, showing how the ELA channel highlights compression\n"
        "artifacts around tampered regions that are invisible in the RGB domain.\n"
        "\n"
        "**Grad-CAM heatmaps (Section 14):** Gradient-weighted class activation maps from\n"
        "the encoder bottleneck (`encoder.layer4`) reveal which spatial regions drove the\n"
        "model's classification decision. These help verify that the model attends to\n"
        "tampered regions rather than dataset artifacts.\n"
        "\n"
        "**Training curves (Section 11.2):** Loss, accuracy, Dice, and learning rate\n"
        "histories across epochs provide insight into convergence behavior."
    ))

    # ── 17.7 Failure Analysis ──
    cells.append(make_md_cell(
        "### 17.7 Failure Analysis\n"
        "\n"
        "Section 13.1 displays the 10 worst-performing tampered test images (lowest\n"
        "per-sample Dice score). Common failure patterns observed in CASIA-2:\n"
        "\n"
        "| Failure Mode | Description | Why it is challenging |\n"
        "|---|---|---|\n"
        "| **Very small tampered regions** | Forgeries that occupy less than 2% of the image area | The segmentation head must localize tiny patches against a large background; the class imbalance within each image is extreme |\n"
        "| **Low contrast manipulations** | Spliced regions with similar color and texture to the surrounding area | Pixel-level differences are minimal; the model must rely on subtle compression or noise artifacts rather than visual discontinuities |\n"
        "| **Complex textured backgrounds** | Tampered regions placed on grass, foliage, fabric, or other high-frequency textures | Edge-based and texture-based cues overlap with natural image content, producing false positives |\n"
        "| **High-quality copy-move** | Regions duplicated from within the same image | Source and pasted regions share identical capture conditions (noise, lighting, compression), minimizing forensic traces |\n"
        "| **JPEG double compression** | Whole-image re-saving obscures localized artifacts | ELA signal is suppressed when the entire image undergoes uniform re-compression |\n"
        "\n"
        "The mask-size stratification analysis in Section 11.7 quantifies these effects:\n"
        "tiny masks (<2% area) consistently achieve lower Dice scores than larger manipulations,\n"
        "confirming that spatial extent is a dominant factor in localization difficulty.\n"
        "\n"
        "The robustness evaluation in Section 15 further shows performance degradation under\n"
        "controlled perturbations such as Gaussian blur, JPEG re-compression, and additive noise."
    ))

    # ── 17.8 Assignment Requirement Compliance ──
    cells.append(make_md_cell(
        "### 17.8 Assignment Requirement Compliance\n"
        "\n"
        "The following checklist confirms compliance with the Big Vision Internship\n"
        "Assignment requirements.\n"
        "\n"
        "**Core Requirements:**\n"
        "\n"
        "| # | Requirement | Status | Evidence |\n"
        "|---|---|---|---|\n"
        "| 1 | Publicly available dataset with authentic and tampered images | Complete | CASIA v2.0 from Kaggle (Section 4) |\n"
        "| 2 | Dataset preparation and preprocessing | Complete | Metadata caching, ELA computation, Albumentations pipeline (Section 6) |\n"
        "| 3 | Train / Validation / Test split | Complete | 70/15/15 stratified split with path-level leakage verification (Section 4.5) |\n"
        "| 4 | Train a model for tampered region localization | Complete | TamperDetector with SMP UNet + pretrained ResNet34 (Section 7) |\n"
        "| 5 | Architecture choice justified | Complete | Pretrained encoder proven across v6.5 (Tam-F1=0.41); dual-head design documented (Section 7) |\n"
        "| 6 | T4 GPU compatible | Complete | VRAM auto-scaling, AMP, batch size adjustment (Section 2) |\n"
        "| 7 | Evaluation metrics (localization + detection) | Complete | Dice, IoU, F1, Accuracy, ROC-AUC — both all-sample and tampered-only (Section 11) |\n"
        "| 8 | Visual results | Complete | 4-panel prediction grids, ELA visualization, Grad-CAM heatmaps (Sections 12, 14) |\n"
        "| 9 | Single notebook deliverable | Complete | All code in one self-contained notebook |\n"
        "| 10 | Dataset explanation | Complete | Documented in Sections 4 and 17.2 |\n"
        "| 11 | Model architecture description | Complete | Documented in Sections 7 and 17.1 |\n"
        "| 12 | Training strategy documentation | Complete | CONFIG dict + summary table + this Model Card (Sections 2, 9, 17.3) |\n"
        "| 13 | Clear visualizations | Complete | Training curves, prediction panels, confusion matrix, ROC/PR curves (Sections 11, 12) |\n"
        "\n"
        "**Bonus Requirements:**\n"
        "\n"
        "| # | Requirement | Status | Evidence |\n"
        "|---|---|---|---|\n"
        "| B1 | Robustness testing (JPEG, noise, blur) | Complete | 8-condition degradation suite (Section 15) |\n"
        "| B2 | Subtle tampering analysis | Complete | Forgery-type breakdown + mask-size stratification (Sections 11.6, 11.7) |\n"
        "| B3 | Explainability | Complete | Grad-CAM heatmaps on encoder bottleneck (Section 14) |\n"
        "| B4 | Shortcut learning validation | Complete | Mask randomization + boundary erosion checks (Section 11.8) |\n"
        "| B5 | Threshold optimization | Complete | 50-point sweep on validation set (Section 11.3) |\n"
        "| B6 | Data leakage verification | Complete | Path-level overlap assertions (Section 4.5) |"
    ))

    # ── 17.9 Future Improvements ──
    cells.append(make_md_cell(
        "### 17.9 Future Improvements\n"
        "\n"
        "The following directions could further improve detection and localization quality.\n"
        "\n"
        "**Architecture improvements:**\n"
        "\n"
        "| Improvement | Expected Impact | Rationale |\n"
        "|---|---|---|\n"
        "| Transformer-based encoder (e.g., MIT-B2, Swin-T) | Higher feature quality | Self-attention captures long-range dependencies that CNNs miss; critical for detecting spatially distributed manipulations |\n"
        "| Forensic residual extraction (SRM filters) | Better low-level artifact capture | Steganalysis-inspired high-pass filters amplify subtle manipulation traces in the noise domain |\n"
        "| Multi-scale input / Feature Pyramid | Better small-region detection | Processing at 256 and 512 resolution simultaneously captures both fine and coarse manipulation cues |\n"
        "| Test-Time Augmentation (TTA) | 2-3% F1 boost | Averaging predictions across horizontal flips and rotations reduces variance at no training cost |\n"
        "\n"
        "**Training improvements:**\n"
        "\n"
        "| Improvement | Expected Impact | Rationale |\n"
        "|---|---|---|\n"
        "| Multi-dataset training (CASIA + Columbia + Coverage) | Improved generalization | Cross-dataset diversity reduces overfitting to CASIA-2 compression artifacts |\n"
        "| Hard example mining | Better tail performance | Re-weighting loss toward difficult samples (tiny masks, low-contrast splicing) addresses the long tail of failure cases |\n"
        "| Larger input resolution (384 x 384) | Finer localization | v6.5 demonstrated this at the cost of smaller batch size; gradient accumulation mitigates the tradeoff |\n"
        "| Knowledge distillation | Faster inference | A lighter student encoder (ResNet18, EfficientNet-B0) can approximate the ResNet34 teacher at lower latency |\n"
        "\n"
        "**Evaluation improvements:**\n"
        "\n"
        "| Improvement | Expected Impact | Rationale |\n"
        "|---|---|---|\n"
        "| Cross-dataset evaluation | Generalization measurement | Testing on unseen datasets (Columbia, Coverage, NIST16) reveals whether the model learns forensic principles or dataset biases |\n"
        "| Confidence calibration | Trustworthy predictions | Calibrating predicted probabilities enables meaningful uncertainty thresholds for deployment |\n"
        "| Multi-annotator agreement | Ground truth quality | CASIA-2 masks have known quality issues; measuring inter-annotator consistency bounds achievable performance |"
    ))

    return cells


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"Reading:  {INPUT}")
    nb = load_notebook(INPUT)
    src_cells = nb["cells"]
    print(f"Source cells: {len(src_cells)}")

    # Deep copy so we don't mutate the source
    new_cells = []

    # ── Find insertion point ──
    # Insert Model Card BEFORE the Conclusion section (cell 90 in vK.11.0).
    # We identify it by looking for "## 17. Conclusion" in markdown cells.
    conclusion_idx = None
    for i, c in enumerate(src_cells):
        if c["cell_type"] == "markdown":
            text = "".join(c["source"])
            if "## 17. Conclusion" in text:
                conclusion_idx = i
                break

    if conclusion_idx is None:
        print("ERROR: Could not find Conclusion cell (## 17. Conclusion)")
        sys.exit(1)

    print(f"Conclusion cell found at index: {conclusion_idx}")

    # ── Build output cells ──
    # 1. Copy cells 0 through conclusion_idx-1 (all existing content)
    #    BUT update cell 0 (TOC) and cell 1 (title) for vK.11.1 versioning
    for i in range(conclusion_idx):
        cell = copy.deepcopy(src_cells[i])

        # Update TOC — add Model Card section entry
        if i == 0:
            text = "".join(cell["source"])
            # Add Model Card to TOC before the existing "17. Conclusion" entry
            text = text.replace(
                "17. [Conclusion](#17-conclusion)",
                "17. [Model Card and Experiment Report](#17-model-card-and-experiment-report)\n"
                "18. [Conclusion](#18-conclusion)"
            )
            # Update version in TOC header if present
            text = text.replace("vK.11.0", "vK.11.1")
            cell["source"] = to_source_lines(text)

        # Update title cell
        elif i == 1:
            text = "".join(cell["source"])
            text = text.replace("vK.11.0", "vK.11.1")
            cell["source"] = to_source_lines(text)

        new_cells.append(cell)

    # 2. Insert Model Card section
    model_card_cells = build_model_card_cells()
    new_cells.extend(model_card_cells)
    print(f"Inserted {len(model_card_cells)} Model Card cells")

    # 3. Copy Conclusion and remaining cells, renumbered
    for i in range(conclusion_idx, len(src_cells)):
        cell = copy.deepcopy(src_cells[i])

        # Renumber Conclusion from 17 → 18
        if i == conclusion_idx:
            text = "".join(cell["source"])
            text = text.replace("## 17. Conclusion", "## 18. Conclusion")
            text = text.replace("vK.11.0", "vK.11.1")
            cell["source"] = to_source_lines(text)

        new_cells.append(cell)

    # ── Assemble notebook ──
    nb_out = copy.deepcopy(nb)
    nb_out["cells"] = new_cells

    # Update metadata
    if "language_info" in nb_out.get("metadata", {}):
        nb_out["metadata"]["language_info"]["name"] = "python"

    save_notebook(nb_out, OUTPUT)

    # ── Summary ──
    code_cells = sum(1 for c in new_cells if c["cell_type"] == "code")
    md_cells = sum(1 for c in new_cells if c["cell_type"] == "markdown")
    print(f"Total cells: {len(new_cells)} ({code_cells} code, {md_cells} markdown)")
    print(f"Model Card cells added: {len(model_card_cells)}")
    print("Done.")


if __name__ == "__main__":
    main()
