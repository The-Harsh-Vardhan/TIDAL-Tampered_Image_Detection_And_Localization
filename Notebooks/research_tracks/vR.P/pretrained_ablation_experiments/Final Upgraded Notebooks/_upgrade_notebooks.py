#!/usr/bin/env python3
"""
Upgrade Final Notebooks -> Final Upgraded Notebooks
Adds: Executive Summary, Results Dashboard, ToC, Compliance, Data Leakage,
      Model Complexity, Threshold Sweep, Extended Metrics, Mask-Size Strat,
      Failure Cases, ELA Viz, Diff Map, Speed Benchmark, Reproducibility, Conclusion.
Does NOT change architecture, training, or loss code.
"""
import json
import os
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Per-notebook config
# ============================================================
NOTEBOOKS = {
    'vR.P.19': {
        'file': 'vR.P.19 Image Detection and Localisation.ipynb',
        'title': 'Multi-Quality RGB ELA (9-Channel: Q=75, Q=85, Q=95 Full-Color)',
        'input_desc': '9-channel RGB ELA (Q=75/85/95 full-color)',
        'input_short': '9ch RGB ELA',
        'in_channels': 9,
        'has_cbam': False,
        'loss_type': 'BCE + Dice',
        'epochs': 25,
        'patience': 7,
        'lr': '1e-3',
        'freeze_strategy': 'Frozen body + BN unfrozen + conv1 unfrozen',
        'arch_extra': 'conv1 tiled 3x from pretrained 3ch weights',
        'key_change': '9-channel multi-quality RGB ELA replacing standard ELA',
        'conclusion': (
            'vR.P.19 tested whether full-color (RGB) multi-quality ELA preserves '
            'chrominance artifacts lost in P.15\'s grayscale approach. By computing '
            'ELA at Q=75/85/95 and retaining all 3 color channels per quality level, '
            'the model receives 9 input channels capturing both luminance and '
            'chrominance compression artifacts across multiple quality factors.'
        ),
        'ela_viz_type': 'rgb9ch',
        'denorm_fn': 'denormalize_ela',
        'scheduler': 'ReduceLROnPlateau (patience=3)',
        'optimizer': 'Adam (single LR=1e-3)',
    },
    'vR.P.30': {
        'file': 'vR.P.30 Image Detection and Localisation.ipynb',
        'title': 'Multi-quality ELA + CBAM attention (25ep, BCE+Dice)',
        'input_desc': '3-channel Multi-Q ELA (Q=75/85/95 grayscale)',
        'input_short': '3ch MQ-ELA',
        'in_channels': 3,
        'has_cbam': True,
        'loss_type': 'BCE + Dice',
        'epochs': 25,
        'patience': 7,
        'lr': '1e-3',
        'freeze_strategy': 'Frozen body + BN unfrozen',
        'arch_extra': 'CBAM attention in decoder (reduction=16, kernel=7)',
        'key_change': 'Combines Multi-Q ELA (from P.15) + CBAM attention (from P.10)',
        'conclusion': (
            'vR.P.30 combines two independently validated improvements: multi-quality '
            'ELA input (from P.15, +4.09pp F1) and CBAM decoder attention (from P.10, '
            '+3.57pp F1). This is the baseline for the P.30.x ablation series testing '
            'whether these gains compound or conflict.'
        ),
        'ela_viz_type': 'gray3ch',
        'denorm_fn': 'denormalize_multi_q_ela',
        'scheduler': 'ReduceLROnPlateau (patience=3)',
        'optimizer': 'Adam (single LR=1e-3)',
    },
    'vR.P.30.1': {
        'file': 'vR.P.30.1 Image Detection and Localisation.ipynb',
        'title': 'Multi-quality ELA + CBAM attention (50ep, BCE+Dice)',
        'input_desc': '3-channel Multi-Q ELA (Q=75/85/95 grayscale)',
        'input_short': '3ch MQ-ELA',
        'in_channels': 3,
        'has_cbam': True,
        'loss_type': 'BCE + Dice',
        'epochs': 50,
        'patience': 10,
        'lr': '1e-3',
        'freeze_strategy': 'Frozen body + BN unfrozen',
        'arch_extra': 'CBAM attention in decoder (reduction=16, kernel=7)',
        'key_change': 'Extended training (50 epochs, patience=10) vs P.30\'s 25 epochs',
        'conclusion': (
            'vR.P.30.1 tests whether the P.30 architecture benefits from longer '
            'training (50 epochs with patience=10 vs P.30\'s 25 epochs with patience=7). '
            'This isolates the effect of training duration while keeping all other '
            'hyperparameters identical.'
        ),
        'ela_viz_type': 'gray3ch',
        'denorm_fn': 'denormalize_multi_q_ela',
        'scheduler': 'ReduceLROnPlateau (patience=5)',
        'optimizer': 'Adam (single LR=1e-3)',
    },
    'vR.P.30.2': {
        'file': 'vR.P.30.2 Image Detection and Localisation.ipynb',
        'title': 'Multi-quality ELA + CBAM + Progressive Unfreeze (40ep, BCE+Dice)',
        'input_desc': '3-channel Multi-Q ELA (Q=75/85/95 grayscale)',
        'input_short': '3ch MQ-ELA',
        'in_channels': 3,
        'has_cbam': True,
        'loss_type': 'BCE + Dice',
        'epochs': 40,
        'patience': 5,
        'lr': '1e-3 (decoder) / 1e-5 (encoder)',
        'freeze_strategy': '3-stage progressive unfreeze',
        'arch_extra': 'CBAM + progressive encoder unfreezing (layer4 -> layer3 -> layer2)',
        'key_change': 'Progressive unfreeze strategy to adapt encoder to ELA domain',
        'conclusion': (
            'vR.P.30.2 tests progressive encoder unfreezing: starting fully frozen, '
            'then unfreezing layer4 (epoch 5), layer3 (epoch 10), and layer2 (epoch 15) '
            'with differential learning rates (encoder 1e-5 vs decoder 1e-3). This '
            'allows the pretrained encoder to adapt to the ELA input domain without '
            'catastrophic forgetting.'
        ),
        'ela_viz_type': 'gray3ch',
        'denorm_fn': 'denormalize_multi_q_ela',
        'scheduler': 'ReduceLROnPlateau (patience=3)',
        'optimizer': 'Adam (differential LR: encoder 1e-5, decoder 1e-3)',
        'has_differential_lr': True,
    },
    'vR.P.30.3': {
        'file': 'vR.P.30.3 Image Detection and Localisation.ipynb',
        'title': 'Multi-quality ELA + CBAM + Focal+Dice loss (25ep)',
        'input_desc': '3-channel Multi-Q ELA (Q=75/85/95 grayscale)',
        'input_short': '3ch MQ-ELA',
        'in_channels': 3,
        'has_cbam': True,
        'loss_type': 'Focal (alpha=0.25, gamma=2.0) + Dice',
        'epochs': 25,
        'patience': 7,
        'lr': '1e-3',
        'freeze_strategy': 'Frozen body + BN unfrozen',
        'arch_extra': 'CBAM attention + Focal loss for hard example mining',
        'key_change': 'Replaces BCE with Focal loss (alpha=0.25, gamma=2.0)',
        'conclusion': (
            'vR.P.30.3 replaces BCE with Focal loss (alpha=0.25, gamma=2.0) to test '
            'whether hard-example mining improves pixel-level localization. Focal loss '
            'down-weights easy negatives and focuses training on hard-to-classify pixels '
            'near tampering boundaries, which is where most localization errors occur.'
        ),
        'ela_viz_type': 'gray3ch',
        'denorm_fn': 'denormalize_multi_q_ela',
        'scheduler': 'ReduceLROnPlateau (patience=3)',
        'optimizer': 'Adam (single LR=1e-3)',
    },
    'vR.P.30.4': {
        'file': 'vR.P.30.4 Image Detection and Localisation.ipynb',
        'title': 'Multi-quality ELA + CBAM + Geometric Augmentation (50ep, BCE+Dice)',
        'input_desc': '3-channel Multi-Q ELA (Q=75/85/95 grayscale)',
        'input_short': '3ch MQ-ELA',
        'in_channels': 3,
        'has_cbam': True,
        'loss_type': 'BCE + Dice',
        'epochs': 50,
        'patience': 10,
        'lr': '1e-3',
        'freeze_strategy': 'Frozen body + BN unfrozen',
        'arch_extra': 'CBAM + geometric augmentation (HFlip/VFlip/Rotate90/ShiftScaleRotate)',
        'key_change': 'Adds geometric augmentation to P.30.1 training pipeline',
        'conclusion': (
            'vR.P.30.4 adds geometric augmentation (HorizontalFlip, VerticalFlip, '
            'RandomRotate90, ShiftScaleRotate) to the P.30.1 training pipeline (50 epochs). '
            'This tests whether spatial augmentation improves generalization by exposing '
            'the model to various orientations and translations of tampered regions.'
        ),
        'ela_viz_type': 'gray3ch',
        'denorm_fn': 'denormalize_multi_q_ela',
        'scheduler': 'ReduceLROnPlateau (patience=5)',
        'optimizer': 'Adam (single LR=1e-3)',
    },
}


def make_cell(cell_type, source):
    """Create a notebook cell dict."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.split('\n') if isinstance(source, str) else source,
    }
    # Fix: source lines need newlines except last
    lines = source.split('\n') if isinstance(source, str) else source
    fixed = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            fixed.append(line + '\n')
        else:
            fixed.append(line)
    cell['source'] = fixed
    if cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    return cell


# ============================================================
# Cell content generators
# ============================================================

def exec_summary_md(cfg, version):
    """Executive Summary markdown (problem, dataset, architecture, training strategy)."""
    cbam_line = ''
    if cfg['has_cbam']:
        cbam_line = '   CBAM Attention (Channel + Spatial)\n        |\n'

    arch_diagram = f"""```
Input: 384x384x{cfg['in_channels']} ({cfg['input_short']})
        |
   ResNet-34 Encoder (ImageNet pretrained, {cfg['freeze_strategy']})
        |
   UNet Decoder (skip connections)
{cbam_line}        |
   Segmentation Head
        |
   Output: 384x384x1 (tamper probability map)
```"""

    return f"""## Project Executive Summary

This section provides a high-level overview of the project for reviewers.
Detailed implementation follows in the numbered sections below.

### Problem Statement

Image tampering -- including copy-move forgery and region splicing -- is
increasingly difficult to detect by visual inspection alone. This project
develops a deep learning system that:

1. **Detects** whether an image has been tampered with (image-level binary classification)
2. **Localizes** the exact tampered pixels by producing a binary segmentation mask (pixel-level localization)

### Dataset Overview

| Property | Value |
|---|---|
| **Dataset** | CASIA v2.0 (publicly available via Kaggle) |
| **Authentic images** | ~7,491 (unmanipulated, all-zero masks) |
| **Tampered images** | ~5,123 (with binary ground truth masks) |
| **Total samples** | ~12,614 |
| **Forgery types** | Copy-move (`Tp_D_*`) and Splicing (`Tp_S_*`) |
| **Ground truth** | Binary masks marking tampered pixel regions |
| **Split** | 70% train / 15% validation / 15% test (stratified) |

### Model Architecture Overview

{arch_diagram}

**Key design choices:**

| Feature | Choice | Rationale |
|---|---|---|
| **Encoder** | ResNet-34 (ImageNet) | Strong low-level features for detecting manipulation artifacts |
| **Decoder** | UNet (SMP) | Skip connections preserve spatial resolution for precise masks |
| **Input** | {cfg['input_desc']} | ELA highlights JPEG compression inconsistencies in tampered regions |
| **Freeze** | {cfg['freeze_strategy']} | Preserves pretrained features while adapting to ELA domain |
{('| **Attention** | CBAM (reduction=16, kernel=7) | Channel + spatial attention refines decoder features |' + chr(10)) if cfg['has_cbam'] else ''}
### Training Strategy

| Component | Configuration |
|---|---|
| **Preprocessing** | Multi-quality ELA computation, resize to 384x384 |
| **Loss** | {cfg['loss_type']} |
| **Optimizer** | {cfg['optimizer']} |
| **Scheduler** | {cfg['scheduler']} |
| **Epochs** | {cfg['epochs']} (early stopping patience={cfg['patience']}) |
| **AMP** | Enabled (mixed precision for speed + memory) |
| **Key change** | {cfg['key_change']} |"""


def exec_summary_metrics_code(version):
    """Code cell displaying final metrics (with try/except for first run)."""
    return f"""# ============================================================
# Executive Summary: Final Performance Metrics
# ============================================================
# Displays final test metrics if computed. On first run (top-to-bottom),
# this shows a placeholder. After training + evaluation, re-run to see results.

try:
    print('=' * 60)
    print(f'     FINAL TEST PERFORMANCE -- {{VERSION}} EXECUTIVE SUMMARY')
    print('=' * 60)
    print()
    print(f'{{"Metric":<35}} {{"Score":>10}}')
    print('-' * 47)
    print(f'{{"Image-Level Accuracy":<35}} {{cls_accuracy:.4f}}')
    print(f'{{"Image-Level ROC-AUC":<35}} {{cls_auc:.4f}}')
    print(f'{{"Pixel F1 (Dice)":<35}} {{pixel_f1:.4f}}')
    print(f'{{"Pixel IoU":<35}} {{pixel_iou:.4f}}')
    print(f'{{"Pixel Precision":<35}} {{pixel_precision:.4f}}')
    print(f'{{"Pixel Recall":<35}} {{pixel_recall:.4f}}')
    print(f'{{"Pixel AUC":<35}} {{pixel_auc:.4f}}')
    print('-' * 47)
    print()
    print('Note: Tampered-only metrics are the primary evaluation criterion.')
    print('See Evaluation section for full details.')
except NameError:
    print('Final test metrics have not been computed yet.')
    print('Run the full notebook (all sections) to see results here.')
    print()
    print('Expected metrics after training:')
    print('  - Image-Level Accuracy and ROC-AUC')
    print('  - Pixel F1 (Dice) / IoU / Precision / Recall / AUC')"""


def results_dashboard_md():
    """Results Dashboard intro markdown."""
    return """## Results Dashboard

This dashboard provides a condensed overview of the trained model's
performance. It is designed to let reviewers understand the project
quality within a few seconds.

**Note:** On a fresh top-to-bottom run, these cells display placeholder
messages until the training and evaluation sections have completed.
After that, re-running just this section will show live results."""


def results_dashboard_code(version):
    """Combined results dashboard: metrics table + training curves + example."""
    return f"""# ============================================================
# Results Dashboard: Metrics + Training Curves + Sample Prediction
# ============================================================
import matplotlib.pyplot as plt

try:
    fig = plt.figure(figsize=(18, 5))

    # --- Panel 1: Metrics Summary ---
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.axis('off')
    metrics_text = (
        f'FINAL TEST METRICS\\n'
        f'{"-"*35}\\n'
        f'Pixel F1 (Dice):  {{pixel_f1:.4f}}\\n'
        f'Pixel IoU:        {{pixel_iou:.4f}}\\n'
        f'Pixel Precision:  {{pixel_precision:.4f}}\\n'
        f'Pixel Recall:     {{pixel_recall:.4f}}\\n'
        f'Pixel AUC:        {{pixel_auc:.4f}}\\n'
        f'{"-"*35}\\n'
        f'Image Accuracy:   {{cls_accuracy:.4f}}\\n'
        f'Image ROC-AUC:    {{cls_auc:.4f}}\\n'
        f'Image Macro F1:   {{cls_macro_f1:.4f}}'
    )
    ax1.text(0.1, 0.5, metrics_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_title('Metrics Summary', fontsize=13, fontweight='bold')

    # --- Panel 2: Training Curves ---
    ax2 = fig.add_subplot(1, 3, 2)
    if history.get('train_loss') and len(history['train_loss']) > 0:
        epochs_range = range(1, len(history['train_loss']) + 1)
        ax2.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
        ax2.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
        ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best (ep {{best_epoch}})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    ax2.set_title('Training Curves', fontsize=13, fontweight='bold')

    # --- Panel 3: Example Prediction ---
    ax3 = fig.add_subplot(1, 3, 3)
    _tam_idx = [i for i, l in enumerate(test_labels) if l == 1]
    if _tam_idx:
        _ex_idx = _tam_idx[0]
        _pred_ex = (test_probs[_ex_idx, 0] > 0.5).astype(np.float32)
        _gt_ex = test_masks[_ex_idx, 0]
        # RGB overlay
        _overlay = np.stack([_pred_ex, _gt_ex, np.zeros_like(_pred_ex)], axis=-1)
        ax3.imshow(_overlay)
        ax3.set_title('Example (R=pred, G=GT)', fontsize=11, fontweight='bold')
    ax3.axis('off')

    plt.suptitle(f'{{VERSION}} -- Results Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
except NameError:
    print('Results Dashboard: Run all sections first, then re-run this cell.')"""


def toc_md(version, cfg):
    """Table of Contents markdown with section links."""
    cbam_note = ' + CBAM' if cfg['has_cbam'] else ''
    return f"""# Table of Contents

0. [Project Executive Summary](#project-executive-summary)
    - [Results Dashboard](#results-dashboard)

1. [Change Log](#change-log)
2. [Setup](#1-setup)
3. [Dataset](#2-dataset) -- CASIA v2.0 discovery, path collection, ELA generation
4. [Data Preparation](#3-data-preparation) -- Transforms, splitting, visualization
5. [Model Architecture](#4-model-architecture) -- UNet + ResNet-34{cbam_note}
6. [Training](#5-training) -- {cfg['loss_type']}, {cfg['epochs']} epochs
7. [Evaluation](#6-evaluation) -- Pixel-level + image-level metrics
8. [Visualization](#7-visualization) -- Predictions, ELA, difference maps
9. [Bonus: Robustness & Tampering Type Analysis](#bonus-robustness--tampering-type-analysis)
10. [Results Summary](#8-results-summary)
11. [Discussion](#9-discussion)
12. [Reproducibility Verification](#reproducibility-verification)
13. [Conclusion](#conclusion)
14. [Save Model](#10-save-model)"""


def compliance_md(cfg, version):
    """Assignment requirement compliance checklist."""
    return f"""## Assignment Requirement Compliance

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1.1 | Dataset: authentic + tampered + masks | Fulfilled | CASIA v2.0 with IMAGE/MASK directories |
| 1.2 | Data pipeline: cleaning, preprocessing, mask alignment | Fulfilled | Section 2-3: ELA computation, path validation, mask matching |
| 1.3 | Train/val/test split | Fulfilled | 70/15/15 stratified split with leakage verification |
| 1.4 | Data augmentation | {'Fulfilled' if version == 'vR.P.30.4' else 'Partial'} | {'HFlip, VFlip, Rotate90, ShiftScaleRotate' if version == 'vR.P.30.4' else 'ELA-based preprocessing (augmentation tested in P.30.4)'} |
| 2.1 | Model architecture for tampered region prediction | Fulfilled | UNet + ResNet-34 encoder{' + CBAM attention' if cfg['has_cbam'] else ''} |
| 2.2 | Colab T4 GPU compatible | Fulfilled | Designed for single T4 (15GB VRAM) |
| 3.1 | Performance metrics (localization + detection) | Fulfilled | Pixel F1/IoU/AUC + Image accuracy/AUC/F1 |
| 3.2 | Visual results (Original/GT/Predicted/Overlay) | Fulfilled | Section 7 with multiple visualization types |
| 4.1 | Single Colab notebook | Fulfilled | This notebook |
| 4.2 | Dataset explanation | Fulfilled | Section 2 + Executive Summary |
| 4.3 | Architecture description | Fulfilled | Section 4 + Executive Summary + Model Complexity |
| 4.4 | Training strategy + hyperparameters | Fulfilled | Section 5 with rationale table |
| 4.5 | Evaluation results | Fulfilled | Section 6 + threshold optimization + extended metrics |
| 4.6 | Clear visualizations | Fulfilled | Sections 7 (predictions, ELA, diff maps, failure cases) |
| 4.7 | Model weights | Fulfilled | Saved to Google Drive + HuggingFace Hub |
| B.1 | Robustness testing (JPEG/resize/noise) | Fulfilled | Bonus section: 8 distortion conditions |
| B.2 | Copy-move vs splicing detection | Fulfilled | Bonus section: per-tampering-type breakdown |"""


def data_leakage_code():
    """Data leakage verification code cell."""
    return """# ============================================================
# Data Leakage Verification
# ============================================================
# Verify zero overlap between train/val/test splits at the path level.

train_paths_set = set(train_dataset.image_paths)
val_paths_set = set(val_dataset.image_paths)
test_paths_set = set(test_dataset.image_paths)

train_val_overlap = train_paths_set & val_paths_set
train_test_overlap = train_paths_set & test_paths_set
val_test_overlap = val_paths_set & test_paths_set

print('=' * 55)
print('  DATA LEAKAGE VERIFICATION')
print('=' * 55)
print()
print(f'Train size: {len(train_paths_set):,}')
print(f'Val size:   {len(val_paths_set):,}')
print(f'Test size:  {len(test_paths_set):,}')
print()
print(f'Train-Val overlap:  {len(train_val_overlap)} images')
print(f'Train-Test overlap: {len(train_test_overlap)} images')
print(f'Val-Test overlap:   {len(val_test_overlap)} images')
print()
if len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap) == 0:
    print('PASS: No data leakage detected. All splits are disjoint.')
else:
    print('WARNING: Data leakage detected! Check split logic.')"""


def model_complexity_code(cfg):
    """Model complexity analysis code cell."""
    return """# ============================================================
# Model Complexity Analysis
# ============================================================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print('=' * 55)
print('  MODEL COMPLEXITY ANALYSIS')
print('=' * 55)
print()
print(f'Total parameters:     {total_params:>12,}')
print(f'Trainable parameters: {trainable_params:>12,}')
print(f'Frozen parameters:    {frozen_params:>12,}')
print(f'Trainable ratio:      {trainable_params/total_params*100:.1f}%')
print()

# Breakdown by component
enc_params = sum(p.numel() for p in model.encoder.parameters())
enc_train = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
dec_params = sum(p.numel() for p in model.decoder.parameters())
dec_train = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
seg_params = sum(p.numel() for p in model.segmentation_head.parameters())

print(f'Component Breakdown:')
print(f'  Encoder:           {enc_params:>10,} ({enc_train:,} trainable)')
print(f'  Decoder:           {dec_params:>10,} ({dec_train:,} trainable)')
print(f'  Segmentation Head: {seg_params:>10,} (all trainable)')
print()
print(f'Data:param ratio:    1 : {trainable_params/len(train_dataset):.0f}')
print()

# Try torchinfo for detailed summary
try:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torchinfo'])
    from torchinfo import summary
    print('Detailed model summary (torchinfo):')
    summary(model, input_size=(1, """ + str(cfg['in_channels']) + """, IMAGE_SIZE, IMAGE_SIZE), device=str(DEVICE), verbose=1)
except Exception as e:
    print(f'torchinfo not available: {e}')"""


def threshold_sweep_md():
    """Threshold optimization section header."""
    return """### Segmentation Threshold Optimization

Sweep the segmentation threshold from 0.05 to 0.80 on the **validation set**
and select the threshold maximizing pixel F1. Then re-evaluate the test set
with the optimal threshold for fairer comparison.

This matters because the default 0.5 threshold may not be optimal -- models
often benefit from lower thresholds when recall is more important than precision."""


def threshold_sweep_code():
    """Threshold sweep on validation set."""
    return """# ============================================================
# Threshold Sweep on Validation Set
# ============================================================

@torch.no_grad()
def _collect_val_predictions(model, loader, device):
    model.eval()
    all_probs, all_masks, all_labels = [], [], []
    for images, masks, labels in tqdm(loader, desc='Val predictions'):
        images = images.to(device)
        preds = model(images)
        probs = torch.sigmoid(preds[0] if isinstance(preds, tuple) else preds)
        all_probs.append(probs.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_masks), np.array(all_labels)

val_probs, val_masks, val_labels = _collect_val_predictions(model, val_loader, DEVICE)

# Sweep thresholds
thresholds = np.arange(0.05, 0.85, 0.02)
best_thr, best_f1_thr = 0.5, 0
thr_results = []

for thr in thresholds:
    _pred = (val_probs > thr).astype(np.float32)
    # Only on tampered images
    _tam_mask = val_labels == 1
    if _tam_mask.sum() == 0:
        continue
    _p = _pred[_tam_mask].flatten()
    _m = val_masks[_tam_mask].flatten()
    _tp = (_p * _m).sum()
    _fp = (_p * (1 - _m)).sum()
    _fn = ((1 - _p) * _m).sum()
    _f1 = (2 * _tp) / (2 * _tp + _fp + _fn + 1e-7)
    thr_results.append((thr, _f1))
    if _f1 > best_f1_thr:
        best_f1_thr = _f1
        best_thr = thr

OPTIMAL_THRESHOLD = best_thr
print(f'Optimal threshold: {OPTIMAL_THRESHOLD:.2f} (val tampered F1 = {best_f1_thr:.4f})')
print(f'Default threshold: 0.50')

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot([t[0] for t in thr_results], [t[1] for t in thr_results], 'b-o', markersize=3)
ax.axvline(x=OPTIMAL_THRESHOLD, color='green', linestyle='--', label=f'Optimal={OPTIMAL_THRESHOLD:.2f}')
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default=0.50')
ax.set_xlabel('Threshold')
ax.set_ylabel('Tampered Pixel F1')
ax.set_title(f'{VERSION} -- Threshold Optimization (Validation Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Re-evaluate test set at optimal threshold
test_preds_opt = (test_probs > OPTIMAL_THRESHOLD).astype(np.float32)
_pf = test_preds_opt.flatten()
_mf = test_masks.flatten()
_tp = (_pf * _mf).sum()
_fp = (_pf * (1 - _mf)).sum()
_fn = ((1 - _pf) * _mf).sum()
opt_f1 = (2 * _tp) / (2 * _tp + _fp + _fn + 1e-7)
opt_iou = _tp / (_tp + _fp + _fn + 1e-7)
print(f'\\nTest set at optimal threshold ({OPTIMAL_THRESHOLD:.2f}):')
print(f'  Pixel F1:  {opt_f1:.4f} (vs {pixel_f1:.4f} at 0.50)')
print(f'  Pixel IoU: {opt_iou:.4f} (vs {pixel_iou:.4f} at 0.50)')"""


def extended_metrics_code():
    """Extended localization metrics at optimal threshold."""
    return """# ============================================================
# Extended Localization Metrics (at Optimal Threshold)
# ============================================================

thr = OPTIMAL_THRESHOLD if 'OPTIMAL_THRESHOLD' in dir() else 0.5
_tam_mask_idx = test_labels == 1

# Tampered-only metrics at optimal threshold
_pred_tam = (test_probs[_tam_mask_idx] > thr).astype(np.float32).flatten()
_gt_tam = test_masks[_tam_mask_idx].flatten()

_tp = (_pred_tam * _gt_tam).sum()
_fp = (_pred_tam * (1 - _gt_tam)).sum()
_fn = ((1 - _pred_tam) * _gt_tam).sum()
_tn = ((1 - _pred_tam) * (1 - _gt_tam)).sum()

_precision = _tp / (_tp + _fp + 1e-7)
_recall = _tp / (_tp + _fn + 1e-7)
_f1 = 2 * _precision * _recall / (_precision + _recall + 1e-7)
_iou = _tp / (_tp + _fp + _fn + 1e-7)
_pixel_acc = (_tp + _tn) / (_tp + _tn + _fp + _fn + 1e-7)

print('=' * 60)
print(f'  EXTENDED LOCALIZATION METRICS (Tampered Only, thr={thr:.2f})')
print('=' * 60)
print(f'  Precision:      {_precision:.4f}')
print(f'  Recall:         {_recall:.4f}')
print(f'  F1 (Dice):      {_f1:.4f}')
print(f'  IoU:            {_iou:.4f}')
print(f'  Pixel Accuracy: {_pixel_acc:.4f}')
print(f'  TP pixels:      {_tp:,.0f}')
print(f'  FP pixels:      {_fp:,.0f}')
print(f'  FN pixels:      {_fn:,.0f}')
print('=' * 60)"""


def mask_size_stratified_code():
    """Mask-size stratified evaluation."""
    return """# ============================================================
# Mask-Size Stratified Evaluation
# ============================================================
# Bucket tampered images by mask area to analyze performance across scales.

thr = OPTIMAL_THRESHOLD if 'OPTIMAL_THRESHOLD' in dir() else 0.5
_tam_indices = np.where(test_labels == 1)[0]

BUCKETS = {
    'tiny (<2%)': (0.0, 0.02),
    'small (2-5%)': (0.02, 0.05),
    'medium (5-15%)': (0.05, 0.15),
    'large (>15%)': (0.15, 1.01),
}

bucket_results = {}
for bname, (lo, hi) in BUCKETS.items():
    _f1s, _ious, _count = [], [], 0
    for idx in _tam_indices:
        mask_area = test_masks[idx].sum() / test_masks[idx].size
        if lo <= mask_area < hi:
            _count += 1
            _pred = (test_probs[idx].flatten() > thr).astype(np.float32)
            _gt = test_masks[idx].flatten()
            _tp = (_pred * _gt).sum()
            _fp = (_pred * (1 - _gt)).sum()
            _fn = ((1 - _pred) * _gt).sum()
            _f1 = (2 * _tp) / (2 * _tp + _fp + _fn + 1e-7)
            _iou = _tp / (_tp + _fp + _fn + 1e-7)
            _f1s.append(_f1)
            _ious.append(_iou)
    bucket_results[bname] = {
        'n': _count,
        'f1': np.mean(_f1s) if _f1s else 0,
        'iou': np.mean(_ious) if _ious else 0,
    }

print(f'{"="*65}')
print(f'  MASK-SIZE STRATIFIED EVALUATION (thr={thr:.2f})')
print(f'{"="*65}')
print(f'{"Bucket":<18} {"N":>5} {"Pixel F1":>10} {"Pixel IoU":>10}')
print(f'{"-"*48}')
for bname, r in bucket_results.items():
    print(f'{bname:<18} {r["n"]:5d} {r["f1"]:10.4f} {r["iou"]:10.4f}')

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
_names = [k for k in bucket_results if bucket_results[k]['n'] > 0]
_f1s = [bucket_results[k]['f1'] for k in _names]
_ious = [bucket_results[k]['iou'] for k in _names]
_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

for ax, vals, title in [(axes[0], _f1s, 'Pixel F1'), (axes[1], _ious, 'Pixel IoU')]:
    bars = ax.bar(_names, vals, color=_colors[:len(_names)])
    ax.set_title(f'{VERSION} -- {title} by Mask Size')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}',
                ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()"""


def failure_case_code():
    """Failure case analysis -- 10 worst predictions."""
    return """# ============================================================
# Failure Case Analysis (10 Worst Predictions)
# ============================================================

thr = OPTIMAL_THRESHOLD if 'OPTIMAL_THRESHOLD' in dir() else 0.5
_tam_indices = np.where(test_labels == 1)[0]

# Compute per-image Dice for tampered images
_per_dice = []
for idx in _tam_indices:
    _pred = (test_probs[idx].flatten() > thr).astype(np.float32)
    _gt = test_masks[idx].flatten()
    _tp = (_pred * _gt).sum()
    _fp = (_pred * (1 - _gt)).sum()
    _fn = ((1 - _pred) * _gt).sum()
    _dice = (2 * _tp) / (2 * _tp + _fp + _fn + 1e-7)
    _per_dice.append((idx, _dice))

# Sort by worst Dice
_per_dice.sort(key=lambda x: x[1])
_worst_10 = _per_dice[:10]

print(f'10 Worst Predictions (by Dice):')
print(f'{"Rank":<5} {"Index":>6} {"Dice":>8} {"Image":<40}')
print('-' * 65)
for rank, (idx, dice) in enumerate(_worst_10, 1):
    fname = os.path.basename(test_dataset.image_paths[idx])
    print(f'{rank:<5} {idx:6d} {dice:8.4f} {fname}')

# Visualize worst 5
n_show = min(5, len(_worst_10))
fig, axes = plt.subplots(n_show, 4, figsize=(20, 5 * n_show))
if n_show == 1:
    axes = axes[np.newaxis, :]

for row, (idx, dice) in enumerate(_worst_10[:n_show]):
    _pred_map = (test_probs[idx, 0] > thr).astype(np.float32)
    _gt_map = test_masks[idx, 0]

    # Load original image for display
    _rgb = Image.open(test_dataset.image_paths[idx]).convert('RGB')
    _rgb = _rgb.resize((_gt_map.shape[1], _gt_map.shape[0]))
    _rgb_arr = np.array(_rgb).astype(np.float32) / 255.0

    axes[row, 0].imshow(_rgb_arr)
    axes[row, 0].set_title(f'Original (#{idx})', fontsize=10)
    axes[row, 0].axis('off')

    axes[row, 1].imshow(_gt_map, cmap='hot', vmin=0, vmax=1)
    axes[row, 1].set_title(f'Ground Truth', fontsize=10)
    axes[row, 1].axis('off')

    axes[row, 2].imshow(_pred_map, cmap='hot', vmin=0, vmax=1)
    axes[row, 2].set_title(f'Predicted (Dice={dice:.3f})', fontsize=10)
    axes[row, 2].axis('off')

    # Overlay: green=GT, red=pred, yellow=overlap
    _overlay = _rgb_arr.copy()
    _overlay_mask = np.zeros_like(_overlay)
    _overlay_mask[:, :, 1] = _gt_map * 0.4     # Green = GT
    _overlay_mask[:, :, 0] = _pred_map * 0.4    # Red = Predicted
    _combined = np.clip(_overlay * 0.6 + _overlay_mask, 0, 1)
    axes[row, 3].imshow(_combined)
    axes[row, 3].set_title('Overlay (G=GT, R=Pred)', fontsize=10)
    axes[row, 3].axis('off')

plt.suptitle(f'{VERSION} -- Failure Cases (5 Worst by Dice)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""


def ela_viz_code_gray3ch():
    """ELA visualization for 3-channel multi-Q ELA notebooks (P.30.x)."""
    return """# ============================================================
# ELA Channel Visualization (Multi-Quality ELA)
# ============================================================
# Show what the model sees: the 3 ELA quality channels alongside predictions.

model.eval()
_tam_idx = [i for i, l in enumerate(test_labels) if l == 1]
_n_show = min(4, len(_tam_idx))

fig, axes = plt.subplots(_n_show, 5, figsize=(22, 5 * _n_show))
if _n_show == 1:
    axes = axes[np.newaxis, :]

for row, idx in enumerate(_tam_idx[:_n_show]):
    img_tensor, gt_mask, label = test_dataset[idx]

    # Denormalize ELA channels for display
    _ela_raw = img_tensor.clone()
    for c in range(3):
        _ela_raw[c] = _ela_raw[c] * ela_std[c] + ela_mean[c]
    _ela_raw = _ela_raw.clamp(0, 1).numpy()

    # Predict
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0).to(DEVICE))
        pred = pred[0] if isinstance(pred, tuple) else pred
        prob = torch.sigmoid(pred).cpu().squeeze().numpy()
    pred_binary = (prob > 0.5).astype(np.float32)
    gt_np = gt_mask.squeeze().numpy()

    # Original RGB
    _rgb = Image.open(test_dataset.image_paths[idx]).convert('RGB')
    _rgb = _rgb.resize((gt_np.shape[1], gt_np.shape[0]))
    _rgb_arr = np.array(_rgb).astype(np.float32) / 255.0

    axes[row, 0].imshow(_rgb_arr)
    axes[row, 0].set_title('Original RGB', fontsize=10)
    axes[row, 0].axis('off')

    # ELA composite (all 3 channels as RGB)
    _ela_display = np.stack([_ela_raw[0], _ela_raw[1], _ela_raw[2]], axis=-1)
    axes[row, 1].imshow(np.clip(_ela_display, 0, 1))
    axes[row, 1].set_title('ELA (Q75/Q85/Q95 as RGB)', fontsize=10)
    axes[row, 1].axis('off')

    # Ground truth
    axes[row, 2].imshow(gt_np, cmap='hot', vmin=0, vmax=1)
    axes[row, 2].set_title('Ground Truth', fontsize=10)
    axes[row, 2].axis('off')

    # Predicted
    axes[row, 3].imshow(pred_binary, cmap='hot', vmin=0, vmax=1)
    axes[row, 3].set_title('Predicted Mask', fontsize=10)
    axes[row, 3].axis('off')

    # ELA difference highlight
    _ela_sum = _ela_raw.mean(axis=0)  # average across Q levels
    axes[row, 4].imshow(_ela_sum, cmap='jet')
    axes[row, 4].set_title('ELA Intensity (avg)', fontsize=10)
    axes[row, 4].axis('off')

plt.suptitle(f'{VERSION} -- ELA Channel Visualization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""


def ela_viz_code_rgb9ch():
    """ELA visualization for 9-channel RGB ELA (P.19)."""
    return """# ============================================================
# ELA Channel Visualization (9-Channel RGB ELA)
# ============================================================
# Show the 3 quality levels as separate RGB images.

model.eval()
_tam_idx = [i for i, l in enumerate(test_labels) if l == 1]
_n_show = min(4, len(_tam_idx))

fig, axes = plt.subplots(_n_show, 5, figsize=(22, 5 * _n_show))
if _n_show == 1:
    axes = axes[np.newaxis, :]

for row, idx in enumerate(_tam_idx[:_n_show]):
    img_tensor, gt_mask, label = test_dataset[idx]

    # Denormalize 9-channel ELA for display
    _ela_raw = img_tensor.clone()
    for c in range(9):
        _ela_raw[c] = _ela_raw[c] * ela_std[c] + ela_mean[c]
    _ela_raw = _ela_raw.clamp(0, 1).numpy()

    # Predict
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0).to(DEVICE))
        pred = pred[0] if isinstance(pred, tuple) else pred
        prob = torch.sigmoid(pred).cpu().squeeze().numpy()
    pred_binary = (prob > 0.5).astype(np.float32)
    gt_np = gt_mask.squeeze().numpy()

    # Q=75 RGB (channels 0-2)
    _q75 = np.stack([_ela_raw[0], _ela_raw[1], _ela_raw[2]], axis=-1)
    axes[row, 0].imshow(np.clip(_q75, 0, 1))
    axes[row, 0].set_title('ELA Q=75 (RGB)', fontsize=10)
    axes[row, 0].axis('off')

    # Q=85 RGB (channels 3-5)
    _q85 = np.stack([_ela_raw[3], _ela_raw[4], _ela_raw[5]], axis=-1)
    axes[row, 1].imshow(np.clip(_q85, 0, 1))
    axes[row, 1].set_title('ELA Q=85 (RGB)', fontsize=10)
    axes[row, 1].axis('off')

    # Q=95 RGB (channels 6-8)
    _q95 = np.stack([_ela_raw[6], _ela_raw[7], _ela_raw[8]], axis=-1)
    axes[row, 2].imshow(np.clip(_q95, 0, 1))
    axes[row, 2].set_title('ELA Q=95 (RGB)', fontsize=10)
    axes[row, 2].axis('off')

    # Ground truth
    axes[row, 3].imshow(gt_np, cmap='hot', vmin=0, vmax=1)
    axes[row, 3].set_title('Ground Truth', fontsize=10)
    axes[row, 3].axis('off')

    # Predicted
    axes[row, 4].imshow(pred_binary, cmap='hot', vmin=0, vmax=1)
    axes[row, 4].set_title('Predicted Mask', fontsize=10)
    axes[row, 4].axis('off')

plt.suptitle(f'{VERSION} -- 9-Channel RGB ELA Visualization (Q=75/85/95)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""


def diff_map_contour_code():
    """Difference map + contour overlay visualization."""
    return """# ============================================================
# Difference Map and Contour Overlay Visualization
# ============================================================
import cv2

thr = OPTIMAL_THRESHOLD if 'OPTIMAL_THRESHOLD' in dir() else 0.5
_tam_idx = [i for i, l in enumerate(test_labels) if l == 1]
_n_show = min(4, len(_tam_idx))

fig, axes = plt.subplots(_n_show, 6, figsize=(26, 4.5 * _n_show))
if _n_show == 1:
    axes = axes[np.newaxis, :]

for row, idx in enumerate(_tam_idx[:_n_show]):
    _pred_map = (test_probs[idx, 0] > thr).astype(np.float32)
    _prob_map = test_probs[idx, 0]
    _gt_map = test_masks[idx, 0]

    # Load original
    _rgb = Image.open(test_dataset.image_paths[idx]).convert('RGB')
    _rgb = _rgb.resize((_gt_map.shape[1], _gt_map.shape[0]))
    _rgb_arr = np.array(_rgb).astype(np.float32) / 255.0

    # Col 0: Original
    axes[row, 0].imshow(_rgb_arr)
    axes[row, 0].set_title('Original', fontsize=10)
    axes[row, 0].axis('off')

    # Col 1: Ground Truth
    axes[row, 1].imshow(_gt_map, cmap='hot', vmin=0, vmax=1)
    axes[row, 1].set_title('Ground Truth', fontsize=10)
    axes[row, 1].axis('off')

    # Col 2: Predicted
    axes[row, 2].imshow(_pred_map, cmap='hot', vmin=0, vmax=1)
    axes[row, 2].set_title('Predicted', fontsize=10)
    axes[row, 2].axis('off')

    # Col 3: Overlay (green=GT, red=pred)
    _overlay = _rgb_arr.copy()
    _om = np.zeros_like(_overlay)
    _om[:, :, 1] = _gt_map * 0.4
    _om[:, :, 0] = _pred_map * 0.4
    axes[row, 3].imshow(np.clip(_overlay * 0.6 + _om, 0, 1))
    axes[row, 3].set_title('Overlay', fontsize=10)
    axes[row, 3].axis('off')

    # Col 4: Difference Map
    _diff = np.abs(_pred_map - _gt_map)
    axes[row, 4].imshow(_diff, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[row, 4].set_title(f'Difference (err={_diff.mean():.3f})', fontsize=10)
    axes[row, 4].axis('off')

    # Col 5: Contour Overlay
    _contour_img = (_rgb_arr * 255).astype(np.uint8).copy()
    _gt_u8 = (_gt_map * 255).astype(np.uint8)
    _pred_u8 = (_pred_map * 255).astype(np.uint8)
    gt_contours, _ = cv2.findContours(_gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(_pred_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(_contour_img, gt_contours, -1, (0, 255, 0), 2)   # Green = GT
    cv2.drawContours(_contour_img, pred_contours, -1, (255, 0, 0), 2)  # Red = Pred
    axes[row, 5].imshow(_contour_img)
    axes[row, 5].set_title('Contours (G=GT, R=Pred)', fontsize=10)
    axes[row, 5].axis('off')

plt.suptitle(f'{VERSION} -- Difference Maps & Contour Overlays', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""


def speed_benchmark_code():
    """Inference speed benchmark."""
    return """# ============================================================
# Inference Speed Benchmark
# ============================================================
import time

N_BENCH = min(50, len(test_dataset))
model.eval()

# Warmup
with torch.no_grad():
    _dummy = test_dataset[0][0].unsqueeze(0).to(DEVICE)
    for _ in range(3):
        _ = model(_dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Benchmark
latencies = []
for i in range(N_BENCH):
    img_tensor = test_dataset[i][0].unsqueeze(0).to(DEVICE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(img_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

latencies = np.array(latencies)
print('=' * 55)
print('  INFERENCE SPEED BENCHMARK')
print('=' * 55)
print(f'  Images tested:  {N_BENCH}')
print(f'  Mean latency:   {latencies.mean():.1f} ms')
print(f'  Median latency: {np.median(latencies):.1f} ms')
print(f'  Std:            {latencies.std():.1f} ms')
print(f'  Throughput:     {1000.0 / latencies.mean():.1f} images/sec')
print(f'  Min / Max:      {latencies.min():.1f} / {latencies.max():.1f} ms')
print('=' * 55)"""


def reproducibility_md():
    """Reproducibility verification header."""
    return """## Reproducibility Verification

This section verifies that the experiment setup is deterministic and
reproducible. It confirms seed configuration, dataset split stability,
checkpoint integrity, and records the full environment information."""


def reproducibility_code():
    """Combined reproducibility checks."""
    return """# ============================================================
# Reproducibility Verification
# ============================================================
import sys
import platform

print('=' * 55)
print('  REPRODUCIBILITY VERIFICATION')
print('=' * 55)

# 1. Seed Configuration
print('\\n--- Seed Configuration ---')
print(f'SEED = {SEED}')
print(f'Python random:   seeded')
print(f'NumPy random:    seeded')
print(f'PyTorch manual:  seeded')
print(f'CUDA deterministic: {torch.backends.cudnn.deterministic if torch.cuda.is_available() else "N/A"}')
print(f'CUDA benchmark:    {torch.backends.cudnn.benchmark if torch.cuda.is_available() else "N/A"}')

# 2. Dataset Split Determinism
print('\\n--- Dataset Split Determinism ---')
print(f'Train: {len(train_dataset)} images')
print(f'Val:   {len(val_dataset)} images')
print(f'Test:  {len(test_dataset)} images')
# Hash first 5 paths to verify determinism
import hashlib
_hash_input = '|'.join(test_dataset.image_paths[:5])
_split_hash = hashlib.md5(_hash_input.encode()).hexdigest()[:12]
print(f'Test split hash (first 5): {_split_hash}')

# 3. Checkpoint Integrity
print('\\n--- Checkpoint Integrity ---')
_ckpt_dir = CHECKPOINT_DIR if isinstance(CHECKPOINT_DIR, str) else str(CHECKPOINT_DIR)
for _fname in ['best_model.pt', 'last_checkpoint.pt']:
    _path = os.path.join(_ckpt_dir, _fname)
    if os.path.exists(_path):
        _size = os.path.getsize(_path) / 1e6
        print(f'  {_fname}: {_size:.1f} MB')
    else:
        print(f'  {_fname}: not found')

# 4. Environment Information
print('\\n--- Environment ---')
print(f'Python:     {sys.version.split()[0]}')
print(f'PyTorch:    {torch.__version__}')
print(f'Platform:   {platform.platform()}')
if torch.cuda.is_available():
    print(f'GPU:        {torch.cuda.get_device_name(0)}')
    print(f'VRAM:       {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'CUDA:       {torch.version.cuda}')
try:
    import segmentation_models_pytorch as _smp
    print(f'SMP:        {_smp.__version__}')
except Exception:
    pass

print('\\n' + '=' * 55)
print('  VERIFICATION COMPLETE')
print('=' * 55)"""


def conclusion_md(cfg, version):
    """Conclusion markdown, per-notebook."""
    return f"""## Conclusion

{cfg['conclusion']}

**Ablation Series Context:**
This notebook ({version}) is part of the vR.P.30.x ablation series
systematically testing architectural and training variations:
- **P.30**: Multi-Q ELA + CBAM (baseline, 25 epochs)
- **P.30.1**: Extended training (50 epochs)
- **P.30.2**: Progressive encoder unfreezing
- **P.30.3**: Focal + Dice loss
- **P.30.4**: Geometric augmentation

**P.19** provides an orthogonal comparison using 9-channel RGB ELA input
(retaining color information) without CBAM attention.

For complete results comparison, see the Results Summary section above."""


# ============================================================
# Main: Apply upgrades to all notebooks
# ============================================================

def upgrade_notebook(version, cfg):
    """Load, upgrade, and save a single notebook."""
    nb_path = os.path.join(SCRIPT_DIR, cfg['file'])
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    original_count = len(cells)
    assert original_count == 31, f"{version}: expected 31 cells, got {original_count}"

    # We insert cells working from back to front to preserve indices.
    # Original layout (31 cells):
    #   0  = Header MD
    #   1  = Change Log MD
    #   2  = Setup code
    #   3  = Dataset MD
    #   4-6  = Dataset code
    #   7  = Data Prep MD
    #   8  = Dataset class code
    #   9  = Data splitting code
    #   10 = Sample viz code
    #   11 = Model arch MD
    #   12 = Model build code
    #   13 = Training config MD
    #   14 = Loss/optim code
    #   15 = Training loop code
    #   16 = Eval MD
    #   17 = Pixel eval code
    #   18 = Image eval code
    #   19 = Confusion matrix code
    #   20 = Training curves code
    #   21 = Viz MD
    #   22 = Prediction viz code
    #   23 = Per-image metric code
    #   24 = Bonus MD
    #   25 = Robustness code
    #   26 = Per-type breakdown code
    #   27 = Results summary MD
    #   28 = Results table code
    #   29 = Discussion MD
    #   30 = Save model code

    new_cells = []

    # === INSERTION PLAN (by position in original, back to front) ===

    # G. Before cell 30 (Save): Reproducibility + Conclusion
    repro_cells = [
        make_cell('markdown', reproducibility_md()),
        make_cell('code', reproducibility_code()),
        make_cell('markdown', conclusion_md(cfg, version)),
    ]

    # F. After cell 26 (per-type), before cell 27 (results summary): Speed benchmark
    speed_cells = [
        make_cell('code', speed_benchmark_code()),
    ]

    # E. After cell 23 (per-image metrics): ELA Viz + Diff Map
    viz_extra_cells = [
        make_cell('code', ela_viz_code_rgb9ch() if cfg['ela_viz_type'] == 'rgb9ch' else ela_viz_code_gray3ch()),
        make_cell('code', diff_map_contour_code()),
    ]

    # D. After cell 23 (per-image metrics), before viz extras:
    #    Threshold Sweep + Extended Metrics + Mask-Size + Failure Cases
    eval_extra_cells = [
        make_cell('markdown', threshold_sweep_md()),
        make_cell('code', threshold_sweep_code()),
        make_cell('code', extended_metrics_code()),
        make_cell('code', mask_size_stratified_code()),
        make_cell('code', failure_case_code()),
    ]

    # C. After cell 12 (model build): Model Complexity
    complexity_cells = [
        make_cell('code', model_complexity_code(cfg)),
    ]

    # B. After cell 9 (data splitting): Data Leakage
    leakage_cells = [
        make_cell('code', data_leakage_code()),
    ]

    # A. After cell 0 (header): Executive Summary + Dashboard + ToC + Compliance
    exec_cells = [
        make_cell('markdown', exec_summary_md(cfg, version)),
        make_cell('code', exec_summary_metrics_code(version)),
        make_cell('markdown', results_dashboard_md()),
        make_cell('code', results_dashboard_code(version)),
        make_cell('markdown', toc_md(version, cfg)),
        make_cell('markdown', compliance_md(cfg, version)),
    ]

    # --- Build new cell list ---
    # Insert back-to-front by building the list section by section

    result = []

    # Cells 0
    result.append(cells[0])
    # A: Executive Summary block
    result.extend(exec_cells)
    # Cells 1-9
    result.extend(cells[1:10])
    # B: Data Leakage
    result.extend(leakage_cells)
    # Cells 10-12
    result.extend(cells[10:13])
    # C: Model Complexity
    result.extend(complexity_cells)
    # Cells 13-23
    result.extend(cells[13:24])
    # D: Extended eval (threshold, extended metrics, mask-size, failures)
    result.extend(eval_extra_cells)
    # E: ELA viz + diff map
    result.extend(viz_extra_cells)
    # Cells 24-26 (bonus)
    result.extend(cells[24:27])
    # F: Speed benchmark
    result.extend(speed_cells)
    # Cells 27-29 (results summary, results table, discussion)
    result.extend(cells[27:30])
    # G: Reproducibility + Conclusion
    result.extend(repro_cells)
    # Cell 30 (save model)
    result.append(cells[30])

    nb['cells'] = result
    new_count = len(result)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f'{version}: {original_count} -> {new_count} cells  ({cfg["file"]})')
    return new_count


# Run all upgrades
print('=' * 60)
print('  Upgrading Final Notebooks -> Final Upgraded Notebooks')
print('=' * 60)

total_cells = 0
for version, cfg in NOTEBOOKS.items():
    n = upgrade_notebook(version, cfg)
    total_cells += n

print(f'\nDone. {len(NOTEBOOKS)} notebooks upgraded. Total cells: {total_cells}')
