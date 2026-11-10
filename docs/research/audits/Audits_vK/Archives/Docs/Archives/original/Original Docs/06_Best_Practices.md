# 6. Best Practices: Engineering Standards for This Project

## 6.1 Overview

This document defines the engineering standards that will govern every aspect of the project — from code organization to evaluation methodology. Following these practices is what separates a student assignment from an industry-grade submission. The evaluators at BigVision are looking for **engineering maturity**, not just a working model.

---

## 6.2 Reproducibility

### 6.2.1 Random Seed Discipline

Every source of randomness must be seeded for reproducible results:

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Call `set_seed()` at the very beginning of the notebook**, before any data loading or model initialization.

### 6.2.2 Library Version Pinning

Pin exact versions of all critical libraries in the first notebook cell:

```python
!pip install segmentation-models-pytorch==0.3.4
!pip install albumentations==1.4.0
!pip install timm==1.0.3
```

This ensures anyone running the notebook gets identical behavior regardless of when they run it.

### 6.2.3 Deterministic Data Splits

Use a fixed `random_state` for data splitting:
```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.15, random_state=42, stratify=labels)
```

---

## 6.3 Data Pipeline Standards

### 6.3.1 Synchronized Augmentation

**Always use `albumentations` for image segmentation augmentation** — it automatically applies the same spatial transforms to both image and mask:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

**Never use `torchvision.transforms` for segmentation** — it doesn't synchronize spatial transforms between image and mask.

### 6.3.2 DataLoader Configuration

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,        # Colab optimal
    pin_memory=True,      # Faster CPU→GPU transfer
    drop_last=True,       # Prevents batch size 1 issues with BatchNorm
    persistent_workers=True
)
```

### 6.3.3 Data Validation

Before training, always run a validation pass:
1. **Verify image-mask pairs**: Check that every tampered image has a corresponding mask
2. **Verify dimensions**: `assert image.shape[:2] == mask.shape[:2]` for every pair
3. **Verify mask values**: After binarization, masks should only contain 0 and 1 (or 0 and 255)
4. **Visual spot-check**: Display 5-10 random image-mask pairs to verify alignment

---

## 6.4 Training Discipline

### 6.4.1 Mixed Precision Training (AMP)

**Always use AMP** on T4 GPU — there is no reason not to:

```python
scaler = torch.cuda.amp.GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(input)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Impact**: ~2× training speed, ~50% VRAM reduction. Free performance.

### 6.4.2 Gradient Accumulation

When batch size 16 doesn't fit in VRAM, accumulate over 4 micro-batches of 4:

```python
accumulation_steps = 4
for i, (input, target) in enumerate(train_loader):
    with torch.cuda.amp.autocast():
        output = model(input)
        loss = criterion(output, target) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 6.4.3 Logging and Monitoring

Track these metrics every epoch (at minimum):

| Metric | Train | Validation |
|--------|-------|------------|
| Loss (total) | ✅ | ✅ |
| BCE component | ✅ | ✅ |
| Dice component | ✅ | ✅ |
| Pixel-F1 | — | ✅ |
| Pixel-IoU | — | ✅ |
| Learning Rate | ✅ | — |

Store all logs in a dictionary and plot training curves at the end.

### 6.4.4 Checkpointing Strategy

```python
if val_f1 > best_f1:
    best_f1 = val_f1
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1,
        'config': config_dict
    }, 'best_model.pth')
```

**Save by validation F1, not by validation loss.** Loss can decrease while F1 stagnates (the model gets better at predicting the background but worse at finding tampered pixels).

### 6.4.5 Early Stopping

Stop training if validation F1 doesn't improve for `patience` epochs:
- **Patience**: 10 epochs
- **Min delta**: 0.001 (ignore improvements smaller than 0.1%)

---

## 6.5 Evaluation Standards

### 6.5.1 Required Metrics

| Metric | Level | What It Measures | Formula |
|--------|-------|-----------------|---------|
| **Pixel-F1** | Pixel | Harmonic mean of precision and recall at pixel level | $\frac{2TP}{2TP + FP + FN}$ |
| **Pixel-IoU** | Pixel | Intersection over Union of predicted vs. ground truth mask | $\frac{TP}{TP + FP + FN}$ |
| **Image-Level Accuracy** | Image | Correct classification of authentic vs. tampered | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| **AUC-ROC** | Pixel | Threshold-independent discrimination ability | Area under the ROC curve |

### 6.5.2 Threshold Selection

- **Default threshold**: 0.5 (for reporting primary metrics)
- **Oracle-F1**: Sweep thresholds from 0.01 to 0.99 and report the best achievable F1 — this shows the model's discriminative potential independent of threshold calibration
- **Report both**: Default F1 and Oracle-F1 in the results table

### 6.5.3 Image-Level Detection from Pixel Predictions

Derive image-level classification from the segmentation output:
```python
# Image is "tampered" if any pixel exceeds threshold
image_pred = (mask_pred.max() > threshold).float()
```

Or use the mean predicted probability:
```python
# Image is "tampered" if average pixel probability exceeds a lower threshold
image_pred = (mask_pred.mean() > 0.1).float()
```

Report both approaches and pick the one with better accuracy.

### 6.5.4 Statistical Rigor

- Report metrics on the **hold-out test set** (never seen during training or validation)
- Report **mean ± standard deviation** if running multiple seeds (optional but impressive)
- Report per-class metrics (splicing vs. copy-move) if possible

---

## 6.6 Visualization Standards

### 6.6.1 Required 4-Column Grid

For every test sample visualized, show:

| Column 1 | Column 2 | Column 3 | Column 4 |
|-----------|----------|----------|----------|
| Original RGB Image | Ground Truth Mask | Predicted Probability Heatmap | Overlay (mask on image, alpha=0.5) |

```python
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[1].imshow(gt_mask, cmap='gray')
axes[1].set_title("Ground Truth")
axes[2].imshow(pred_heatmap, cmap='hot')
axes[2].set_title("Prediction Heatmap")
axes[3].imshow(image)
axes[3].imshow(pred_mask, cmap='jet', alpha=0.5)
axes[3].set_title("Overlay")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
```

### 6.6.2 Visualization Selection

Show **at least**:
- 3-5 **best** predictions (high F1 samples) — demonstrates capability
- 3-5 **worst** predictions (low F1 samples) — demonstrates honest evaluation and understanding of failure modes
- 2-3 **authentic images** — demonstrates the model doesn't hallucinate tampering on clean images
- 1-2 **edge cases** — e.g., very small tampered regions, copy-move with similar textures

### 6.6.3 Training Curves

Always plot:
1. Training loss vs. Validation loss (per epoch)
2. Validation Pixel-F1 vs. epoch
3. Learning rate schedule vs. epoch

---

## 6.7 Notebook Structure

The Colab notebook should follow this exact structure (matching the assignment's 4 sections):

```
📓 Tampered Image Detection & Localization
│
├── [Markdown] Title, Author, Date, Abstract
├── [Markdown] Table of Contents
│
├── 🔷 Section 1: Dataset Selection & Preparation
│   ├── [Markdown] Dataset explanation (CASIA v2.0 description, why chosen)
│   ├── [Code] Environment setup & library installations
│   ├── [Code] Dataset download via Kaggle API
│   ├── [Code] Data cleaning & validation (misalignment check, binarization)
│   ├── [Code] Data exploration (class distribution, sample visualization)
│   ├── [Code] Train/Val/Test split
│   ├── [Code] Augmentation pipeline
│   └── [Code] DataLoader setup
│
├── 🔷 Section 2: Model Architecture & Learning
│   ├── [Markdown] Architecture description (U-Net + SRM, why this choice)
│   ├── [Code] SRM Filter Bank implementation
│   ├── [Code] Model definition (SMP U-Net with 6-channel input)
│   ├── [Markdown] Loss function explanation
│   ├── [Code] Hybrid loss implementation (BCE + Dice + Edge)
│   ├── [Code] Training loop with AMP + gradient accumulation
│   └── [Code] Training execution + live logging
│
├── 🔷 Section 3: Testing & Evaluation
│   ├── [Code] Load best checkpoint
│   ├── [Code] Compute all metrics (F1, IoU, Accuracy, AUC-ROC)
│   ├── [Markdown] Results table with analysis
│   ├── [Code] Visual results (4-column grid)
│   ├── [Code] Training curves plot
│   ├── [Code] (Bonus) Robustness evaluation table
│   └── [Code] (Bonus) COVERAGE dataset evaluation
│
├── 🔷 Section 4: Deliverables & Documentation
│   ├── [Markdown] Summary of findings
│   ├── [Markdown] Known limitations & future work
│   ├── [Code] Save model weights to Google Drive
│   └── [Markdown] Links to assets (model weights, notebook)
│
└── [Markdown] References
```

Every code cell should have a markdown cell above it explaining **what** it does and **why**.

---

## 6.8 Code Quality Standards

### 6.8.1 Do's

- ✅ Use descriptive variable names (`train_loader` not `tl`)
- ✅ Use functions for repeated operations (e.g., `visualize_predictions()`, `compute_metrics()`)
- ✅ Use config dictionaries for hyperparameters (easy to modify + self-documenting)
- ✅ Print shapes after key operations during development (remove in final version)
- ✅ Use `tqdm` for progress bars during training

### 6.8.2 Don'ts

- ❌ Don't hardcode file paths (use variables defined once at the top)
- ❌ Don't silence errors or use bare `except:` clauses
- ❌ Don't leave debug print statements in the final notebook
- ❌ Don't use deprecated APIs (`torch.cuda.amp` is current; old `apex` is deprecated)
- ❌ Don't train without monitoring — always log loss and metrics per epoch

### 6.8.3 Configuration Pattern

```python
CONFIG = {
    'seed': 42,
    'image_size': 512,
    'batch_size': 4,
    'accumulation_steps': 4,
    'epochs': 50,
    'lr_encoder': 1e-4,
    'lr_decoder': 1e-3,
    'weight_decay': 1e-4,
    'patience': 10,
    'encoder_name': 'efficientnet-b1',
    'threshold': 0.5,
}
```

---

## 6.9 Common Pitfalls to Avoid

| Pitfall | Why It Happens | How to Avoid |
|---------|---------------|-------------|
| **Empty masks (all zeros)** | Class imbalance; BCE-only loss | Use Dice Loss; verify loss components are non-zero |
| **Overfitting to training set** | Small dataset, large model | Augmentation + early stopping + weight decay |
| **NaN loss** | AMP underflow or extreme learning rate | Use GradScaler properly; reduce LR if NaN appears |
| **Memory overflow (OOM)** | Batch too large for 512×512 | Reduce batch size; use gradient accumulation |
| **Misaligned image-mask pairs** | CASIA v2.0 naming/dimension issues | Run validation script before training |
| **Model predicts semantic objects** | No forensic preprocessing | Always include SRM/noise features |
| **Reporting inflated metrics** | Evaluating on training data or biased split | Strict train/val/test separation; report test-only metrics |
