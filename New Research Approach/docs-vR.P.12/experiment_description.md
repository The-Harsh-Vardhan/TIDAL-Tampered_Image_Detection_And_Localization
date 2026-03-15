# Experiment Description — vR.P.12: ELA + Data Augmentation

| Field | Value |
|-------|-------|
| **Version** | vR.P.12 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.3 (ELA as input, frozen body + BN unfrozen) |
| **Change** | Add controlled Albumentations augmentation pipeline to training |
| **Encoder** | ResNet-34 (ImageNet, frozen body, BatchNorm unfrozen) |
| **Input** | ELA 384x384x3 (RGB ELA map, Q=90) |

---

## 1. Motivation

### Limited Augmentation Causes Overfitting

All pretrained experiments (vR.P.0-P.11) train with **no data augmentation**. The model sees each training image exactly once per epoch in a fixed orientation. This means:

- The model may memorize specific tampering patterns present in CASIA v2.0
- Predictions may be orientation-dependent (a horizontal splice boundary may be detected but the same boundary rotated 90 degrees may be missed)
- The effective training set diversity is limited to the ~8,800 training images

Data augmentation artificially expands the training distribution by applying geometric and photometric transforms, forcing the model to learn **orientation-invariant and position-invariant** forensic features.

### Forensic Signals Must Be Preserved

Unlike natural image classification, tampering detection relies on **subtle compression artifacts** visible in ELA maps. Aggressive augmentations can destroy these forensic signals:

| Augmentation Type | Effect on ELA Signal | Safe? |
|-------------------|---------------------|-------|
| Horizontal/Vertical Flip | Preserves all pixel values | Yes |
| 90-degree Rotation | Preserves all pixel values | Yes |
| Small shift/scale/rotate | Minimal interpolation artifacts | Yes (with limits) |
| Mild Gaussian blur | Slight smoothing of ELA patterns | Mostly safe (low p) |
| Mild brightness/contrast | Small intensity shifts | Mostly safe (low limits) |
| **Heavy color jitter** | **Distorts ELA intensity patterns** | **No** |
| **Strong JPEG compression** | **Creates new compression artifacts** | **No** |
| **Large rotations** | **Heavy interpolation destroys ELA edges** | **No** |
| **Extreme scaling** | **Resampling artifacts mask forensic signal** | **No** |

The augmentation pipeline for this experiment is carefully designed to **improve geometric diversity while preserving ELA fidelity**.

---

## 2. Safe Augmentation Strategy

### Albumentations Pipeline

```python
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3
    ),
    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    A.RandomBrightnessContrast(
        brightness_limit=0.1, contrast_limit=0.1, p=0.2
    ),
])
```

**Why these specific transforms:**

| Transform | Purpose | Parameters | Risk Level |
|-----------|---------|------------|------------|
| HorizontalFlip | Orientation invariance | p=0.5 | None (lossless) |
| VerticalFlip | Orientation invariance | p=0.3 | None (lossless) |
| RandomRotate90 | Rotation invariance | p=0.5 | None (lossless) |
| ShiftScaleRotate | Position/scale invariance | shift=5%, scale=5%, rot=10deg | Low (minimal interpolation) |
| GaussianBlur | Robustness to slight defocus | blur_limit=(3,5), p=0.1 | Low (rare, mild) |
| RandomBrightnessContrast | Intensity robustness | limit=0.1, p=0.2 | Low (small range) |

### Why Not RandomCrop?

Images are already resized to 384x384. A RandomCrop at the same size is a no-op. The `shift_limit=0.05` in ShiftScaleRotate provides equivalent translation augmentation by shifting the image by up to 5% in each direction.

### Destructive Augmentations — Explicitly Avoided

The following are **never used** because they destroy the forensic signal:

1. **Heavy ColorJitter / HueSaturationValue** — ELA maps encode compression artifact intensity. Large hue/saturation shifts corrupt this signal.
2. **Strong JPEG compression simulation** — Creates new compression artifacts that overlap with the genuine tampering signal, making the two indistinguishable.
3. **Large rotations (> 15 degrees)** — Require heavy interpolation that smears fine ELA edges and creates artificial patterns.
4. **Extreme scaling (> 10%)** — Resampling artifacts at boundaries mimic tampering artifacts, creating false positives.
5. **Elastic/GridDistortion** — Non-rigid deformations create artificial intensity gradients that confuse the model.
6. **CoarseDropout/Cutout** — Removing patches destroys spatial relationships that the model needs to learn.

---

## 3. What Changed from vR.P.3

| Aspect | vR.P.3 | vR.P.12 (This Version) |
|--------|--------|------------------------|
| **Augmentation** | None | **Albumentations pipeline (6 transforms)** |
| **Loss function** | SoftBCEWithLogitsLoss + DiceLoss | **FocalLoss(alpha=0.25, gamma=2.0) + DiceLoss** |
| **EPOCHS** | 25 | **50** (augmented data needs more epochs) |
| **PATIENCE** | 7 | **10** (longer convergence allowance) |
| **NUM_WORKERS** | 2 | **4** |
| **DataLoader** | No prefetch_factor | **prefetch_factor=2** |
| **cudnn.benchmark** | default | **True** |
| **Dependencies** | smp | **smp + albumentations** |

---

## 4. What DID NOT Change (Frozen)

- Architecture: UNet + ResNet-34 (SMP)
- Resolution: 384x384
- Batch size: 16
- Input: ELA (Q=90, brightness-scaled)
- Normalization: ELA-specific mean/std (computed from training set)
- Encoder state: Frozen body + BN unfrozen
- Optimizer: Adam, single LR=1e-3, weight_decay=1e-5
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling (same 3-tier fallback)
- AMP + TF32 enabled
- Evaluation: pixel-level + image-level metrics
- Val/test augmentation: None (clean evaluation)

---

## 5. Experiment Lineage

```
vR.P.0 (baseline)
  +-- P.1 (dataset fix)
       +-- P.1.5 (speed optimizations)
            +-- P.2 (gradual unfreeze, RGB)
                 +-- P.3 (ELA input, frozen + BN)  <- BEST Pixel F1 = 0.6920
                      +-- P.4 (RGB + ELA 4-channel)
                      +-- P.7 (ELA + extended training)
                      +-- P.8 (ELA + progressive unfreeze)
                      +-- P.9 (Focal + Dice loss)
                      +-- P.11 (Higher resolution 512x512)
                      +-- P.12 (ELA + data augmentation)  <- THIS
            +-- P.5 (ResNet-50 encoder)
            +-- P.6 (EfficientNet-B0 encoder)
```

vR.P.12 tests whether **controlled data augmentation improves generalization** for ELA-based tampering localization. This isolates the effect of training data diversity while keeping all other variables identical to P.3.

---

## 6. Augmentation Applied to Both Image and Mask

Albumentations applies identical spatial transforms to both the ELA image and the ground truth mask. This ensures:

- A horizontally flipped ELA map has a correspondingly flipped mask
- A rotated ELA map has a correspondingly rotated mask
- Pixel-level alignment is maintained after all transforms

Photometric transforms (GaussianBlur, RandomBrightnessContrast) are applied **only to the image**, not the mask. This is handled automatically by Albumentations' `image`/`mask` API.
