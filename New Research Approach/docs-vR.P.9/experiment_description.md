# Experiment Description — vR.P.9: Focal + Dice Loss

| Field | Value |
|-------|-------|
| **Version** | vR.P.9 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.3 (ELA as input, frozen body + BN unfrozen) |
| **Change** | Replace BCE+Dice loss with Focal+Dice loss |
| **Encoder** | ResNet-34 (ImageNet, frozen body, BatchNorm unfrozen) |
| **Input** | ELA 384×384×3 (RGB ELA map, Q=90) |

---

## 1. Motivation

### Limitations of BCE Loss

The current pretrained experiments (vR.P.0–P.8) use **SoftBCEWithLogitsLoss + DiceLoss** as the combined objective. While effective, standard BCE treats all pixels equally — every pixel contributes the same gradient magnitude regardless of whether it is easy or hard to classify.

In tampering localization, this creates a problem: **tampered regions are small**. In CASIA v2.0, tampering typically covers 5–30% of the image. The remaining 70–95% of pixels are authentic (background). BCE loss is dominated by these easy-to-classify background pixels, leaving limited gradient signal for the hard boundary pixels that matter most for localization quality.

### Class Imbalance at the Pixel Level

| Region | Typical Coverage | Classification Difficulty |
|--------|-----------------|--------------------------|
| Authentic background | 70–95% | Easy (uniform compression) |
| Tampered interior | 2–20% | Moderate (different compression artifacts) |
| Tampered boundary | 1–5% | Hard (transition zone, ambiguous pixels) |

The boundary pixels are the most important for Pixel F1 and IoU — they determine the segmentation quality — but receive the least training attention under BCE.

### Why Focal Loss Improves Hard-Pixel Learning

Focal loss (Lin et al., 2017) modifies the standard cross-entropy by adding a modulating factor:

```
FL(p) = -alpha * (1 - p)^gamma * log(p)
```

Where:
- `p` = model's predicted probability for the correct class
- `alpha = 0.25` — balances positive/negative class contribution
- `gamma = 2.0` — reduces loss for well-classified pixels (when p > 0.5, the factor (1-p)^2 < 0.25)

**Effect:** Easy pixels (p close to 1.0) contribute near-zero loss. Hard pixels (p close to 0.5) contribute full loss. This automatically focuses training on the informative boundary region without explicit hard example mining.

---

## 2. Loss Design

### Component 1: Dice Loss

Dice loss directly optimizes the **Dice coefficient** (equivalent to F1 score for binary segmentation):

```
DiceLoss = 1 - (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)
```

Properties:
- Optimizes **region overlap** directly
- Naturally handles class imbalance (operates on set intersection, not per-pixel)
- Encourages better segmentation boundaries
- Implementation: `smp.losses.DiceLoss(mode='binary', from_logits=True)`

### Component 2: Focal Loss

Binary focal loss with logits:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Parameters:
- `alpha = 0.25` — standard for binary segmentation tasks
- `gamma = 2.0` — standard modulating factor (reduces easy-pixel contribution by ~4×)
- Implementation: `smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)`

### Combined Loss

```python
loss = FocalLoss(pred, target) + DiceLoss(pred, target)
```

**Why this combination works:**
- Dice loss ensures good global region overlap (macro quality)
- Focal loss ensures hard boundary pixels receive adequate gradient (micro quality)
- The combination balances **region accuracy** (Dice) with **hard example mining** (Focal)
- Both operate on raw logits — no sigmoid pre-processing needed

---

## 3. What Changed from vR.P.3

| Aspect | vR.P.3 | vR.P.9 (This Version) |
|--------|--------|----------------------|
| **Loss function** | SoftBCEWithLogitsLoss + DiceLoss | **FocalLoss(alpha=0.25, gamma=2.0) + DiceLoss** |
| **Hard pixel focus** | No (all pixels weighted equally) | **Yes (focal modulation, gamma=2.0)** |
| **NUM_WORKERS** | 2 | **4** |
| **DataLoader** | No prefetch_factor | **prefetch_factor=2** |

---

## 4. What DID NOT Change (Frozen)

- Architecture: UNet + ResNet-34 (SMP)
- Input: ELA (Q=90, brightness-scaled)
- Normalization: ELA-specific mean/std (computed from training set)
- Encoder state: Frozen body + BN unfrozen
- Optimizer: Adam, single LR=1e-3, weight_decay=1e-5
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=7, monitor=val_loss
- Batch size: 16
- Max epochs: 25
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling (same 3-tier fallback)
- AMP + TF32 enabled
- Evaluation: pixel-level + image-level metrics

---

## 5. Experiment Lineage

```
vR.P.0 (baseline)
  └→ P.1 (dataset fix)
       └→ P.1.5 (speed optimizations)
            └→ P.2 (gradual unfreeze, RGB)
                 └→ P.3 (ELA input, frozen + BN)
                      ├→ P.4 (RGB + ELA 4-channel)
                      ├→ P.7 (ELA + extended training)
                      ├→ P.8 (ELA + progressive unfreeze)
                      └→ P.9 (Focal + Dice loss)  ← THIS
            ├→ P.5 (ResNet-50 encoder)
            └→ P.6 (EfficientNet-B0 encoder)
```

vR.P.9 tests whether **optimization objectives can improve segmentation quality without changing architecture or input representation**. This isolates the effect of the loss function alone.

---

## 6. Why Alpha=0.25 and Gamma=2.0

These are the standard parameters from the original Focal Loss paper (Lin et al., 2017):

- **alpha=0.25**: The foreground (tampered) class gets weight 0.25, and background (authentic) gets 0.75. This may seem counterintuitive, but focal loss already up-weights hard examples — using alpha < 0.5 for the minority class prevents over-correction.
- **gamma=2.0**: At gamma=2, a pixel classified with 0.9 confidence contributes only 1% of the loss of a pixel at 0.5 confidence. This 100× reduction effectively eliminates easy background pixels from the gradient.
