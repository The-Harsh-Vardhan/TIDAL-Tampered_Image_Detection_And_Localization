# Experiment Description — vR.P.3: ELA as Input

| Field | Value |
|-------|-------|
| **Version** | vR.P.3 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.2 (Gradual unfreeze) |
| **Change** | Replace RGB input with ELA input (Q=90, brightness-scaled) |
| **Encoder** | ResNet-34 (ImageNet, frozen body, BatchNorm unfrozen for domain adaptation) |
| **Input** | ELA 384×384×3 (RGB ELA map) |

---

## 1. Motivation

The pretrained track has used **RGB images** as input because ImageNet-pretrained encoders expect RGB statistics. However, the ETASR paper and Track 1 both demonstrate that **Error Level Analysis (ELA)** is a powerful forensic signal — ELA highlights re-compression artifacts in tampered regions.

vR.P.3 tests a fundamental question: **Can a pretrained encoder extract useful forensic features from ELA maps, even though ELA's distribution differs from ImageNet?**

The encoder is returned to **frozen** (reversing vR.P.2's unfreeze) with **BatchNorm layers unfrozen**. This allows the BN running statistics to adapt to ELA's different mean/std without modifying the convolutional weights. This is the standard approach for domain adaptation with frozen pretrained encoders.

---

## 2. What Changed from vR.P.2

| Aspect | vR.P.2 | vR.P.3 (This Version) |
|--------|--------|----------------------|
| **Input type** | RGB (raw image) | **ELA (Q=90, brightness-scaled)** |
| **Normalization** | ImageNet mean/std | **ELA-specific mean/std (computed from training set)** |
| **Encoder state** | Partially unfrozen (layer3+4, lr=1e-5) | **Frozen body, BN unfrozen** |
| **Optimizer** | 2 param groups (differential LR) | **Single LR (decoder + BN only)** |
| **Trainable params** | ~5M | **~500K (decoder) + BN params** |
| **Dataset transform** | Load RGB → ImageNet normalize | **Load RGB → compute ELA → normalize** |

---

## 3. What DID NOT Change (Frozen)

- Architecture: UNet + ResNet-34 (SMP)
- `IN_CHANNELS = 3` (ELA is 3-channel RGB)
- Loss: BCEDiceLoss
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=7, monitor=val_loss
- Batch size: 16
- Max epochs: 25
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling (same 3-tier fallback)
- AMP + TF32 enabled
- Checkpoint save/resume
- Evaluation: pixel-level + image-level metrics

---

## 4. ELA Preprocessing

```
For each image:
  1. Open as RGB
  2. Re-save as JPEG at quality=90
  3. Compute pixel-wise absolute difference (original - resaved)
  4. Find maximum channel difference across image
  5. Scale brightness by 255/max_diff (normalize to full range)
  6. Resize to 384×384
  7. Convert to tensor [0, 1]
  8. Normalize with ELA-specific mean/std (computed from training set)
```

ELA produces 3-channel RGB difference maps where:
- **Black regions** = no compression artifacts (authentic or never-tampered)
- **Bright regions** = compression mismatch (potential tampering indicators)
- **Distribution** is fundamentally different from natural images (sparse, high-contrast)
