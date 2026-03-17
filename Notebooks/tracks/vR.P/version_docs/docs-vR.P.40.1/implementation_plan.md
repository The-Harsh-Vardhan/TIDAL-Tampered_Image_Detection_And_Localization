# vR.P.40.1 — Implementation Plan

## Core Implementation: EfficientNet-B4 Encoder Swap

### Hypothesis

EfficientNet-B4's compound-scaled architecture with built-in SE attention will outperform ResNet-34 on forensic feature extraction from ELA maps, even without input pipeline changes.

### Model Build

1. Create UNet with `encoder_name='efficientnet-b4'`, `encoder_weights='imagenet'`, `in_channels=3`
2. Freeze ALL encoder parameters
3. Unfreeze only BatchNorm layers (domain adaptation for ELA input)
4. No CBAM injection (EfficientNet-B4 already has SE blocks)
5. BATCH_SIZE=8 to fit GPU memory

### ELA Preprocessing (Same as P.3)

- `compute_ela_rgb(image_path, quality=90)` — single quality ELA, RGB output (3ch)
- `compute_ela_statistics()` — per-channel mean/std from 500 training samples
- Normalize: `(pixel - mean) / std` per channel

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.40.1 — EfficientNet-B4 baseline (ELA Q=90, 3ch)" |
| 1 | Changelog | Add P.40.x series entries |
| 2 | Setup | VERSION='vR.P.40.1', ENCODER='efficientnet-b4', BATCH_SIZE=8 |
| 8 | Dataset class | Single-Q ELA RGB (3ch) — simplified from P.30.1's multi-Q |
| 12 | Model build | EfficientNet-B4 + freeze + BN unfreeze, no CBAM |
| 14 | Loss/optimizer | Single LR Adam, BCE+Dice loss |
| 26 | Discussion | Encoder ablation analysis |

### Unchanged Cells

Cells 3-7, 9-11, 13, 15-25, 27 remain structurally identical to P.30.1 template.

### Risks

- EfficientNet-B4 with frozen body may have too few trainable BN params for meaningful adaptation
- Smaller batch size (8 vs 16) may destabilize BatchNorm running statistics
- SE attention in frozen encoder cannot adapt to ELA domain

### Verification Checklist

- [ ] Model loads EfficientNet-B4 pretrained weights
- [ ] Only BN layers + decoder are trainable
- [ ] BATCH_SIZE=8 fits within T4/P100 GPU memory
- [ ] Training loss decreases over first 3 epochs
- [ ] No NaN/Inf in loss
- [ ] Model checkpoint saves with encoder='efficientnet-b4' config
