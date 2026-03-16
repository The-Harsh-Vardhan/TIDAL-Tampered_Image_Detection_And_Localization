# vR.P.40.2 — Implementation Plan

## Core Implementation: EfficientNet-B4 + Multi-Q RGB ELA 9ch

### Model Build

1. Create UNet with `encoder_name='efficientnet-b4'`, `encoder_weights='imagenet'`, `in_channels=9`
2. SMP handles 9-channel adaptation internally (repeats pretrained conv1 weights)
3. Freeze ALL encoder parameters except BatchNorm
4. BATCH_SIZE=8 (same as P.40.1)

### Multi-Q RGB ELA Preprocessing (Same as P.19)

- `compute_ela_rgb(image_path, quality)` — ELA at given Q, returns RGB (H, W, 3)
- `compute_multi_quality_rgb_ela(image_path, qualities=[75,85,95])` — stack 3 RGB ELAs → (H, W, 9)
- `compute_mqela_statistics()` — per-channel mean/std for 9 channels from 500 training samples

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.40.2 — EfficientNet-B4 + Multi-Q RGB ELA (9ch)" |
| 2 | Setup | IN_CHANNELS=9, ELA_QUALITIES=[75,85,95] |
| 8 | Dataset class | Multi-Q RGB ELA 9ch (from P.19) |
| 9 | Splitting | 9-channel stats computation |
| 10 | Visualization | Show Q=75/Q=85/Q=95 RGB channels |
| 12 | Model build | EfficientNet-B4 with in_channels=9 |

### Key Comparison

| vs Experiment | Variable Isolated |
|---------------|-------------------|
| vs P.40.1 | Input pipeline (3ch → 9ch) on same encoder |
| vs P.19 | Encoder (ResNet-34 → EfficientNet-B4) on same input |
| vs P.30.1 | Both encoder and no CBAM (EfficientNet SE replaces CBAM) |

### Risks

- 9-channel conv1 initialization may still be suboptimal despite SMP's handling
- 3x preprocessing time per image
- Memory: 9ch × EfficientNet-B4 at batch=8 may approach GPU limits

### Verification Checklist

- [ ] DataLoader yields batches of shape (8, 9, 384, 384)
- [ ] ELA statistics have shape (9,) for mean and std
- [ ] Model accepts 9-channel input without errors
- [ ] Training loss decreases over first 3 epochs
- [ ] Compare pixel F1 with P.40.1 and P.19 baselines
