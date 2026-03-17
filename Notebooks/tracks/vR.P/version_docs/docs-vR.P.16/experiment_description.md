# vR.P.16 — DCT Spatial Map Baseline

## Experiment Description

**Version:** vR.P.16
**Track:** Pretrained Localization (Track 2)
**Parent:** vR.P.3 (ELA input, frozen body + BN unfrozen)
**Single Variable Changed:** Input representation — replace ELA with DCT spatial feature maps

### Hypothesis

Blockwise DCT statistics (energy distribution of AC coefficients per 8x8 block) provide a spatially-resolved compression artifact signal that is complementary to ELA. JPEG compression quantizes DCT coefficients using a quality-dependent quantization table — tampered regions that have been through different compression pipelines will exhibit different DCT coefficient distributions than authentic regions.

### Pipeline

```
Raw Image
    |
    v
Convert to YCbCr → Extract Y (luminance) channel
    |
    v
Split into 8x8 blocks → Apply cv2.dct() to each block
    |
    v
Compute per-block statistics:
  - Ch0: AC energy (sum of squared AC coefficients)
  - Ch1: DC coefficient (block mean luminance)
  - Ch2: High-frequency energy (bottom-right 4x4 quadrant)
    |
    v
Spatial feature map (48x48 for 384px input) → Bilinear upsample to 384x384
    |
    v
Normalize (computed from training set statistics)
    |
    v
UNet (frozen ResNet-34 + BN unfrozen) → 384x384 binary mask
    |
    v
Pixel F1, IoU, AUC + Image-level accuracy
```

### Architecture

Same as vR.P.3: UNet with frozen ResNet-34 encoder, BN layers unfrozen for domain adaptation. IN_CHANNELS=3 (3-channel DCT spatial map). ~3.17M trainable parameters.

### Dataset

CASIA v2.0 (12,614 images: 7,491 Au + 5,123 Tp). 70/15/15 train/val/test split, stratified, seed=42.

### What Changes from P.3

| Aspect | P.3 | P.16 |
|--------|-----|------|
| Input | ELA (Q=90) RGB | DCT spatial map (AC energy / DC / HF energy) |
| Preprocessing | JPEG recompress + pixel diff + brightness scale | YCbCr conversion + blockwise DCT + statistics |
| IN_CHANNELS | 3 | 3 |
| Everything else | Same | Same |

### What Does NOT Change

- Architecture (UNet + ResNet-34)
- Freeze strategy (frozen body + BN unfrozen)
- Loss (BCE + Dice)
- Optimizer (Adam, lr=1e-3)
- Training config (25 epochs, patience=7)
- Data split (70/15/15, seed=42)
- Image size (384x384)
