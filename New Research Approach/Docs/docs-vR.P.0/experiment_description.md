# Experiment Description: vR.P.0 — ResNet-34 UNet Baseline

---

## Version Info

| Field | Value |
|-------|-------|
| Version | vR.P.0 |
| Track | Pretrained Localization (Track 2) |
| Parent | New track (informed by ETASR ablation vR.1.x) |
| Change | First pretrained experiment — ResNet-34 UNet, frozen encoder, RGB input |
| Category | Architecture / Localization |
| Weaknesses Fixed | W13 (no localization), W15 (no pretrained encoder), W17 (ETASR cannot localize) |

---

## Change Description

Introduce a completely new model architecture for image tampering **localization** using a pretrained encoder-decoder:

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation=None  # sigmoid applied in loss/postprocessing
)

# Freeze encoder — only train decoder (~500K params)
for param in model.encoder.parameters():
    param.requires_grad = False
```

This replaces the ETASR 2-layer CNN (classification only, 29.5M fully trainable params) with:
- **ResNet-34 encoder** (ImageNet pretrained, frozen) — provides rich hierarchical visual features
- **UNet decoder** (trainable, ~500K params) — reconstructs pixel-level tampered region masks
- **RGB input at 384×384** — 3× more detail than ETASR's 128×128 ELA

---

## Motivation

### Why This Change

The ETASR CNN (vR.1.x track) has a **fundamental architectural limitation**: it outputs a single binary label (Authentic/Tampered) and **cannot produce pixel-level localization masks**. The assignment explicitly requires:

> "Train a model to predict tampered regions."
> "Visual results: Original / Ground Truth / Predicted / Overlay"

No amount of ablation can make the ETASR CNN localize — it was designed for classification, not segmentation. A pretrained encoder-decoder is the only viable path to localization.

### Why ResNet-34

1. **Proven in this project:** v6.5 (SMP UNet + ResNet-34, ImageNet) achieved Tam-F1 = 0.41 for pixel-level segmentation
2. **Literature support:** Common in forensic detection (19 of 21 surveyed papers use pretrained encoders)
3. **Data efficiency:** Only ~500K trainable params vs 29.5M (60× better ratio for 8,829 training images)
4. **T4 compatible:** ~3 GB GPU memory at batch=16, 384×384

### Why Frozen Encoder

- With only 8,829 training images, unfreezing 21.3M encoder params risks catastrophic overfitting
- Frozen weights act as a regularizer — the decoder learns to *use* stable features
- This is what v6.5 used (differential LR: encoder 1e-4, decoder 1e-3)
- Unfreezing will be tested in vR.P.1 (gradual unfreeze)

### Why RGB (Not ELA)

- ImageNet features transfer perfectly to natural RGB images (same distribution)
- ELA maps have fundamentally different statistics → BatchNorm domain shift
- v6.5 achieved Tam-F1 = 0.41 with RGB input
- ELA input will be tested as an ablation in vR.P.2

---

## What Changes

### From ETASR Track (vR.1.x)

| Aspect | ETASR (vR.1.x) | vR.P.0 (This Version) |
|--------|----------------|----------------------|
| Framework | TensorFlow/Keras | **PyTorch + SMP** |
| Architecture | 2-layer CNN (Sequential) | **UNet + ResNet-34 (encoder-decoder)** |
| Input | ELA maps (128×128) | **RGB images (384×384)** |
| Normalization | [0, 1] scaling | **ImageNet mean/std** |
| Output | Binary class label [P(Au), P(Tp)] | **384×384 pixel mask** |
| Encoder | From scratch (all 29.5M params) | **ImageNet pretrained (frozen)** |
| Trainable params | 29,520,034 | **~500,000** |
| Loss | categorical_crossentropy | **BCEDiceLoss** |
| Optimizer | Adam (lr=1e-4, all params) | **Adam (lr=1e-3, decoder only)** |
| Batch size | 32 | **16** (larger images) |
| Early stopping | val_accuracy, patience=5 | **val_loss, patience=7** |
| Localization | Impossible | **Native** |
| Metrics | Classification only | **Pixel F1, IoU, Dice, AUC + Classification** |

### Configuration (Frozen)

| Parameter | Value |
|-----------|-------|
| Dataset | CASIA v2.0 (same as ETASR track) |
| Random seed | 42 |
| Data split | 70/15/15 (stratified) |
| Image size | 384×384 |
| Encoder | ResNet-34 (ImageNet, frozen) |
| Decoder | UNet (SMP default) |
| Loss | SoftBCEWithLogitsLoss + DiceLoss |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | 25 |
| Early stopping | val_loss, patience=7 |

---

## Ground Truth Masks

CASIA v2.0 ground truth mask handling:

- **Authentic images:** All-zero mask (no tampering) — always correct
- **Tampered images with GT:** Binary mask from CASIA ground truth directory
- **Tampered images without GT:** ELA-based pseudo-mask (adaptive thresholding)

The notebook auto-detects GT mask availability and falls back to ELA pseudo-masks if needed.
