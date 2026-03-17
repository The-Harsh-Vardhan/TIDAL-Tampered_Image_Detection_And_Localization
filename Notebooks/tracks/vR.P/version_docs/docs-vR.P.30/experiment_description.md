# vR.P.30 -- Experiment Description

| Field | Value |
|-------|-------|
| **Version** | vR.P.30 |
| **Change** | Multi-quality ELA + CBAM attention (25ep, BCE+Dice) |
| **Parent** | vR.P.15 |
| **Single Variable** | Add CBAM attention (Channel + Spatial) to all 5 UNet decoder blocks |
| **Track** | Pretrained Localization (Track 2) |

---

## Hypothesis

Combining multi-quality ELA input (from P.15, +4.09pp F1) with CBAM attention (from P.10, +3.54pp F1 isolated) should produce an additive improvement because these techniques operate on different parts of the pipeline:
- **Multi-Q ELA** improves WHAT the model sees (input representation)
- **CBAM** improves WHERE the decoder focuses (attention mechanism)

## Motivation

P.15 and P.10 are the two most impactful single-variable improvements in the ablation study. Their combination has never been tested.

## Configuration

| Parameter | Value |
|-----------|-------|
| Input | Multi-Q ELA (Q=75/85/95) |
| Attention | CBAM (reduction=16, kernel=7) |
| Loss | BCE+Dice |
| Epochs | 25 |
| Encoder | ResNet-34 (frozen+BN) |

## What DID NOT Change
- Dataset: CASIA v2.0 with GT masks
- Encoder: ResNet-34 (ImageNet pretrained)
- Data split: 70/15/15 stratified, seed=42
- Image size: 384x384
- Batch size: 16
