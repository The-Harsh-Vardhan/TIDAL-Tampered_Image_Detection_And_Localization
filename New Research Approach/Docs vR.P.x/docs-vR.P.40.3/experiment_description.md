# vR.P.40.3 -- Experiment Description

## InceptionV1 Custom Encoder + Multi-Q RGB ELA (9ch)

### Hypothesis

A custom InceptionV1 encoder with parallel multi-scale branches (1x1, 3x3, 5x5, pool) is architecturally well-suited for ELA forensic analysis because tamper artifacts exist at multiple spatial scales. Training from scratch on forensic data may learn domain-specific features that pretrained ImageNet encoders cannot capture.

### Motivation

All prior experiments used ImageNet-pretrained encoders (ResNet-34, EfficientNet-B4). While transfer learning is powerful, ImageNet features are optimized for natural image classification, not forensic artifact detection. ELA maps have fundamentally different statistical properties (sparse high-frequency edges vs dense textures). A custom encoder designed for multi-scale forensic feature extraction, trained from scratch on CASIA2, could potentially learn more task-relevant features.

### Architecture: InceptionV1 Module

Classic GoogLeNet-style parallel branches:
- **1x1 branch**: Captures pixel-level artifacts
- **3x3 branch**: Captures local edge discontinuities (with 1x1 reduction)
- **5x5 branch**: Captures larger block-level artifacts (with 1x1 reduction)
- **MaxPool branch**: Preserves strongest local features

No BatchNorm (original 2014 Inception design). MaxPool in pool branch.

### Key Configuration

| Parameter | P.40.2 (parent) | P.40.3 (this) |
|-----------|-----------------|---------------|
| Encoder | EfficientNet-B4 (pretrained) | InceptionV1 Custom (from scratch) |
| Input | Multi-Q RGB ELA (9ch) | Multi-Q RGB ELA (9ch) |
| Encoder stages | [3, 24, 32, 56, 160, 448] | [9, 32, 128, 240, 336, 432] |
| Pretrained | ImageNet | None (trained from scratch) |
| Trainable ratio | ~15% (BN only) | 100% |

### Pipeline

```
Image -> Multi-Q RGB ELA (9ch, Q=75/85/95)
      -> InceptionV1 Encoder (5 stages, all trainable)
         Stage 0: Stem Conv 3x3 stride 2 -> 32ch
         Stage 1: InceptionV1 + 1x1 proj -> 128ch + MaxPool
         Stage 2: InceptionV1 + 1x1 proj -> 240ch + MaxPool
         Stage 3: InceptionV1 + 1x1 proj -> 336ch + MaxPool
         Stage 4: InceptionV1 + 1x1 proj -> 432ch + MaxPool
      -> UNet Decoder (skip connections)
      -> Sigmoid -> 384x384 binary mask
```

### Expected Impact

Likely lower than pretrained encoders (limited training data), but establishes baseline for from-scratch Inception approaches. Expected F1: 0.55–0.70.

### Risk

Training from scratch on ~12K images is data-limited. No BN makes training potentially unstable. May require careful LR tuning.
