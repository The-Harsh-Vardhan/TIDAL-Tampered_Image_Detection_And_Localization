# vR.P.40.2 -- Experiment Description

## EfficientNet-B4 + Multi-Q RGB ELA (9ch, Q=75/85/95)

### Hypothesis

Combining the best encoder (EfficientNet-B4) with the best input pipeline (Multi-Q RGB ELA 9ch) will produce the highest-performing model in the ablation series. EfficientNet-B4's built-in SE attention can naturally recalibrate cross-quality features, while 9 information-rich channels capture artifacts across the compression spectrum.

### Motivation

P.19 achieved F1=0.7965 with Multi-Q RGB ELA on ResNet-34. P.40.1 tests EfficientNet-B4 with single-Q ELA. This experiment combines both improvements to test whether gains stack. If F1 > 0.7965, the combination is synergistic; if F1 ≈ P.19, the encoder is not the bottleneck.

### Single Variable Changed from vR.P.40.1

**Input pipeline** -- Replace single-quality ELA Q=90 (3ch) with multi-quality RGB ELA Q=75/85/95 (9ch). Everything else identical to P.40.1.

### Key Configuration

| Parameter | P.40.1 (parent) | P.40.2 (this) |
|-----------|-----------------|---------------|
| Encoder | EfficientNet-B4 | EfficientNet-B4 |
| Input | ELA Q=90 (3ch RGB) | Multi-Q RGB ELA (9ch, Q=75/85/95) |
| IN_CHANNELS | 3 | 9 |
| Pretrained | ImageNet | ImageNet |
| BATCH_SIZE | 8 | 8 |

### Pipeline

```
Image -> ELA(Q=75) RGB -> 3ch
      -> ELA(Q=85) RGB -> 3ch  -> Concatenate -> 9ch input (384x384)
      -> ELA(Q=95) RGB -> 3ch
      -> EfficientNet-B4 Encoder (frozen + BN unfrozen)
      -> UNet Decoder -> Sigmoid -> 384x384 binary mask
```

### Expected Impact

Target: F1 > 0.7965 (beat P.19, the current best). Expected +1-3pp over P.40.1.

### Risk

9-channel input on frozen EfficientNet-B4 requires conv1 adaptation. The first conv layer must be modified to accept 9 channels, initialized from 3x tiling of pretrained 3-channel weights scaled by 1/3.
