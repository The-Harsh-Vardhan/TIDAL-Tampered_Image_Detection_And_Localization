# vR.P.40.4 -- Experiment Description

## InceptionV2 Custom Encoder + Multi-Q RGB ELA (9ch)

### Hypothesis

InceptionV2 improvements (factorized 5x5 convolutions, BatchNorm, AvgPool) over InceptionV1 will provide better training stability and parameter efficiency when training from scratch on forensic data.

### Motivation

P.40.3 tests InceptionV1 without BatchNorm, which may struggle with training stability on small datasets. InceptionV2 adds three key improvements: (1) factorized 5x5 into two 3x3 (same receptive field, 28% fewer params), (2) BatchNorm throughout (critical for from-scratch training), (3) AvgPool instead of MaxPool (smoother gradients for segmentation).

### Single Variable Changed from vR.P.40.3

**Inception module version** -- Replace InceptionV1 (no BN, MaxPool, 5x5 conv) with InceptionV2 (BN, AvgPool, factorized 5x5→two 3x3). Same encoder stage dimensions, same input pipeline.

### Key Configuration

| Parameter | P.40.3 (parent) | P.40.4 (this) |
|-----------|-----------------|---------------|
| Inception version | V1 (no BN, MaxPool) | V2 (BN, AvgPool, factorized 5x5) |
| Input | Multi-Q RGB ELA (9ch) | Multi-Q RGB ELA (9ch) |
| Encoder stages | [9, 32, 128, 240, 336, 432] | [9, 32, 128, 240, 336, 432] |
| Pretrained | None | None |
| BatchNorm | No | Yes |
| Pool type | MaxPool | AvgPool |

### Pipeline

```
Image -> Multi-Q RGB ELA (9ch, Q=75/85/95)
      -> InceptionV2 Encoder (5 stages, BN, AvgPool)
      -> UNet Decoder (skip connections)
      -> Sigmoid -> 384x384 binary mask
```

### Expected Impact

+5-10pp over P.40.3 from BatchNorm alone. Expected F1: 0.60–0.72.

### Risk

Still limited by training data size. Factorized convolutions may not capture the same features as full 5x5 kernels for forensic artifacts.
