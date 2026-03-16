# vR.P.40.1 -- Experiment Description

## EfficientNet-B4 Baseline (ELA Q=90, 3ch)

### Hypothesis

A larger, more capable encoder (EfficientNet-B4) with built-in Squeeze-and-Excitation attention will extract richer forensic features from ELA maps than ResNet-34, even with the same single-quality ELA input. The compound scaling of EfficientNet (depth + width + resolution) should capture multi-scale tamper artifacts more effectively.

### Motivation

All prior experiments (P.3 through P.30.1) used ResNet-34 as the encoder backbone. While input pipeline improvements (Multi-Q ELA, CBAM) yielded gains, the encoder capacity has never been varied. EfficientNet-B4 has ~19M parameters vs ResNet-34's ~21M, but uses them more efficiently via mobile inverted bottleneck blocks with SE attention. This experiment isolates the encoder contribution.

### Single Variable Changed from vR.P.3

**Encoder backbone** -- Replace ResNet-34 (21.3M params) with EfficientNet-B4 (19.3M params). Same single-quality ELA Q=90 RGB input, same frozen body + BN unfrozen strategy, same training pipeline.

### Key Configuration

| Parameter | P.3 (baseline) | P.40.1 (this) |
|-----------|----------------|---------------|
| Encoder | ResNet-34 | EfficientNet-B4 |
| Input | ELA Q=90 (3ch RGB) | ELA Q=90 (3ch RGB) |
| IN_CHANNELS | 3 | 3 |
| Pretrained | ImageNet | ImageNet |
| Freeze strategy | Frozen body + BN unfrozen | Frozen body + BN unfrozen |
| BATCH_SIZE | 16 | 8 (larger model) |

### Pipeline

```
Image -> ELA(Q=90) RGB -> 3ch input (384x384)
      -> EfficientNet-B4 Encoder (frozen + BN unfrozen)
      -> UNet Decoder (5 stages)
      -> Sigmoid -> 384x384 binary mask
```

### Expected Impact

+2-5pp Pixel F1 over P.3 (0.6920) from encoder capacity alone.

### Risk

EfficientNet-B4 requires BATCH_SIZE=8 (vs 16) due to higher memory usage, which may affect gradient stability.
