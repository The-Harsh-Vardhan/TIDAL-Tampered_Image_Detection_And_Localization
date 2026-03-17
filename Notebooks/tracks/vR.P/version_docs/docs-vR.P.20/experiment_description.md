# vR.P.20 -- Experiment Description

## ELA Magnitude Channel

### Hypothesis

The scalar magnitude (L2 norm) of ELA difference vectors across RGB channels provides a rotation-invariant forensic signal that is more robust than individual channel values. A single magnitude channel combined with two directional channels (e.g., chrominance ratios) gives 3 complementary channels emphasizing different artifact properties.

### Motivation

Standard ELA produces 3 correlated RGB channels. The magnitude (sqrt(R^2 + G^2 + B^2)) captures the total error energy regardless of color direction, while the ratio channels (R/mag, G/mag) capture the chrominance direction of the error. This decomposition separates "how much error" from "what kind of error" -- the magnitude is more robust to color shifts while the direction reveals compression artifact patterns.

### Single Variable Changed from vR.P.3

**Input representation** -- Replace 3-channel RGB ELA with 3-channel (Magnitude, ChromaDir1, ChromaDir2) decomposition. Architecture unchanged (still 3 channels).

### Key Configuration

| Parameter | P.3 (parent) | P.20 (this) |
|-----------|-------------|-------------|
| ELA input | Q=90 RGB (R, G, B) | Q=90 Decomposed (Magnitude, ChromaDir1, ChromaDir2) |
| IN_CHANNELS | 3 | 3 |
| Encoder | ResNet-34, frozen+BN | ResNet-34, frozen+BN (no change) |
| Everything else | Same | Same |

### Pipeline

```
Image -> ELA(Q=90) -> RGB channels (R, G, B)
    -> Magnitude: sqrt(R^2 + G^2 + B^2)
    -> ChromaDir1: R / (Magnitude + eps)
    -> ChromaDir2: G / (Magnitude + eps)
    -> Stack (Mag, CD1, CD2) -> 3ch input -> UNet -> mask
```

### Expected Impact

+1-3pp Pixel F1. Magnitude channel provides cleaner signal for thresholding; chrominance directions reveal splice boundary artifacts.

### Risk

Chrominance ratios may be noisy in low-magnitude regions (near-zero ELA). Epsilon term prevents division by zero but noisy ratios in "clean" areas may confuse the model.
