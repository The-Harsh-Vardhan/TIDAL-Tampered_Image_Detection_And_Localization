# vR.P.22 -- Experiment Description

## SRM Noise Maps

### Hypothesis

Steganalysis Rich Model (SRM) noise residuals capture forensic noise patterns that are invisible in pixel-level or ELA representations. SRM filters detect inconsistencies in the noise floor across image regions -- tampered areas exhibit different noise characteristics from authentic areas because they have undergone different processing pipelines.

### Motivation

SRM filters were originally designed for steganalysis (detecting hidden data in images) but have been adopted in image forensics because they extract residual noise patterns that reveal processing history. The key insight: every image processing operation leaves a characteristic noise fingerprint. Spliced regions from different sources have different noise fingerprints.

Three SRM high-pass filters are used to capture different noise patterns:
1. **1st order edge filter** (3x3): Horizontal/vertical edge residuals
2. **2nd order edge filter** (3x3): Diagonal edge residuals
3. **3rd order filter** (5x5): Complex texture residuals

### Single Variable Changed from vR.P.3

**Input representation** -- Replace ELA with 3-channel SRM noise residual maps.

### Key Configuration

| Parameter | P.3 (parent) | P.22 (this) |
|-----------|-------------|-------------|
| Input | ELA (Q=90) RGB | SRM noise maps (3 filters) |
| Preprocessing | JPEG resave + diff + scale | High-pass SRM convolution on grayscale |
| IN_CHANNELS | 3 | 3 |
| Everything else | Same | Same |

### Pipeline

```
Image -> Grayscale
    -> SRM Filter 1 (1st order high-pass, 3x3) -> Ch0
    -> SRM Filter 2 (2nd order high-pass, 3x3) -> Ch1
    -> SRM Filter 3 (3rd order high-pass, 5x5) -> Ch2
    -> Stack -> 3ch -> normalize -> UNet -> mask
```

### SRM Kernels

```python
SRM_FILTER_1 = np.array([[ 0,  0,  0],
                          [ 0, -1,  1],
                          [ 0,  0,  0]], dtype=np.float32)

SRM_FILTER_2 = np.array([[ 0,  0,  0],
                          [ 0, -1,  0],
                          [ 0,  0,  1]], dtype=np.float32)

SRM_FILTER_3 = np.array([[-1,  2, -2,  2, -1],
                          [ 2, -6,  8, -6,  2],
                          [-2,  8, -12, 8, -2],
                          [ 2, -6,  8, -6,  2],
                          [-1,  2, -2,  2, -1]], dtype=np.float32) / 12.0
```

### Expected Impact

+1-4pp Pixel F1. SRM captures noise-level forensic features orthogonal to ELA's compression-artifact features.

### Risk

SRM noise patterns are subtle (low amplitude). May require careful normalization to prevent the encoder from ignoring the signal. ResNet-34's pretrained weights are tuned for natural images, not noise maps.
