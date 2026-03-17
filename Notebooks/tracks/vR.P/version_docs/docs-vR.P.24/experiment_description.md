# vR.P.24 -- Experiment Description

## Noiseprint Forensic Features

### Hypothesis

Camera-model-specific noise patterns (Noiseprint) extracted via a pretrained DnCNN denoiser provide a forgery detection signal that is fundamentally different from ELA. Authentic regions share consistent noise patterns from a single camera pipeline, while tampered regions exhibit noise inconsistencies from the source image's different camera or processing chain.

### Motivation

Noiseprint (Cozzolino & Verdoliva, 2020) extracts the camera-model fingerprint by computing: Noiseprint = Image - DnCNN(Image). The residual captures the noise pattern characteristic of the imaging pipeline. Spliced regions from a different camera/processing chain produce a different noise pattern, creating a detectable boundary.

This is state-of-the-art in image forensics and provides complementary information to ELA (which captures compression artifacts, not camera noise).

### Single Variable Changed from vR.P.3

**Input representation** -- Replace ELA with Noiseprint residual maps.

### Key Configuration

| Parameter | P.3 (parent) | P.24 (this) |
|-----------|-------------|-------------|
| Input | ELA (Q=90) RGB | Noiseprint residual (3ch, from DnCNN) |
| Preprocessing | JPEG resave + diff | DnCNN denoiser residual extraction |
| IN_CHANNELS | 3 | 3 |
| Additional model | None | Pretrained DnCNN (frozen, inference only) |
| Everything else | Same | Same |

### Pipeline

```
Image (RGB) -> Pretrained DnCNN denoiser -> Denoised image
    -> Noiseprint = Image - Denoised
    -> 3ch residual map -> normalize -> UNet -> mask
```

### Implementation Notes

- Use pretrained DnCNN weights (publicly available, e.g., from Zhang et al. 2017 or Cozzolino 2020)
- DnCNN runs in inference mode only (frozen, no gradients)
- Noiseprint extraction adds ~0.1s per image overhead
- May need to train a simple DnCNN on CASIA images if pretrained weights are not compatible

### Expected Impact

+2-6pp Pixel F1 (if noise patterns are discriminative for CASIA tamperings). Highly dependent on whether CASIA images preserve enough noise signal.

### Risk

CASIA images are heavily JPEG compressed, which may destroy camera noise patterns. Noiseprint works best on lightly compressed or RAW images. If the noise floor is dominated by JPEG artifacts rather than camera noise, the signal will be weak.
