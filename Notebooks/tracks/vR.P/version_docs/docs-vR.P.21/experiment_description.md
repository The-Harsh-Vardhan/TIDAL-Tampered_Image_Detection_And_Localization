# vR.P.21 -- Experiment Description

## ELA Residual Learning (Laplacian High-Pass Residuals)

### Hypothesis

High-frequency residuals extracted from ELA maps via Laplacian filtering emphasize manipulation boundary artifacts while suppressing smooth-area noise, producing a sharper forensic signal that improves tampered region boundary localization.

### Motivation

ELA maps contain both forensic signal (manipulation boundaries) and noise (general compression artifacts across the entire image). A Laplacian high-pass filter extracts edge-like structures from ELA, effectively computing: Residual = ELA - GaussianBlur(ELA). This removes the low-frequency "baseline" compression signal and isolates the high-frequency anomalies at tampering boundaries.

This is inspired by residual learning approaches in image forensics (e.g., BayarConv, SRM filters) where filtering raw inputs to extract noise patterns has been shown to dramatically improve forgery detection.

### Single Variable Changed from vR.P.3

**Input representation** -- Replace raw ELA with Laplacian-filtered ELA residuals. Architecture unchanged.

### Key Configuration

| Parameter | P.3 (parent) | P.21 (this) |
|-----------|-------------|-------------|
| ELA input | Q=90 RGB (raw) | Q=90 Laplacian residual (high-pass) |
| Preprocessing | ELA -> brightness scale -> resize | ELA -> Laplacian filter -> clip -> scale -> resize |
| IN_CHANNELS | 3 | 3 |
| Everything else | Same | Same |

### Pipeline

```
Image -> ELA(Q=90) RGB
    -> cv2.Laplacian(ELA, cv2.CV_64F, ksize=3)
    -> np.abs(residual)  # absolute value of Laplacian
    -> clip to [0, 255], normalize
    -> 3ch input -> UNet -> mask
```

### Expected Impact

+2-4pp Pixel F1. Sharpened boundary signal should improve recall at tampered region edges without sacrificing precision in clean areas.

### Risk

Laplacian amplifies noise. If ELA noise floor is high relative to forensic signal, residual map quality degrades. May need to tune kernel size.
