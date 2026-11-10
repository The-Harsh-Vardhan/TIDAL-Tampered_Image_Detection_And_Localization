# DCT Ablation Study Plan

## Overview

This document describes three new experiments that extend the pretrained localization track with JPEG compression artifact analysis using Discrete Cosine Transform (DCT) features.

These experiments complement the existing ELA-based pipeline by exploring frequency-domain features, fusion approaches, and compression robustness.

---

## Experiment Roadmap

| ID | Name | Type | Parent | Single Variable |
|----|------|------|--------|-----------------|
| vR.P.16 | DCT Spatial Map Baseline | New input | P.3 | Replace ELA with DCT spatial feature maps |
| vR.P.17 | ELA + DCT Fusion | Fusion | P.3 | Replace 3ch ELA with 6ch ELA+DCT |
| vR.P.18 | Compression Robustness | Evaluation | P.3 | Test under Q=70/80/90/95 recompression |

---

## vR.P.16 — DCT Spatial Map Baseline

**Hypothesis:** JPEG compression artifacts provide a spatially-resolved forgery signal via blockwise DCT coefficient analysis.

**Method:** Extract 8x8 DCT blocks from the luminance channel, compute 3 per-block statistics (AC energy, DC value, HF energy), form a 3-channel spatial map, feed into UNet.

**Pipeline:**
```
Image → YCbCr → Y channel → 8x8 blocks → cv2.dct()
    → AC energy | DC coeff | HF energy
    → 3-channel map (48x48) → upsample to 384x384
    → UNet (frozen ResNet-34 + BN) → binary mask
```

**Expected Impact:** Pixel F1 0.40–0.65 (moderate — DCT maps are lower resolution than ELA)

---

## vR.P.17 — ELA + DCT Fusion

**Hypothesis:** Combining spatial (ELA) and frequency (DCT) artifact signals provides complementary information for forgery localization.

**Method:** Concatenate 3-channel ELA and 3-channel DCT as 6-channel input. Modify conv1 to accept 6 channels (duplicate pretrained weights, scale by 0.5).

**Pipeline:**
```
Image ─┬─ ELA (Q=90) → 3ch ELA map
       └─ DCT blocks → 3ch DCT map
              │
              v
       Concatenate → 6ch input → UNet (conv1 unfrozen) → binary mask
```

**Expected Impact:** Pixel F1 0.68–0.78 (may improve over P.3 if features are complementary)

---

## vR.P.18 — Compression Robustness Testing

**Hypothesis:** ELA-based detection degrades under additional JPEG compression; quantifying this degradation reveals practical deployment limits.

**Method:** Load P.3 model (no training). Evaluate test set under 5 conditions: Original, Q=95, Q=90, Q=80, Q=70.

**Pipeline:**
```
For each Q ∈ {Original, 95, 90, 80, 70}:
    Test image → JPEG recompress at Q → ELA (Q=90) → predict → measure
```

**Expected Impact:** Graceful degradation: <2pp drop at Q=95, 7–15pp at Q=80, 15–25pp at Q=70.

---

## Execution Order

1. **P.16 first** — establishes DCT feature quality baseline
2. **P.17 second** — informed by P.16's DCT signal strength
3. **P.18 last** — no training needed, fastest to execute

## Relation to Existing Work

- The reference notebook `casia-2-0-dataset-for-image-forgery-detecion-run-01.ipynb` attempted DCT with LSTM and achieved ~55% accuracy. That approach treated each block independently with no spatial context. Our approach preserves spatial information via feature maps.
- vR.P.4 (4-channel RGB+ELA) is the closest precedent for multi-channel fusion. It scored F1=0.7053 (NEUTRAL vs P.3). P.17 may follow a similar pattern.
- vK.10.6 included a basic robustness test (JPEG Q=50/70, noise, blur). P.18 provides a systematic, controlled version focused specifically on compression.
