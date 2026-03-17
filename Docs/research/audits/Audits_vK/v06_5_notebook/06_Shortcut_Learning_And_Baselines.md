# Audit 6.5 — Part 6: Shortcut Learning Risk & Baseline Comparison

## Shortcut Learning Risk Assessment

### What is Shortcut Learning?

Shortcut learning occurs when a model relies on dataset artifacts (compression boundaries, resolution changes, color statistics) rather than genuine tampering signals. The model appears to perform well on the training distribution but fails on real-world images.

### Risk Factors for This Training Run

#### 1. JPEG Compression Artifacts — **HIGH RISK**

CASIA images are JPEG-compressed, and tampered regions often have different compression levels than the surrounding image. The model may learn to detect JPEG double-compression boundaries rather than actual content manipulation.

**Evidence from robustness testing:**
```
clean:       F1=0.7208
jpeg_qf70:   F1=0.5912  (Δ=-0.1296)
jpeg_qf50:   F1=0.5938  (Δ=-0.1269)
```

The **13% F1 drop** under JPEG recompression is a **strong signal that the model partially relies on compression artifacts**. If the model was learning true manipulation signals (edge inconsistencies, lighting mismatches), JPEG compression should have minimal effect. The nearly identical performance at QF50 and QF70 also suggests the model collapses to a baseline behavior once original compression artifacts are destroyed.

#### 2. Color/Texture Inconsistencies — **MEDIUM RISK**

Splicing operations often introduce subtle color temperature and noise pattern differences between the pasted region and host image. These are legitimate tampering signals, but if the model overfits to CASIA-specific color distributions (e.g., camera models), it may not generalize.

The absence of photometric augmentation (no ColorJitter, no Brightness/Contrast changes) increases this risk. The model has no training experience with varied color conditions.

#### 3. Mask Boundary Artifacts — **LOW-MEDIUM RISK**

CASIA ground truth masks may have anti-aliased borders or imprecise boundaries from manual annotation. The model could learn to detect mask creation artifacts rather than image manipulation artifacts. With the `> 0` binarization threshold, any anti-aliased mask edges are included as positive pixels.

**Mitigation:** The Dice loss is relatively robust to minor boundary noise. The 384×384 resize also smooths out fine boundary details.

#### 4. Image Resolution Patterns — **LOW RISK**

Evidence from robustness:
```
resize_0.75x: F1=0.6631  (Δ=-0.0576)
resize_0.5x:  F1=0.6134  (Δ=-0.1073)
```

The model degrades gracefully under resolution reduction, suggesting it's not heavily dependent on fine-grained pixel-level artifacts. The 6–10% drop is expected for any resolution-sensitive task.

#### 5. Dataset-Specific Patterns — **MEDIUM RISK**

CASIA is a well-known benchmark with known limitations:
- Limited diversity of forgery techniques
- All images from a small number of source cameras
- Possibly formulaic manipulation patterns

Without cross-dataset evaluation (e.g., training on CASIA, testing on Coverage or CoMoFoD), it's impossible to determine whether the model generalizes beyond CASIA.

---

## Robustness Analysis Detailed

| Condition | F1 | Delta | Interpretation |
|---|---|---|---|
| Clean | 0.7208 | — | Baseline |
| JPEG QF70 | 0.5912 | -0.1296 | **Compression artifact dependency** |
| JPEG QF50 | 0.5938 | -0.1269 | Same as QF70 — **suspicious plateau** |
| Gaussian noise (light) | 0.5938 | -0.1270 | **Identical to JPEG** — suspicious |
| Gaussian noise (heavy) | 0.5938 | -0.1270 | **Identical to light noise** — highly suspicious |
| Gaussian blur | 0.5881 | -0.1326 | Consistent degradation |
| Resize 0.75× | 0.6631 | -0.0576 | Moderate, expected |
| Resize 0.5× | 0.6134 | -0.1073 | Expected |

### Critical Observation

**Four degradation conditions (JPEG QF50, JPEG QF70, Gaussian noise light, Gaussian noise heavy) produce nearly identical F1 values (0.5912–0.5938).** This suggests the model degrades to a common baseline performance whenever the input distribution shifts. It's as if the model has two modes:

1. **Clean CASIA data:** Uses compression/texture artifacts → F1≈0.72
2. **Any degradation:** Falls back to coarser structural features → F1≈0.59

This pattern is consistent with partial shortcut learning. The ~0.13 F1 attributable to shortcuts is destroyed by any degradation, while the remaining ~0.59 F1 comes from genuine structural signals.

---

## Comparison to Expected Baselines

### Literature Results on CASIA

| Method | F1 (pixel) | Notes |
|---|---|---|
| U-Net (basic, various papers) | 0.35–0.55 | Depending on preprocessing and evaluation protocol |
| ManTraNet (CVPR 2019) | ~0.45–0.58 | Manipulation tracing network |
| SPAN (ECCV 2020) | ~0.50–0.60 | Spatial pyramid attention |
| MVSS-Net (ICCV 2021) | ~0.55–0.65 | Multi-view multi-scale |
| CAT-Net | ~0.60–0.70 | Compression artifact tracing |

### This Run's Position

- **Mixed-set F1 = 0.72:** Would place at the top of the table, but this is inflated by authentic images (see Part 2)
- **Tampered-only F1 = 0.41:** Below basic U-Net baselines
- **Splicing F1 = 0.59:** Within the expected range for U-Net on splicing
- **Copy-move F1 = 0.31:** Below expected — U-Net should achieve 0.35–0.45 on copy-move with proper training

### Assessment

The splicing performance is **competitive** with expected baselines. The copy-move performance is **below expected** and drags down the tampered-only average. The overall tampered-only F1 of 0.41 is **not suspiciously high** — it's genuinely moderate and consistent with a first-pass U-Net without advanced techniques.

The results are neither suspiciously high (no metric bugs) nor suspiciously low (model is learning something). The model sits in a realistic range with clear room for improvement.

---

## Verdict

**Shortcut learning is present but not dominant.** The robustness analysis provides strong evidence that ~15–18% of the model's clean performance comes from compression/texture artifacts. However, the remaining performance (F1≈0.59 under degradation) demonstrates genuine feature learning. The model is a legitimate baseline that would benefit from:

1. Augmentation with JPEG compression during training
2. SRM (Spatial Rich Model) noise filters as additional input channels
3. Cross-dataset evaluation to quantify generalization
