# 07 — Shortcut Learning Risk Assessment

## Purpose

Analyze the evidence for shortcut learning in Run01, quantify the risk, and define mitigations for v8.

---

## What Is Shortcut Learning?

Shortcut learning occurs when a model exploits dataset-specific artifacts or statistical regularities instead of learning the intended task features. In image forensics, common shortcuts include:

- **JPEG compression artifacts** at tampered boundaries (different quality factors in pasted vs original regions)
- **Noise distribution inconsistencies** between authentic and tampered regions
- **Color/brightness discontinuities** at paste boundaries
- **Dataset-specific patterns** (file naming, resolution, camera characteristics)

A model using shortcuts will perform well in-distribution but collapse under distribution shift — which is exactly what Run01 shows.

---

## Evidence From Run01

### Evidence 1: The Robustness Plateau

Four degradation conditions produce nearly identical F1 scores:

| Condition | F1 | Δ from clean |
|---|---|---|
| JPEG QF70 | 0.5912 | −0.1296 |
| JPEG QF50 | 0.5938 | −0.1269 |
| Gaussian noise (light) | 0.5938 | −0.1270 |
| Gaussian noise (heavy) | 0.5938 | −0.1270 |

**Interpretation:** The model has two operating modes:
1. **Clean input mode:** Uses full feature set including compression/noise artifacts → F1≈0.72
2. **Degraded input mode:** Artifact features destroyed → falls back to structural features → F1≈0.59

The ~0.13 difference (0.72 − 0.59) represents **performance attributable to artifacts** rather than genuine forensic understanding.

This means: **approximately 13% of Run01's mixed-set performance comes from shortcut features.**

### Evidence 2: Severity Independence

- JPEG QF50 (heavy compression) produces the **same** F1 as QF70 (moderate compression)
- Light noise and heavy noise produce **identical** F1 (0.5938 vs 0.5938)

If the model were using noise-level-sensitive features, heavier degradation should cause proportionally worse performance. The flat response suggests a binary feature regime: artifacts present → one behavior, artifacts absent → different behavior.

### Evidence 3: Copy-Move Failure

Copy-move forgeries duplicate content from the same image. This means:
- Source and target regions have identical camera noise, JPEG quality, and color statistics
- The only differentiating signal is boundary artifacts (re-compression, interpolation)
- These boundary artifacts are exactly the shortcuts that JPEG/noise degradation destroys

Copy-move F1=0.31 (vs splicing F1=0.59) is consistent with the model relying on inter-region statistical differences that don't exist in copy-move.

### Evidence 4: Low Threshold

The optimal threshold of 0.1327 means the model outputs low probabilities for tampered pixels. This could indicate:
- The model is "uncertain" about its forensic features (consistent with partial shortcut reliance)
- Strong shortcut features (when present) push probabilities above 0.13
- Genuine forensic features alone produce probabilities near or below the threshold

### Evidence 5: RGB-Only Architecture

The model receives only RGB channels. Forensic shortcuts in CASIA are primarily visible as:
- JPEG block boundary artifacts (8×8 grid discontinuities)
- Statistical noise inconsistencies between regions
- Slight color shifts from different post-processing pipelines

These are all low-level signals that happen to be partially visible in RGB but would be much stronger in frequency domain, noise residual, or error level analysis representations.

---

## Risk Quantification

### Estimated Feature Budget

Based on the robustness evidence, Run01's mixed-set F1=0.72 breaks down approximately as:

| Feature Category | Estimated F1 Contribution | Evidence |
|---|---|---|
| Structural/semantic features | ~0.59 (82%) | F1 floor under all degradations |
| Compression/noise artifacts | ~0.13 (18%) | F1 drop under degradation |

For tampered-only F1=0.41:

| Feature Category | Estimated F1 Contribution | Evidence |
|---|---|---|
| Structural features on tampered | ~0.28 | Extrapolating 0.59×(0.41/0.72) |
| Artifact features on tampered | ~0.13 | Same absolute artifact contribution |

**The artifact contribution is proportionally larger for tampered-only metrics (~32% vs ~18%).** This makes sense — authentic images don't need artifact features to score F1=1.0.

### Severity Assessment

| Factor | Rating | Justification |
|---|---|---|
| Degree of shortcut reliance | **MODERATE** | ~13–18% of performance from artifacts |
| Impact on deployment | **HIGH** | Any real-world pipeline involves JPEG re-encoding |
| Impact on evaluation credibility | **MEDIUM** | Clean metrics are inflated but not fabricated |
| Detectability | **HIGH** | Robustness suite clearly reveals it |

---

## Mitigations for v8

### P0: Training-Time Mitigation

**JPEG Compression Augmentation**
```python
A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3)
```
Train with random JPEG compression so the model cannot rely on compression artifacts. This directly addresses the biggest identified shortcut.

**Gaussian Noise Augmentation**
```python
A.GaussNoise(var_limit=(10, 50), p=0.3)
```
Destroy noise-distribution shortcuts during training.

### P1: Evaluation-Time Detection

**Artifact Attribution Test**

After v8 training, compare:
1. Clean test F1
2. JPEG QF50 test F1
3. The gap should be <0.05 (vs Run01's 0.13)

If the gap persists after augmentation, the model is still learning shortcuts — consider architecture changes (forensic streams).

**Feature Ablation**

Apply aggressive JPEG compression (QF30) + noise addition during evaluation to measure the "artifact-free" baseline. This is the model's true forensic capability.

### P2: Architecture-Level Mitigation (v9)

**Forensic Input Streams**

Add non-RGB channels that encode forensic signals explicitly:
- SRM (Spatial Rich Model) residuals
- ELA (Error Level Analysis) maps
- High-pass filter responses

These make the model less dependent on implicit RGB artifacts by providing explicit forensic evidence channels.

---

## Monitoring Plan

For every future training run, track these shortcut indicators:

| Indicator | Threshold | Action if exceeded |
|---|---|---|
| Clean F1 − JPEG QF50 F1 | >0.10 | Review augmentation pipeline |
| QF70 F1 ≈ QF50 F1 (within 0.01) | Yes | Model operating in binary mode — investigate |
| Light noise F1 ≈ Heavy noise F1 | Yes | Noise features are binary, not graded |
| Copy-move F1 < 0.40 | Yes | Forensic feature gap for copy-move |
| Optimal threshold < 0.20 | Yes | Probability calibration still broken |

---

## Bottom Line

Run01 shows **moderate shortcut reliance** (~13% of performance). This is not catastrophic — the model does learn some genuine structural features (F1≈0.59 under degradation). But the shortcuts inflate clean-condition metrics and would cause failure in any real-world pipeline that involves JPEG re-encoding or social media processing.

The fix is straightforward: add compression and noise augmentation during training. If this reduces the robustness gap to <5%, the shortcut risk is adequately mitigated for the current scope.
