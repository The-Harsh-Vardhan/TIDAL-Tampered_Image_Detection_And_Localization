# 11 — Training Failure Cases

## Purpose

Catalog the specific failure modes observed in Run01, classify them by root cause, and define targeted mitigations for v8.

---

## Failure Mode Taxonomy

Run01 produced identifiable failure patterns that fall into five categories.

---

## Category 1: Complete Prediction Failure (F1=0.0)

### Description

The model produces no meaningful tampered-region prediction. The predicted mask is either entirely blank (all below threshold) or completely mislocalized.

### Run01 Evidence

- **10 worst images all have F1=0.0**
- 8/10 are copy-move forgeries
- 6/10 have tampered mask area <2%
- 2/10 are splicing with small tampered regions

### Root Cause Analysis

| Factor | Contribution |
|---|---|
| **Copy-move forgery type** | PRIMARY — the model cannot detect regions that duplicate existing content |
| **Small mask size** | AMPLIFYING — small regions produce weak gradient signal due to no pos_weight |
| **Low threshold (0.1327)** | CONTRIBUTING — even at this low threshold, the model assigns <13% probability to these regions |
| **RGB-only input** | STRUCTURAL — copy-move forensic signals are in noise/compression domain, not RGB |

### v8 Mitigation

| Action | Expected Impact | Priority |
|---|---|---|
| Add pos_weight to BCE | Increases gradient for small regions by 10–30× | P0 |
| Report mask-size-stratified F1 | Quantifies the problem precisely | P1 |
| Augmentation with compression | Reduces shortcut dependency | P1 |
| Forensic input streams (v9) | Adds the signal needed for copy-move | Deferred |

---

## Category 2: Copy-Move Systematic Failure

### Description

Copy-move forgeries consistently underperform across all evaluation conditions, not just in worst cases.

### Run01 Evidence

| Metric | Copy-move | Splicing | Ratio |
|---|---|---|---|
| F1 | 0.3105 | 0.5901 | 0.53× |
| Count | 495 | 274 | 1.81× |
| In worst-10 | 8/10 | 2/10 | — |

Copy-move accounts for 64% of tampered test images but produces barely half the F1 of splicing.

### Root Cause Analysis

Copy-move manipulations paste a region from the **same image** to another location. This means:
1. Source and target have identical camera noise profiles
2. Source and target have identical JPEG compression quality
3. Source and target have matching color temperature and lighting
4. The only forensic signal is at the paste boundary (re-encoding artifacts, interpolation residuals)

An RGB-only model at 384×384 resolution has almost no access to these signals. The model must rely on:
- Visible boundary discontinuities (often subtle)
- Semantic implausibility (repeated content)
- Any residual compression artifacts that survived dataset preparation

### v8 Mitigation

| Action | Expected Impact | Priority |
|---|---|---|
| Add pos_weight (amplifies small-region gradients) | Modest — helps detection but doesn't solve feature gap | P0 |
| Augmentation (compression, noise) | Modest — improves robustness but doesn't add forensic capability | P1 |
| Per-forgery-type loss tracking | Diagnoses whether copy-move converges or diverges | P1 |
| Forensic input (SRM/ELA) — v9 | HIGH — provides the actual signal the model needs | Deferred |

**Honest assessment:** v8 may improve copy-move F1 from 0.31 to ~0.35–0.40, but reaching splicing-level performance (0.59) likely requires architectural changes (forensic input streams).

---

## Category 3: Overfitting-Induced Degradation

### Description

The model's performance degrades after epoch 15 because constant learning rate + minimal augmentation causes memorization of training-set artifacts.

### Run01 Evidence

| Epoch | Val Loss | Val F1 | Status |
|---|---|---|---|
| 11 | 0.7739 | 0.7237 | Convergence zone |
| 15 | 0.8141 | 0.7289 | Best F1 but loss already rising |
| 20 | 1.0095 | 0.7028 | Clear overfitting |
| 25 | 1.2010 | 0.6766 | Severe — val loss 55% above minimum |

The model wasted 10 epochs (15→25) on overfitting. Training compute was 40% inefficient.

### Root Cause

1. **No LR scheduler:** Constant decoder LR=1e-3 overshoots after convergence
2. **Minimal augmentation:** Only geometric transforms, no photometric regularization
3. **Small dataset:** ~8,800 training images with limited diversity

### v8 Mitigation

| Action | Expected Impact | Priority |
|---|---|---|
| ReduceLROnPlateau (patience=3) | Reduces LR after epoch ~14 → extends useful training to 30+ | P0 |
| Expanded augmentation | Creates more diverse training distribution → delays memorization | P1 |
| LR warmup (2 epochs) | Prevents early disruption of pretrained weights | P1 |

---

## Category 4: Robustness Collapse

### Description

Under input degradation (JPEG, noise, blur), the model collapses to a fixed baseline performance rather than degrading gracefully.

### Run01 Evidence

| Condition | F1 | Δ |
|---|---|---|
| Clean | 0.7208 | — |
| JPEG QF70 | 0.5912 | −0.1296 |
| JPEG QF50 | 0.5938 | −0.1269 |
| Noise (light) | 0.5938 | −0.1270 |
| Noise (heavy) | 0.5938 | −0.1270 |
| Blur | 0.5881 | −0.1326 |

Four conditions produce F1≈0.593 (±0.003). The model has a binary operating regime: artifacts present → full performance, artifacts destroyed → baseline.

### Root Cause

The model learned compression/noise artifacts as features. When these artifacts are destroyed by any degradation, the model falls back to structural features that produce F1≈0.59. See [07_Shortcut_Learning_Risk_Assessment.md](07_Shortcut_Learning_Risk_Assessment.md) for detailed analysis.

### v8 Mitigation

| Action | Expected Impact | Priority |
|---|---|---|
| JPEG compression augmentation | Forces model to not rely on compression artifacts | P1 |
| Gaussian noise augmentation | Forces model to not rely on noise patterns | P1 |
| Post-training robustness check | Validates mitigation worked (gap should be <0.05) | P1 |

---

## Category 5: Probability Calibration Failure

### Description

The model outputs poorly calibrated probabilities, requiring an abnormally low threshold (0.1327) to achieve best F1.

### Run01 Evidence

- Optimal threshold: 0.1327 (expected: 0.3–0.5)
- This means: a pixel is classified "tampered" if the model predicts >13% probability
- Implication: the model is highly uncertain about tampered pixels

### Root Cause

No `pos_weight` in BCE loss. With tampered pixels at ~2–10% of area, background pixels dominate the gradient 10–50×. The model learns to output low probabilities for everything because predicting "not tampered" is correct 90–98% of the time.

### v8 Mitigation

| Action | Expected Impact | Priority |
|---|---|---|
| Add pos_weight to BCE | Directly fixes calibration — expected threshold shift to 0.30–0.50 | P0 |
| Monitor threshold after training | If still <0.20, investigate further | P0 |

---

## Failure Interaction Map

The five failure categories are not independent:

```
Probability Calibration Failure (Cat 5)
    ↓ amplifies
Small-Region Detection Failure (Cat 1)
    ↓ concentrated in
Copy-Move Systematic Failure (Cat 2)

Overfitting (Cat 3) → Shortcut Learning → Robustness Collapse (Cat 4)
```

**Key insight:** Fixing Category 5 (pos_weight) partially fixes Category 1 (small regions). Fixing Category 3 (scheduler + augmentation) partially fixes Category 4 (robustness). But Category 2 (copy-move) requires both training fixes AND architectural changes to fully resolve.

---

## Expected v8 Failure Profile

After implementing all P0 and P1 fixes:

| Category | Run01 Status | Expected v8 Status |
|---|---|---|
| Complete failure (F1=0.0) | 10+ images | Reduced to 3–5 images (still mostly copy-move) |
| Copy-move F1 | 0.3105 | 0.35–0.42 (improved but not resolved) |
| Overfitting onset | Epoch 15 | Epoch 30+ |
| Robustness Δ | −0.13 | −0.03 to −0.06 |
| Optimal threshold | 0.1327 | 0.30–0.50 |

**Remaining gap after v8:** Copy-move will still underperform splicing because the fundamental signal gap (RGB-only) is not addressed until v9's forensic input streams.
