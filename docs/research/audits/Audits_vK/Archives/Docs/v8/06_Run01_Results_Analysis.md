# 06 — Run01 Results Analysis

## Purpose

Comprehensive analysis of every metric produced by Run01 (`v6-5-tampered-image-detection-localization-run-01.ipynb`), with interpretation of what each result means for v8 planning.

---

## Run Identity

| Property | Value |
|---|---|
| Notebook | v6-5-tampered-image-detection-localization-run-01.ipynb |
| Platform | Kaggle, 2× Tesla T4 (DataParallel) |
| Run date | March 11, 2026 |
| Training epochs | 25/50 (early stop triggered) |
| Best epoch | 15 |
| Best threshold | 0.1327 |
| Seed | 42 |

---

## 1. Training Curve Analysis

### Loss Trajectory

| Epoch | Train Loss | Val Loss | Gap | Status |
|---|---|---|---|---|
| 1 | 0.9534 | 0.8981 | -0.055 | Normal — val lower than train (regularization) |
| 5 | 0.7276 | 0.8236 | +0.096 | Transition — val starts exceeding train |
| 10 | 0.6704 | 0.7959 | +0.126 | Moderate gap |
| 11 | 0.6599 | **0.7739** | +0.114 | **Val loss minimum** |
| 15 | 0.6338 | 0.8141 | +0.180 | Val loss rising, but F1 still improving |
| 20 | 0.5706 | 1.0095 | +0.440 | Severe overfitting |
| 25 | 0.5149 | 1.2010 | +0.686 | Training terminated |

### Interpretation

1. **Convergence zone: epochs 10–15.** The model achieves useful representation by epoch 10 and peaks at 15.
2. **Loss-metric divergence at epoch 15.** Val loss was already rising (0.77→0.81) but val F1 was still 0.7289. This means the model was becoming more confident on correct predictions while making worse mistakes elsewhere — a classic overconfidence pattern.
3. **Wasted compute: epochs 15–25.** Ten epochs of patience were consumed while the model progressively overfit. A scheduler would have reduced LR at ~epoch 13, potentially extending useful training to epoch 30+.
4. **Train loss never plateaued** (0.95→0.51), confirming the model has sufficient capacity — the limitation is generalization, not fitting.

### Val F1 Trajectory

| Epoch | Val F1 | Δ from previous |
|---|---|---|
| 1 | 0.5028 | — |
| 3 | 0.5949 | +0.092 |
| 5 | 0.6600 | +0.065 |
| 8 | 0.7039 | +0.044 |
| 11 | 0.7237 | +0.020 |
| 15 | **0.7289** | +0.005 |
| 20 | 0.7028 | −0.026 |
| 25 | 0.6766 | −0.026 |

F1 growth rate was decelerating sharply by epoch 11 (+0.02/epoch → +0.005/epoch). A scheduler reducing LR here could have maintained incremental improvement.

---

## 2. Test Metrics Deep Dive

### Primary Results (threshold=0.1327)

| Metric | Mixed (1893) | Tampered-only (769) | Interpretation |
|---|---|---|---|
| Pixel-F1 | 0.7208 ± 0.4158 | 0.4101 ± 0.4148 | 0.31 gap = authentic inflation |
| Pixel-IoU | 0.6989 ± 0.4194 | 0.3563 ± 0.3798 | Consistent with F1 scaling |
| Precision | 0.7455 | — | Higher than recall → conservative |
| Recall | 0.7634 | — | Slightly lower than precision |

**Standard deviation analysis:** σ ≈ 0.41 on both metrics means the distribution is bimodal:
- Many images score near 1.0 (authentic images and well-predicted tampered images)
- Many images score near 0.0 (failed predictions)
- Very few images score in the 0.3–0.7 range

This is **not** a Gaussian distribution — it's a success/failure distribution.

### Per-Forgery-Type Analysis

| Type | Count | F1 | σ | Fraction of tampered |
|---|---|---|---|---|
| Splicing | 274 | 0.5901 | 0.3850 | 36% |
| Copy-move | 495 | 0.3105 | 0.3968 | 64% |

**Critical finding:** Copy-move is both the majority class (64%) and the worst-performing class. This means:
- Overall tampered-only F1 (0.41) is dragged down by copy-move's 0.31
- If copy-move alone were fixed to match splicing (0.59), overall tampered-only F1 would reach ~0.52
- Copy-move improvement is the highest-leverage single improvement available

**Why copy-move fails:**
- Copy-move duplicates existing content from the same image
- RGB features cannot easily distinguish pasted content from identical source content
- The model would need to detect subtle artifacts: boundary discontinuities, compression re-encoding, noise inconsistency
- These signals are weak or absent in RGB at 384×384 resolution

### Image-Level Detection

| Metric | Value | Method |
|---|---|---|
| Accuracy | 0.8246 | max(prob_map) > threshold |
| AUC-ROC | 0.8703 | max(prob_map) as continuous score |

**Interpretation:**
- AUC=0.87 is decent — the heuristic provides useful ranking
- Accuracy=0.82 means ~18% of images are misclassified
- The heuristic method (max pixel probability) is sensitive to single hot pixels — a single high-confidence false positive triggers a tampered classification
- A learned classification head would likely outperform this

---

## 3. Robustness Analysis

| Condition | F1 | Δ from clean | Category |
|---|---|---|---|
| Clean | 0.7208 | — | Baseline |
| JPEG QF70 | 0.5912 | −0.1296 | Compression |
| JPEG QF50 | 0.5938 | −0.1269 | Compression |
| Gaussian noise (light) | 0.5938 | −0.1270 | Noise |
| Gaussian noise (heavy) | 0.5938 | −0.1270 | Noise |
| Gaussian blur | 0.5881 | −0.1326 | Blur |
| Resize 0.75× | 0.6631 | −0.0576 | Geometric |
| Resize 0.5× | 0.6134 | −0.1073 | Geometric |

### The Plateau Problem

**JPEG QF70, JPEG QF50, noise light, and noise heavy all produce F1 ≈ 0.593 (±0.003).** This is not a coincidence — it indicates the model collapses to a fixed baseline when any input distribution shift occurs.

Possible explanations:
1. **Artifact dependency:** The model has learned JPEG compression artifacts as features. ANY degradation that modifies these artifacts destroys the same signal.
2. **Feature fragility:** The learned features are narrowband — they don't survive perturbation.
3. **Baseline regression:** Under degradation, the model reverts to outputting a generic "likely tampered region" prior learned from the dataset.

### Resize Resilience

Resize degrades less than compression/noise:
- 0.75× resize: only −0.058 drop
- 0.5× resize: −0.107 drop

This makes sense — resize preserves relative spatial relationships better than compression/noise, which corrupt pixel-level statistics the model relies on.

---

## 4. Failure Case Analysis

### Worst 10 Predictions (F1=0.0)

| Characteristic | Count |
|---|---|
| Copy-move | 8/10 |
| Splicing | 2/10 |
| Mask area <2% | 6/10 |
| Mask area 2–5% | 2/10 |
| Mask area >5% | 2/10 |

### Failure Taxonomy

1. **Copy-move + small region** (6/10): The model completely misses small duplicate-pasted regions. This is the dominant failure mode.
2. **Copy-move + medium region** (2/10): Even moderately sized copy-move regions are missed.
3. **Splicing + small region** (2/10): Rare, but small splicing can also fail.

### Root Cause Analysis

| Factor | Contribution | Evidence |
|---|---|---|
| Forgery type (copy-move) | PRIMARY | 8/10 worst cases are copy-move |
| Region size (small) | SECONDARY | 6/10 have <2% area; pos_weight absence amplifies |
| Loss design | CONTRIBUTING | No pos_weight → small regions get negligible gradient |
| Architecture | CONTRIBUTING | RGB-only → no forensic signal for copy-move |

---

## 5. Key Takeaways for v8

| Finding | Action | Priority |
|---|---|---|
| Overfitting at epoch 15 | Add LR scheduler | P0 |
| Threshold=0.1327 | Add pos_weight | P0 |
| Tampered-only F1=0.41 | Report as primary metric | P0 |
| Copy-move F1=0.31 | Investigate failure mode, add copy-move-specific analysis | P1 |
| 4 degradations give same F1 | Add JPEG/noise augmentation in training | P1 |
| Small region failure | Add mask-size stratification to evaluation | P1 |
| Bimodal F1 distribution | Investigate per-image confidence vs accuracy | P2 |
| Image-level heuristic | Consider learned classification head | P2 |
