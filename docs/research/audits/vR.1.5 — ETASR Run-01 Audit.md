# Technical Audit: vR.1.5 — ETASR Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-1-5-learning-rate-scheduler-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (15,511 MB VRAM) |
| **Cells** | 31 total (19 code, 12 markdown) |
| **Executed** | 19 of 19 code cells (all executed) |
| **Training** | 10 epochs (early stopped), best at epoch 5 |
| **Version** | vR.1.5 — ReduceLROnPlateau Learning Rate Scheduler |
| **Parent** | vR.1.4 (BatchNormalization — NEUTRAL) |
| **Change** | Add ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6) |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.1.5 adds a **ReduceLROnPlateau** learning rate scheduler to address the training instability and short training duration observed in vR.1.4 (epoch 1 val_loss=16.13, only 8 epochs total). The scheduler monitors val_loss and halves the learning rate after 3 epochs of no improvement.

### LR Scheduler Behavior in This Run

- **Initial LR:** 1e-4 (0.0001)
- **Trigger:** Once, after epoch 8 (val_loss did not improve from best at epoch 5 for 3 consecutive epochs)
- **Reduced LR:** 5e-5 (0.00005)
- **Final LR:** 5e-5
- **Epochs:** 10 total (best at 5), early stopping fired at epoch 10

---

## 2. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | Single-variable ablation maintained | Only change: ReduceLROnPlateau callback added |
| S2 | All frozen constants preserved | ELA Q=90, 128×128, seed=42, Adam lr=1e-4, batch=32, BN, class weights |
| S3 | LR scheduler triggered correctly | Reduced LR from 1e-4 to 5e-5 after epoch 8 (3 epochs of no val_loss improvement) |
| S4 | Training 2 epochs longer than vR.1.4 | 10 vs 8 epochs — scheduler gave model 2 more epochs at lower LR |
| S5 | Test accuracy improved over vR.1.4 | 88.96% vs 88.75% (+0.21pp) |
| S6 | Tampered recall maintained at best-in-series | 0.9194 (tied with vR.1.4) |
| S7 | FN rate matched vR.1.4 best | 8.1% (62/769) — tied best in series |
| S8 | Epoch 1 spike slightly reduced | val_loss=14.74 vs vR.1.4's 16.13 (−8.6%) |

---

## 3. Weaknesses

| # | Severity | Weakness | Impact |
|---|----------|----------|--------|
| W1 | **MAJOR** | Epoch 1 catastrophe persists: val_loss=14.74, val_acc=0.4059 | The LR scheduler did NOT prevent the BN warmup spike. It can only reduce LR AFTER an epoch completes, so it cannot dampen the first-epoch catastrophe. |
| W2 | **MAJOR** | Only 2 more epochs than vR.1.4 (10 vs 8) | The scheduler was expected to produce 15-30 epochs. Instead, early stopping (patience=5) fired because val_accuracy peaked at epoch 5 and never recovered. The scheduler only triggered once. |
| W3 | **MAJOR** | Val_loss monotonically increased after epoch 5 | 0.2807→0.3067→0.3247→0.3284→0.3437→0.3521. Even with LR reduction at epoch 9, val_loss continued rising. The model is overfitting and the scheduler cannot fix this. |
| W4 | MODERATE | ROC-AUC: 0.9560 — still below vR.1.1 baseline (0.9601) | −0.0041 from baseline. Five ablations in, the model's threshold-independent discriminatory power has not improved. |
| W5 | MODERATE | Macro F1 (0.8873) below vR.1.3's 0.8889 | The scheduler did not improve the best known Macro F1. |
| W6 | MINOR | Tp Precision: 0.8279 is the lowest-tied in the honest-eval series | Same as vR.1.1's 0.8393→0.8279 regressed with BN and scheduler |
| W7 | MINOR | No localization capability | Still classification only. |

---

## 4. Paper Reproduction Fidelity

| Aspect | Paper | vR.1.5 | Match? |
|--------|-------|--------|--------|
| ELA quality | 90 | 90 | ✅ |
| Image size | 128×128 | 128×128 | ✅ |
| Conv layers | 2× Conv2D(32, 5×5) | 2× Conv2D(32, 5×5) | ✅ |
| BatchNorm | Not in paper | Added (vR.1.4) | ❌ (intentional) |
| LR Scheduler | Not in paper | Added (vR.1.5) | ❌ (intentional) |
| Pooling | MaxPool(2×2) | MaxPool(2×2) | ✅ |
| Dense | 256 units | 256 units | ✅ |
| Optimizer | Adam | Adam | ✅ |
| Dropout | 0.25 + 0.5 | 0.25 + 0.5 | ✅ |

---

## 5. Dataset Pipeline Review

| Check | Status |
|-------|--------|
| Dataset: CASIA v2.0 | ✅ 12,614 images (7,491 Au + 5,123 Tp) |
| Stratified 70/15/15 split | ✅ Train=8,829 / Val=1,892 / Test=1,893 |
| Seed=42 | ✅ Deterministic split |
| ELA preprocessing (Q=90) | ✅ Standard pipeline |
| Normalization [0,1] | ✅ `/255.0` |
| Class weights | ✅ Au=0.8420, Tp=1.2310 |

---

## 6. Training Pipeline Review

| Parameter | Value | Status |
|-----------|-------|--------|
| Optimizer | Adam | ✅ Frozen |
| Learning rate | 0.0001 (initial) | ✅ Frozen |
| Batch size | 32 | ✅ Frozen |
| Max epochs | 50 | ✅ Frozen |
| Early stopping | patience=5, val_accuracy | ✅ Frozen |
| Class weights | inverse-frequency | ✅ Carried from vR.1.3 |
| BatchNormalization | After each Conv2D | ✅ Carried from vR.1.4 |
| LR Scheduler | ReduceLROnPlateau(val_loss, 0.5, 3, 1e-6) | **NEW** |

### Training History (All 10 Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Notes |
|-------|-----------|-----------|----------|---------|-----|-------|
| 1 | 1.0747 | 0.7677 | **14.7414** | 0.4059 | 1e-4 | BN warmup catastrophe |
| 2 | 0.3036 | 0.8799 | 3.8575 | 0.7363 | 1e-4 | Recovery |
| 3 | 0.2473 | 0.9023 | 0.2893 | 0.8626 | 1e-4 | Stabilized |
| 4 | 0.2053 | 0.9114 | 0.2838 | 0.8832 | 1e-4 | |
| **5** | **0.1698** | **0.9219** | **0.2807** | **0.8916** | **1e-4** | **Best epoch** |
| 6 | 0.1796 | 0.9227 | 0.3067 | 0.8864 | 1e-4 | Val_loss starts rising |
| 7 | 0.1424 | 0.9346 | 0.3247 | 0.8842 | 1e-4 | |
| 8 | 0.1465 | 0.9334 | 0.3284 | 0.8747 | 1e-4 | LR reduction triggered |
| 9 | 0.1238 | 0.9414 | 0.3437 | 0.8864 | **5e-5** | Reduced LR, slight recovery |
| 10 | 0.1120 | 0.9467 | 0.3521 | 0.8858 | 5e-5 | Early stopping |

**Key observations:**
- **Epoch 1:** BN warmup spike persists (val_loss=14.74). Slightly better than vR.1.4's 16.13, but still catastrophic.
- **Epochs 2-5:** Rapid convergence, peaking at epoch 5 (val_acc=0.8916, val_loss=0.2807).
- **Epochs 6-8:** Overfitting begins. Val_loss rises while train_loss drops. Classic overfit signature.
- **Epoch 8→9:** LR halved (1e-4→5e-5). Val_acc briefly recovers from 0.8747→0.8864, but this doesn't surpass the epoch 5 best.
- **Epoch 10:** Val_loss still rising (0.3521). Early stopping fires (5 epochs since best at epoch 5).
- **The LR scheduler bought exactly 2 extra epochs** — it prevented early stopping from firing at epoch 8 by providing a brief improvement at reduced LR, but the underlying overfitting was not resolved.

---

## 7. Performance Summary

### Test Set Results

| Metric | vR.1.4 (Parent) | vR.1.5 | Delta | Assessment |
|--------|-----------------|--------|-------|------------|
| **Test Accuracy** | 88.75% | 88.96% | +0.21pp | Within noise |
| Au Precision | 0.9401 | 0.9403 | +0.0002 | Unchanged |
| Au Recall | 0.8657 | 0.8692 | +0.0035 | Marginal |
| Au F1 | 0.9013 | 0.9034 | +0.0021 | Marginal |
| Tp Precision | 0.8240 | 0.8279 | +0.0039 | Marginal |
| **Tp Recall** | **0.9194** | **0.9194** | **0.0000** | **Identical** |
| Tp F1 | 0.8691 | 0.8712 | +0.0021 | Marginal |
| **Macro F1** | 0.8852 | 0.8873 | +0.0021 | Within noise |
| **ROC-AUC** | 0.9536 | 0.9560 | +0.0024 | Within noise |
| Epochs | 8 (3) | 10 (5) | +2 | Slightly longer |

### Confusion Matrix

| | Predicted Au | Predicted Tp |
|-|-------------|-------------|
| **True Au (1124)** | 977 (TN) | 147 (FP) |
| **True Tp (769)** | 62 (FN) | 707 (TP) |

- FP rate: 13.1% (147/1124) — slightly improved from vR.1.4's 13.4%
- FN rate: 8.1% (62/769) — unchanged from vR.1.4
- Net: 4 fewer false positives, same false negatives

### Comparison with vR.1.3 (Best Previous)

| Metric | vR.1.3 | vR.1.5 | Delta |
|--------|--------|--------|-------|
| Test Accuracy | **89.17%** | 88.96% | −0.21pp |
| Macro F1 | **0.8889** | 0.8873 | −0.0016 |
| ROC-AUC | 0.9580 | 0.9560 | −0.0020 |
| Tp Recall | 0.9012 | **0.9194** | +0.0182 |

vR.1.5 still has not surpassed vR.1.3's accuracy or Macro F1, though it has better Tp recall.

---

## 8. Result Analysis

### Verdict: **NEUTRAL**

All deltas from parent vR.1.4 are within ±0.5pp:
- Test Accuracy: +0.21pp (within noise)
- Macro F1: +0.0021 (within noise)
- ROC-AUC: +0.0024 (within noise)

The LR scheduler produced marginal improvements across all metrics but nothing exceeds the POSITIVE threshold (≥0.5pp).

### What ReduceLROnPlateau Achieved

1. **Extended training by 2 epochs** (10 vs 8). The LR reduction at epoch 9 briefly improved val_acc, delaying early stopping.
2. **Marginal metric improvements across the board** — every metric is slightly better than vR.1.4, but all within noise.
3. **Epoch 1 spike slightly reduced** — val_loss=14.74 vs 16.13. The BN warmup effect is less severe (possibly due to randomness, not the scheduler).

### What ReduceLROnPlateau Did NOT Achieve

1. **Did not extend training to 15-30 epochs as expected.** Only 10 epochs. The scheduler triggered once and then early stopping fired anyway.
2. **Did not prevent the BN epoch 1 catastrophe.** The scheduler can only act AFTER an epoch, so it cannot prevent the first-epoch spike.
3. **Did not push accuracy past 89%.** Best accuracy in the honest-eval series remains vR.1.3's 89.17%.
4. **Did not fix the monotonic val_loss increase.** After epoch 5, val_loss rose in every epoch — a clear overfitting signature that even reduced LR cannot fix.

### Decision: KEEP LR Scheduler

The scheduler is retained despite a NEUTRAL verdict because:
- All metrics marginally improved (consistent direction)
- It provides infrastructure for longer training if future architecture changes reduce overfitting
- It does not harm any metric

---

## 9. Paper Reproduction Score

| Criterion | Score | Notes |
|-----------|-------|-------|
| Architecture fidelity | 7/10 | BN + LR scheduler deviate from paper |
| Training pipeline | 9/10 | Robust pipeline with scheduler, class weights |
| Evaluation rigor | 10/10 | Full suite: per-class, macro, AUC, CM |
| Reproducibility | 10/10 | seed=42, deterministic split, model saved |
| **Overall** | **36/40** | |

---

## 10. Assignment Readiness Score

| Criterion | Score | Notes |
|-----------|-------|-------|
| Detection accuracy | 7/10 | 88.96% — 7.25pp below paper |
| Localization | 0/10 | Not applicable (classification only) |
| Visualization | 7/10 | ELA viz, training curves, ROC, CM |
| Documentation | 9/10 | Full ablation tracking, version notes |
| **Overall** | **23/40** | Localization addressed by Track 2 (vR.P.x) |

---

## 11. Recommended Next Steps

1. **vR.1.6 — Deeper CNN** (IMMEDIATE): Add a 3rd Conv2D(64, 3×3) + MaxPool before Flatten. This addresses W10 (shallow feature extraction) and will dramatically reduce the Flatten→Dense size, potentially fixing the overfitting that the scheduler couldn't solve.

2. **Key insight from vR.1.5:** The overfitting problem is not solvable with training tricks (class weights, BN, LR scheduler). The root cause is architectural: 29.5M parameters in Flatten→Dense with insufficient convolutional feature extraction. vR.1.6 and vR.1.7 (GAP) are the critical experiments.

3. **Retain all vR.1.5 components** (class weights, BN, LR scheduler) for subsequent experiments.
