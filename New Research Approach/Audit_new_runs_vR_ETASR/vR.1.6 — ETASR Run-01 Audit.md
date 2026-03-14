# Technical Audit: vR.1.6 — ETASR Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-1-6-deeper-cnn-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (15,511 MB VRAM) |
| **Cells** | 31 total (19 code, 12 markdown) |
| **Executed** | 19 of 19 code cells (all executed) |
| **Training** | 18 epochs (early stopped), best at epoch 13 |
| **Version** | vR.1.6 — Deeper CNN (3rd Conv2D layer) |
| **Parent** | vR.1.5 (LR Scheduler — NEUTRAL) |
| **Change** | Add 3rd Conv2D(64, 3x3) + MaxPool(2,2) before Flatten |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.1.6 adds a **third convolutional layer** to address the fundamental overfitting problem identified across vR.1.3-1.5: the Flatten->Dense(256) bottleneck had 29.5M parameters because 2 conv layers produced a 60x60x32 feature map after one MaxPool. By adding Conv2D(64, 3x3) + MaxPool(2,2), the feature map shrinks to 29x29x64 before Flatten, reducing the Flatten->Dense pathway from 29.5M to 13.8M parameters — a **53.2% reduction**.

This is the **BEST ETASR result**: first model to break 90% test accuracy (90.23%), highest Macro F1 (0.9004), and highest ROC-AUC (0.9657).

---

## 2. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **First model to break 90% test accuracy** | 90.23% — +1.27pp from vR.1.5, +1.85pp from vR.1.1 baseline |
| S2 | **Best Macro F1 in series** | 0.9004 — first to cross 0.90 (vR.1.3 was 0.8889) |
| S3 | **Best ROC-AUC in series** | 0.9657 — surpasses vR.1.1's 0.9601 for the first time |
| S4 | **53.2% parameter reduction** | 13.8M vs 29.5M (vR.1.5) — fewer params, better results |
| S5 | **Longest training in series** | 18 epochs (best at 13) — LR scheduler reduces twice, model converges slowly |
| S6 | **LR scheduler fully utilized** | Triggered twice: 1e-4 -> 5e-5 (epoch 13) -> 2.5e-5 (epoch 16) |
| S7 | **Lowest FP rate in series** | 12.5% (141/1124) — fewest false alarms |
| S8 | **Highest Au Precision** | 0.9572 — best authentic detection confidence |
| S9 | **Tampered recall near-best** | 0.9428 — 725/769 tampered images correctly identified |
| S10 | Single-variable ablation maintained | Only change: added Conv2D(64,3x3)+MaxPool before Flatten |

---

## 3. Weaknesses

| # | Severity | Weakness | Impact |
|---|----------|----------|--------|
| W1 | **MAJOR** | Epoch 1 BN catastrophe persists: val_loss=0.7624, val_acc=0.4059 | Same first-epoch spike as vR.1.4/1.5. BN warmup remains unsolved. |
| W2 | **MAJOR** | Train-val gap grows: train_acc=0.9360 vs val_acc=0.9001 at best epoch | ~3.6pp gap indicates overfitting is reduced but not eliminated |
| W3 | MODERATE | Still 5.98pp below paper's 96.21% | Accuracy gap reduced from 7.04pp (vR.1.5) but remains significant |
| W4 | MODERATE | 13.8M params still dominated by Flatten->Dense | Dense(256) layer alone has 13.78M params (99.7% of total) |
| W5 | MINOR | No localization capability | Classification only — addressed by Track 2 |

---

## 4. Paper Reproduction Fidelity

| Aspect | Paper | vR.1.6 | Match? |
|--------|-------|--------|--------|
| ELA quality | 90 | 90 | Yes |
| Image size | 128x128 | 128x128 | Yes |
| Conv layers | 2x Conv2D(32, 5x5) | 2x Conv2D(32, 5x5) + **Conv2D(64, 3x3)** | No (deeper) |
| BatchNorm | Not in paper | After each Conv2D | No (intentional) |
| LR Scheduler | Not in paper | ReduceLROnPlateau | No (intentional) |
| Pooling | MaxPool(2,2) | MaxPool(2,2) | Yes |
| Dense | 256 units | 256 units | Yes |
| Optimizer | Adam | Adam | Yes |
| Dropout | 0.25 + 0.5 | 0.25 + 0.5 | Yes |
| Class weights | Not in paper | Inverse-frequency | No (intentional) |

---

## 5. Dataset Pipeline Review

| Check | Status |
|-------|--------|
| Dataset: CASIA v2.0 | 12,614 images (7,491 Au + 5,123 Tp) |
| Stratified 70/15/15 split | Train=8,829 / Val=1,892 / Test=1,893 |
| Seed=42 | Deterministic split |
| ELA preprocessing (Q=90) | Standard pipeline |
| Normalization [0,1] | `/255.0` |
| Class weights | Au=0.8420, Tp=1.2310 |

---

## 6. Training Pipeline Review

| Parameter | Value | Status |
|-----------|-------|--------|
| Optimizer | Adam | Frozen |
| Learning rate | 0.0001 (initial) | Frozen |
| Batch size | 32 | Frozen |
| Max epochs | 50 | Frozen |
| Early stopping | patience=5, val_accuracy | Frozen |
| Class weights | inverse-frequency | Carried from vR.1.3 |
| BatchNormalization | After each Conv2D | Carried from vR.1.4 |
| LR Scheduler | ReduceLROnPlateau(val_loss, 0.5, 3, 1e-6) | Carried from vR.1.5 |
| **3rd Conv layer** | **Conv2D(64, 3x3) + MaxPool(2,2)** | **NEW** |

### Model Architecture

```
Layer (type)                    Output Shape          Params
────────────────────────────────────────────────────────────
conv2d (Conv2D 32, 5x5)        (None, 124, 124, 32)  2,432
batch_norm                      (None, 124, 124, 32)  128
conv2d_1 (Conv2D 32, 5x5)      (None, 120, 120, 32)  25,632
batch_norm_1                    (None, 120, 120, 32)  128
max_pooling2d (2x2)             (None, 60, 60, 32)    0
conv2d_2 (Conv2D 64, 3x3)      (None, 58, 58, 64)    18,496    ** NEW **
max_pooling2d_1 (2x2)           (None, 29, 29, 64)    0         ** NEW **
dropout (0.25)                  (None, 29, 29, 64)    0
flatten                         (None, 53824)         0
dense (256, relu)               (None, 256)           13,779,200
dropout_1 (0.5)                 (None, 256)           0
dense_1 (2, softmax)            (None, 2)             514
────────────────────────────────────────────────────────────
Total params:           13,826,530 (52.74 MB)
Trainable params:       13,826,402 (52.74 MB)
Non-trainable params:   128 (512 B)
```

**Parameter reduction from vR.1.5:** 29,520,290 -> 13,826,530 (**-53.2%**)

### Training History (All 18 Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Notes |
|-------|-----------|-----------|----------|---------|-----|-------|
| 1 | 0.9019 | 0.7516 | 0.7624 | 0.4059 | 1e-4 | BN warmup spike |
| 2 | 0.3709 | 0.8572 | 0.5430 | 0.7156 | 1e-4 | Recovery |
| 3 | 0.3418 | 0.8706 | 0.3175 | 0.8726 | 1e-4 | Stabilized |
| 4 | 0.3082 | 0.8824 | 0.2969 | 0.8821 | 1e-4 | |
| 5 | 0.2736 | 0.8908 | 0.2893 | 0.8800 | 1e-4 | |
| 6 | 0.2600 | 0.8931 | 0.2634 | 0.8890 | 1e-4 | |
| 7 | 0.2218 | 0.9083 | 0.2793 | 0.8842 | 1e-4 | |
| 8 | 0.1913 | 0.9163 | 0.2557 | 0.8948 | 1e-4 | |
| 9 | 0.1733 | 0.9236 | 0.2552 | 0.8911 | 1e-4 | |
| 10 | 0.1583 | 0.9254 | 0.2730 | 0.8911 | 1e-4 | |
| 11 | 0.1518 | 0.9272 | 0.2682 | 0.8953 | 1e-4 | |
| 12 | 0.1452 | 0.9303 | 0.3058 | 0.8858 | 1e-4 | Val_loss spike |
| **13** | **0.1260** | **0.9360** | **0.2724** | **0.9001** | **5e-5** | **Best epoch, LR reduced** |
| 14 | 0.1230 | 0.9378 | 0.3145 | 0.8927 | 5e-5 | |
| 15 | 0.1120 | 0.9408 | 0.2759 | 0.8969 | 5e-5 | |
| 16 | 0.1097 | 0.9399 | 0.3140 | 0.8943 | 2.5e-5 | LR reduced again |
| 17 | 0.0979 | 0.9443 | 0.3122 | 0.8980 | 2.5e-5 | |
| 18 | 0.0939 | 0.9445 | 0.3149 | 0.8943 | 2.5e-5 | Early stopping |

**Key observations:**
- **Epoch 1:** BN warmup spike persists (val_acc=0.4059) but val_loss=0.7624 is much milder than vR.1.5's 14.74.
- **Epochs 2-8:** Steady convergence with consistent val_acc improvement.
- **Epoch 8-13:** Slower gains. LR reduced at epoch 13, which coincides with the best val_acc (0.9001).
- **Epochs 14-18:** Val_acc oscillates around 0.89-0.90 but never surpasses epoch 13. LR reduced again at epoch 16. Early stopping fires at epoch 18.
- **The LR scheduler worked as intended:** Two reductions extended training from ~10 epochs (vR.1.5) to 18, allowing the model to reach 90.01% val_acc.

---

## 7. Performance Summary

### Test Set Results

| Metric | vR.1.5 (Parent) | vR.1.6 | Delta | Assessment |
|--------|-----------------|--------|-------|------------|
| **Test Accuracy** | 88.96% | **90.23%** | **+1.27pp** | **POSITIVE** |
| Au Precision | 0.9403 | **0.9572** | +0.0169 | Strong improvement |
| Au Recall | 0.8692 | **0.8746** | +0.0054 | Marginal |
| Au F1 | 0.9034 | **0.9140** | +0.0106 | Clear improvement |
| Tp Precision | 0.8279 | **0.8372** | +0.0093 | Improved |
| Tp Recall | 0.9194 | **0.9428** | +0.0234 | Strong improvement |
| Tp F1 | 0.8712 | **0.8869** | +0.0157 | Strong improvement |
| **Macro F1** | 0.8873 | **0.9004** | **+0.0131** | **POSITIVE** |
| **ROC-AUC** | 0.9560 | **0.9657** | **+0.0097** | **Best in series** |
| Epochs | 10 (5) | **18 (13)** | +8 | Much longer training |

### Confusion Matrix

| | Predicted Au | Predicted Tp |
|-|-------------|-------------|
| **True Au (1124)** | 983 (TN) | 141 (FP) |
| **True Tp (769)** | 44 (FN) | 725 (TP) |

- FP rate: 12.5% (141/1124) — **best in series** (was 13.1% in vR.1.5)
- FN rate: 5.7% (44/769) — near-best (vR.1.4/1.5 had 8.1%)
- Net: 6 fewer FPs and 18 fewer FNs than vR.1.5

### Comparison with All Previous Versions

| Metric | vR.1.1 | vR.1.3 | vR.1.5 | **vR.1.6** |
|--------|--------|--------|--------|-----------|
| Test Acc | 88.38% | 89.17% | 88.96% | **90.23%** |
| Macro F1 | 0.8805 | 0.8889 | 0.8873 | **0.9004** |
| ROC-AUC | 0.9601 | 0.9580 | 0.9560 | **0.9657** |
| Tp Recall | 0.8830 | 0.9012 | 0.9194 | **0.9428** |
| Params | 29.5M | 29.5M | 29.5M | **13.8M** |

**vR.1.6 is the undisputed best across ALL metrics.**

---

## 8. Result Analysis

### Verdict: **POSITIVE**

Clear improvements over parent vR.1.5:
- Test Accuracy: +1.27pp (well above +0.5pp threshold)
- Macro F1: +0.0131 (first to cross 0.90)
- ROC-AUC: +0.0097 (new series best, surpassing even vR.1.1)

### What the Deeper CNN Achieved

1. **Broke the 90% barrier.** Test accuracy jumped from 88.96% to 90.23% — the largest single improvement since vR.1.3's class weights.
2. **53.2% parameter reduction.** Adding a conv layer paradoxically reduced total parameters by compressing the spatial dimensions before Flatten.
3. **Better discriminative power.** ROC-AUC 0.9657 is the highest ever, meaning the model's probability outputs are better calibrated.
4. **Extended useful training.** 18 epochs (best at 13) vs 10 epochs (best at 5) in vR.1.5. The deeper architecture has more capacity to learn from longer training.
5. **Improved both precision and recall.** Unlike previous versions which traded one for the other, vR.1.6 improved all four per-class metrics.

### Why This Worked: The Flatten Bottleneck

The root cause of overfitting in vR.1.1-1.5 was the 115,200-dimensional Flatten output (60x60x32) feeding into Dense(256) — a 29.5M parameter layer acting as a memorization engine. By adding Conv2D(64,3x3)+MaxPool, the feature map shrinks to 53,824 (29x29x64), cutting the Dense layer to 13.8M params. The extra conv layer also provides better feature extraction before classification.

### Architectural Insight

The 3rd conv layer serves a dual purpose:
1. **Feature extraction:** Learns 64 higher-level filters from the 32-channel feature maps
2. **Spatial reduction:** MaxPool(2,2) halves the spatial dimensions, dramatically reducing Flatten output

This is a textbook example of "more compute per parameter" — a deeper network with fewer total parameters outperforms a shallower, wider one.

---

## 9. Paper Reproduction Score

| Criterion | Score | Notes |
|-----------|-------|-------|
| Architecture fidelity | 6/10 | 3 conv layers (paper has 2) + BN + LR scheduler |
| Training pipeline | 9/10 | Robust pipeline with all improvements |
| Evaluation rigor | 10/10 | Full suite: per-class, macro, AUC, CM |
| Reproducibility | 10/10 | seed=42, deterministic split, model saved |
| **Overall** | **35/40** | |

---

## 10. Assignment Readiness Score

| Criterion | Score | Notes |
|-----------|-------|-------|
| Detection accuracy | 8/10 | 90.23% — 5.98pp below paper but strong |
| Localization | 0/10 | Not applicable (classification only) |
| Visualization | 7/10 | ELA viz, training curves, ROC, CM |
| Documentation | 9/10 | Full ablation tracking, version notes |
| **Overall** | **24/40** | Localization addressed by Track 2 (vR.P.x) |

---

## 11. Recommended Next Steps

1. **vR.1.7 — GAP replaces Flatten** (IMMEDIATE): Replace Flatten with GlobalAveragePooling2D. This eliminates the 13.8M Flatten->Dense bottleneck entirely, reducing to ~64K params. Tests whether spatial pooling can preserve accuracy.

2. **Key insight from vR.1.6:** The performance ceiling is architectural, not training-related. BN, class weights, and LR scheduler provided incremental gains (+0.5pp each at best). The conv layer addition provided +1.27pp — the largest single-change improvement. Further architectural changes (GAP, more conv layers) are the path forward.

3. **The 96.21% gap is now 5.98pp** — reduced from 7.04pp at vR.1.5. Two more POSITIVE architectural changes could potentially close it further, but the 96.21% paper figure may not be honestly achievable on a proper test split.
