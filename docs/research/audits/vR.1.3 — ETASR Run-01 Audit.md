# Technical Audit: vR.1.3 — ETASR Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-1-3-etasr-ablation-study-class-weights-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (15,511 MB VRAM) |
| **Cells** | 31 total (19 code, 12 markdown) |
| **Executed** | 19 of 19 code cells (all executed) |
| **Training** | 14 epochs (early stopped), best at epoch 9 |
| **Version** | vR.1.3 — Class Weights (inverse-frequency balanced) |
| **Parent** | vR.1.1 (evaluation fix) — skips rejected vR.1.2 |
| **Change** | Add inverse-frequency class weights (Au=0.8420, Tp=1.2310) |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.1.3 adds **inverse-frequency class weights** to address the 1.46:1 class imbalance (7,491 authentic vs 5,123 tampered). This is the third ablation version, branching from vR.1.1 because vR.1.2 (augmentation) was REJECTED.

### Notebook Structure (31 cells)

| Cell | Type | Section / Purpose |
|------|------|-------------------|
| 0 | Markdown | Title, paper citation, change log, pipeline diagram, TOC |
| 1 | Markdown | Section 1: Version Change Log (vR.1.1 → vR.1.3) |
| 2 | Code | Section 2.1: Imports, seed, config, version info |
| 3 | Markdown | Section 3 header: Dataset Preparation |
| 4 | Code | Section 3.1: Mount dataset, count images |
| 5 | Code | Section 3.2: Load and parse image paths + labels |
| 6 | Markdown | Section 4 header: ELA Preprocessing |
| 7 | Code | Section 4.1: ELA function (JPEG Q=90, pixel diff, brightness) |
| 8 | Markdown | Section 5 header: ELA Visualization |
| 9 | Code | Section 5.1: ELA Visualization |
| 10 | Code | Section 5.2: Process all images → ELA arrays |
| 11 | Markdown | Section 6 header: Data Splitting |
| 12 | Code | Section 6.1: Stratified 70/15/15 split + class weight computation |
| 13 | Markdown | Section 7 header: Model Architecture |
| 14 | Code | Section 7.1: Build CNN model (unchanged from vR.1.1) |
| 15 | Markdown | Section 8 header: Training Pipeline |
| 16 | Code | Section 8.1: Compile Model |
| 17 | Code | Section 8.2: model.fit with class_weight (THE CHANGE) |
| 18 | Markdown | Section 9 header: Evaluation |
| 19 | Code | Section 9.1: Test set evaluation |
| 20 | Code | Section 9.2: Classification report |
| 21 | Code | Section 9.3: Confusion matrix |
| 22 | Code | Section 9.4: ROC curve |
| 23 | Markdown | Section 10 header: Results Visualization |
| 24 | Code | Section 10.1: Training curves |
| 25 | Code | Section 10.2: Precision/Recall curves |
| 26 | Code | Section 10.3: Sample predictions |
| 27 | Markdown | Section 11 header: Ablation Comparison |
| 28 | Code | Section 11.1: Ablation tracking table |
| 29 | Markdown | Section 12: Discussion |
| 30 | Code | Section 12.1: Save model weights |

---

## 2. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | Single-variable ablation maintained | Only change: `class_weight=CLASS_WEIGHT_DICT` added to model.fit |
| S2 | Correctly branches from vR.1.1 | Skips rejected vR.1.2 (augmentation) — proper ablation protocol |
| S3 | Class weights computed from training data | `compute_class_weight('balanced', ...)` on training split only — no data leakage |
| S4 | Highest test accuracy in series | 89.17% — surpasses vR.1.1's 88.38% by +0.79pp |
| S5 | Tampered recall improved significantly | 0.9012 vs 0.8830 (+1.82pp) — the stated goal of class weights |
| S6 | FN rate reduced | 9.9% (76/769) vs 11.7% (90/769) — 14 fewer tampered images missed |
| S7 | Best macro F1 in series | 0.8889 vs 0.8805 (+0.0084) |
| S8 | All frozen constants preserved | ELA Q=90, 128×128, seed=42, Adam lr=1e-4, batch=32, patience=5 |
| S9 | Model weights saved | `vR.1.3_ela_cnn_model.keras` |

---

## 3. Weaknesses

| # | Severity | Weakness | Impact |
|---|----------|----------|--------|
| W1 | **MAJOR** | Catastrophic val collapse at epochs 12-14: val_acc drops from 0.8901→0.8399→0.8266→0.8039, val_loss explodes 0.2717→0.4286→0.4831→0.5828 | Class weights did NOTHING to fix training instability. Same catastrophe as vR.1.1 (epochs 12-13). The 29.5M-param Flatten→Dense layer remains the structural root cause. |
| W2 | **MAJOR** | Still 7.04pp below paper claims (89.17% vs 96.21%) | Class weights closed only 0.79pp of the 7.83pp gap. At this rate, you'd need ~10 more equally successful ablations to match the paper. |
| W3 | **MAJOR** | No localization capability | Classification-only model cannot produce pixel-level masks. Assignment requirement failure persists. |
| W4 | MODERATE | ROC-AUC regressed: 0.9580 vs 0.9601 (−0.0021) | Though within noise, the threshold-independent discriminatory power did not improve — the opposite of what class weights should produce. The model shifted the decision boundary (improving recall) without actually learning better features. |
| W5 | MODERATE | FP rate essentially unchanged: 11.5% (129/1124) vs 11.5% (130/1124) in vR.1.1 | The 1.46:1 imbalance correction helped tampered recall but did nothing for authentic precision. |
| W6 | MINOR | 29.5M params for 8,829 training images (1:3,343 data-to-param ratio) | Grotesquely overparameterized. The Flatten→Dense(256) layer alone has 29.49M of the 29.52M total params. |
| W7 | MINOR | Blanket `warnings.filterwarnings('ignore')` | Suppresses potentially useful deprecation and convergence warnings |
| W8 | MINOR | No ModelCheckpoint callback | If Kaggle session dies, all training is lost |

---

## 4. Paper Reproduction Fidelity

| Aspect | Paper | vR.1.3 | Match? |
|--------|-------|--------|--------|
| ELA quality | 90 | 90 | ✅ |
| Image size | 128×128 | 128×128 | ✅ |
| Conv layers | 2× Conv2D(32, 5×5) | 2× Conv2D(32, 5×5) | ✅ |
| Pooling | MaxPool(2×2) | MaxPool(2×2) | ✅ |
| Dense | 256 units | 256 units | ✅ |
| Optimizer | Adam | Adam | ✅ |
| Dropout | 0.25 + 0.5 | 0.25 + 0.5 | ✅ |
| Class weights | Not in paper | **Added** | ❌ (intentional ablation) |
| Data split | Not specified | 70/15/15 | ✅ (honest eval) |

---

## 5. Dataset Pipeline Review

| Check | Status |
|-------|--------|
| Dataset: CASIA v2.0 | ✅ 12,614 images (7,491 Au + 5,123 Tp) |
| Stratified 70/15/15 split | ✅ Train=8,829 / Val=1,892 / Test=1,893 |
| Seed=42 | ✅ Deterministic split |
| ELA preprocessing (Q=90) | ✅ Standard pipeline |
| Normalization [0,1] | ✅ `/255.0` |
| Class weights computed from training set | ✅ Au=0.8420, Tp=1.2310 — no data leakage |
| Class ratio | ✅ 1.46:1 (Au/Tp) — weights inversely proportional |

---

## 6. Training Pipeline Review

| Parameter | Value | Status |
|-----------|-------|--------|
| Optimizer | Adam | ✅ Frozen |
| Learning rate | 0.0001 | ✅ Frozen |
| Batch size | 32 | ✅ Frozen |
| Max epochs | 50 | ✅ Frozen |
| Early stopping | patience=5, val_accuracy | ✅ Frozen |
| Class weights | Au=0.8420, Tp=1.2310 | **NEW** (the single change) |
| LR scheduler | None | ⚠️ Still missing |
| BatchNorm | None | ⚠️ Still missing |

### Training History (All 14 Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|-------|-----------|-----------|----------|---------|-------|
| 1 | 0.4905 | 0.7730 | 0.3737 | 0.8399 | |
| 2 | 0.3237 | 0.8747 | 0.3343 | 0.8621 | |
| 3 | 0.2913 | 0.8866 | 0.3130 | 0.8695 | |
| 4 | 0.2597 | 0.8976 | 0.3057 | 0.8700 | |
| 5 | 0.2335 | 0.9043 | 0.2977 | 0.8710 | |
| 6 | 0.2133 | 0.9127 | 0.2693 | 0.8837 | |
| 7 | 0.1876 | 0.9226 | 0.2835 | 0.8705 | Val dip |
| 8 | 0.1819 | 0.9223 | 0.2639 | 0.8890 | |
| **9** | **0.1670** | **0.9249** | **0.2717** | **0.8901** | **Best epoch (restored)** |
| 10 | 0.1610 | 0.9279 | 0.2694 | 0.8901 | Tied with 9 |
| 11 | 0.1614 | 0.9270 | 0.2776 | 0.8874 | Slight decline begins |
| **12** | 0.1598 | 0.9286 | **0.4286** | **0.8399** | **Val collapse begins** |
| **13** | 0.1582 | 0.9273 | **0.4831** | **0.8266** | **Collapse continues** |
| **14** | 0.1559 | 0.9271 | **0.5828** | **0.8039** | **Catastrophic — early stopping** |

**Key observations:**
- **Epochs 1-9:** Clean convergence. Val_acc rises from 0.84→0.89, train-val gap moderate.
- **Epoch 9-10:** Plateau at val_acc=0.8901. Model has peaked.
- **Epoch 12:** Catastrophic collapse. Val_loss doubles (0.2776→0.4286), val_acc drops 4.75pp in one epoch.
- **Epochs 13-14:** Collapse accelerates. Val_acc falls to 0.8039 (−8.62pp from best).
- **Pattern match:** Identical to vR.1.1's collapse at epochs 12-13. Class weights did NOT fix this.
- **Root cause:** The 29.5M-param Flatten→Dense layer memorizes training data, then the memorized features become anti-correlated with validation data as overfitting deepens.

---

## 7. Performance Summary

### Test Set Results

| Metric | vR.1.1 (Parent) | vR.1.3 | Delta | Assessment |
|--------|-----------------|--------|-------|------------|
| **Test Accuracy** | 88.38% | **89.17%** | **+0.79pp** | **Improved** |
| Au Precision | 0.9170 | 0.9290 | +0.0120 | Improved |
| Au Recall | 0.8843 | 0.8852 | +0.0009 | Unchanged |
| Au F1 | 0.9004 | 0.9066 | +0.0062 | Improved |
| **Tp Precision** | 0.8393 | 0.8431 | +0.0038 | Marginally improved |
| **Tp Recall** | 0.8830 | **0.9012** | **+0.0182** | **Significantly improved** |
| **Tp F1** | 0.8606 | **0.8712** | **+0.0106** | **Improved** |
| **Macro F1** | 0.8805 | **0.8889** | **+0.0084** | **Improved** |
| **ROC-AUC** | **0.9601** | 0.9580 | **−0.0021** | **Regressed (within noise)** |
| Epochs | 13 (8) | 14 (9) | +1 | Slightly longer |

### Confusion Matrix

| | Predicted Au | Predicted Tp |
|-|-------------|-------------|
| **True Au (1124)** | 995 (TN) | 129 (FP) |
| **True Tp (769)** | 76 (FN) | 693 (TP) |

- FP rate: 11.5% (129/1124) — unchanged from vR.1.1
- FN rate: 9.9% (76/769) — **improved** from 11.7% (90/769)
- Net: 14 more tampered images correctly detected, 1 fewer false alarm

### Model Architecture (Unchanged)

```
Conv2D(32, 5×5, ReLU)        → 2,432 params
Conv2D(32, 5×5, ReLU)        → 25,632 params
MaxPooling2D(2×2)             → 0
Dropout(0.25)                 → 0
Flatten                       → 0
Dense(256, ReLU)              → 29,491,456 params  ← 99.9% of all params
Dropout(0.5)                  → 0
Dense(2, Softmax)             → 514 params

Total params:      29,520,034
Trainable params:  29,520,034
Non-trainable:     0
```

---

## 8. Result Analysis

### Verdict: **POSITIVE** ✅

Test accuracy improved by +0.79pp (88.38%→89.17%) and Macro F1 improved by +0.0084 (0.8805→0.8889). Both exceed the ±0.5pp threshold for a POSITIVE verdict.

### What Class Weights Achieved

1. **Tampered recall rose from 0.8830→0.9012** — the primary goal. The loss function now penalizes missed tampered images 1.23× more, and the model responded by shifting its decision boundary toward classifying more images as tampered.
2. **FN rate dropped from 11.7%→9.9%** — 14 fewer tampered images escape detection.
3. **Test accuracy improved +0.79pp** — modest but real.
4. **Macro F1 improved +0.0084** — the most honest metric improved.

### What Class Weights Did NOT Achieve

1. **ROC-AUC regressed** (0.9601→0.9580). This is damning. ROC-AUC measures threshold-independent discrimination — the model's ability to separate the two classes at ANY threshold. The regression means the model didn't learn better features; it just moved the threshold. Class weights shifted WHERE the boundary sits, not HOW WELL the model separates classes.
2. **FP rate unchanged** (11.5%). Authentic images are still misclassified at the same rate. The model's authentic-detection ability is no better.
3. **Training instability unfixed.** Catastrophic val collapse at epochs 12-14: val_acc drops 8.62pp from best. Class weights do not address the overfitting caused by the 29.5M-param architecture.
4. **Paper gap barely closed.** 89.17% vs 96.21% = 7.04pp gap. Class weights closed 0.79pp. The gap is structural, not a class-balance problem.

---

## 9. Paper Reproduction Score

| Criterion | Score | Notes |
|-----------|-------|-------|
| Architecture fidelity | 9/10 | Unchanged architecture; class weights are training config |
| Training pipeline | 8/10 | Class weights deviate from paper but improve rigor |
| Evaluation rigor | 10/10 | Full suite: per-class, macro, AUC, CM |
| Reproducibility | 10/10 | seed=42, deterministic split, model saved |
| **Overall** | **37/40** | |

---

## 10. Assignment Readiness Score

| Criterion | Score | Notes |
|-----------|-------|-------|
| Detection accuracy | 7/10 | 89.17% — best in series but 7pp below paper |
| Localization | 0/10 | Not applicable (classification only) |
| Visualization | 7/10 | ELA viz, training curves, ROC, CM |
| Documentation | 9/10 | Full ablation tracking, version notes |
| **Overall** | **23/40** | Localization addressed by Track 2 (vR.P.x) |

---

## 11. Recommended Next Steps

1. **vR.1.4 — BatchNormalization** (IMMEDIATE): Add BN after each Conv2D to address the catastrophic val collapse at epochs 12-14. BN normalizes activations and smooths the loss landscape.
2. **vR.1.5 — ReduceLROnPlateau**: If BN stabilizes training, the LR scheduler can then exploit the smoother loss landscape for better convergence.
3. **Retain class weights for all future versions**: The improvement is real (+0.79pp acc, +1.82pp Tp recall).
4. **Long-term:** The 29.5M-param Flatten→Dense bottleneck must be addressed (vR.1.7 GlobalAveragePooling). No amount of training tricks can fix an architecture where 99.9% of parameters are in one layer.
