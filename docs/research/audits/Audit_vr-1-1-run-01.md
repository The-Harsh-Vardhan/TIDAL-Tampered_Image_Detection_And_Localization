# Technical Audit: vR.1.1 (Run 01) — Evaluation Fix

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vr-1-1-etasr-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (15,511 MB VRAM) |
| **Cells** | 31 total (19 code, 12 markdown) |
| **Executed** | 19 of 19 code cells (all executed) |
| **Training** | 13 epochs (early stopped), best at epoch 8 |
| **Version** | vR.1.1 — First ablation study version |
| **Change** | Evaluation fix: 70/15/15 split, per-class metrics, ROC-AUC, ELA viz, model save |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.1.1 is the **first ablation study version** — it fixes the evaluation methodology so all subsequent experiments can be measured honestly. This is the most important version in the ablation series because it establishes the unbiased baseline against which every future change is judged.

### Configuration

```
VERSION          = 'vR.1.1'
CHANGE           = 'Evaluation fix: 70/15/15 split, per-class metrics, ROC-AUC, ELA viz, model save'
IMAGE_SIZE       = (128, 128)
ELA_QUALITY      = 90
BATCH_SIZE       = 32
EPOCHS           = 50
LEARNING_RATE    = 0.0001
TEST_SPLIT       = 0.15
VAL_SPLIT        = 0.15   (70/15/15 split)
EARLY_STOP       = patience=5, monitor=val_accuracy
SEED             = 42
```

### Changes from vR.1.0 (Baseline)

| Change | vR.1.0 | vR.1.1 |
|--------|--------|--------|
| Data split | 80/20 train/val | **70/15/15 train/val/test** |
| Evaluation set | Validation (biased) | **Test set (unbiased)** |
| Metric averaging | `weighted` (headline) | **Per-class + macro (headline)** |
| ROC-AUC | Not computed | **Computed: 0.9601** |
| ELA visualization | Not shown | **4×4 grid (4 Au + 4 Tp with ELA maps)** |
| Model save | Commented out | **Active (vR.1.1_ela_cnn_model.keras)** |
| Version tracking | None | **VERSION + CHANGE strings in config** |
| Ablation table | None | **Comparison table with previous versions** |

---

## 2. Results Summary

### Final Metrics (Test Set — Proper Hold-Out)

| Metric | Value | Paper Claims | Gap |
|--------|-------|-------------|-----|
| **Accuracy** | **88.38%** | 96.21% | **-7.83pp** |
| **ROC-AUC** | **0.9601** | — | New metric |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Authentic | 0.9170 | 0.8843 | 0.9004 | 1,124 |
| Tampered | 0.8393 | 0.8830 | 0.8606 | 769 |
| **Macro avg** | **0.8781** | **0.8837** | **0.8805** | 1,893 |
| Weighted avg | 0.8854 | 0.8838 | 0.8842 | 1,893 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|---|---|---|
| **True Au** | 994 | 130 |
| **True Tp** | 90 | 679 |

- FP rate: 11.6% (130 authentic misclassified as tampered)
- FN rate: 11.7% (90 tampered missed)

---

## 3. Training Dynamics

| Epoch | Train Acc | Val Acc | Val Loss | Notes |
|-------|-----------|---------|----------|-------|
| 1 | 0.7530 | 0.8325 | 0.3967 | |
| 4 | 0.8978 | 0.8631 | 0.3237 | |
| 6 | 0.9142 | 0.8768 | 0.2755 | |
| **8** | **0.9289** | **0.8864** | **0.2662** | **Best epoch (restored)** |
| 9 | 0.9309 | 0.8642 | 0.2966 | Val drops, overfitting starts |
| 10 | 0.9328 | 0.8726 | 0.2999 | Partial recovery |
| 11 | 0.9307 | 0.8716 | 0.3089 | Plateau |
| **12** | 0.9303 | **0.8208** | **0.4981** | **Val collapse — 5.1% drop, loss nearly doubles** |
| **13** | 0.9260 | **0.7960** | **0.6360** | **Catastrophic — val_acc below 80%, loss 2.4× best** |

**Critical observations:**
- Best epoch is 8 — consistent with all prior runs (same seed, same architecture)
- Epochs 12–13 show **catastrophic collapse**: val_loss explodes from 0.31 → 0.50 → 0.64 in two epochs
- This is **worse instability** than vR.ETASR/vR.1 (which had one bad epoch at 11 then recovered)
- **Root cause**: 10% less training data (8,829 vs 10,091 images) makes the model more prone to overfitting oscillation, especially with the 29.5M-param Flatten→Dense(256) bottleneck
- Early stopping correctly restored epoch 8 weights, so the collapse doesn't affect final metrics

---

## 4. Data Splits

| Split | Count | Authentic | Tampered | Ratio |
|-------|-------|-----------|----------|-------|
| Train | 8,829 (70%) | 5,243 | 3,586 | 1.46:1 |
| Validation | 1,892 (15%) | 1,124 | 768 | 1.46:1 |
| Test | 1,893 (15%) | 1,124 | 769 | 1.46:1 |
| **Total** | **12,614** | **7,491** | **5,123** | **1.46:1** |

Stratification is correctly maintained across all three splits. The class ratio is consistent.

---

## 5. Strengths

1. **Proper 3-way split** — The single most important methodological fix. Test set is used ONLY for final metrics, never seen during training or model selection.
2. **Per-class + macro metrics** — Honest reporting. The tampered precision (0.8393) is no longer hidden behind a weighted average (0.8854).
3. **ROC-AUC computed** — 0.9601. Strong threshold-independent discriminatory power. This metric is now available for all future comparisons.
4. **ELA visualization** — 4×4 grid showing 4 authentic + 4 tampered images with their ELA maps. Demonstrates the paper's core preprocessing contribution.
5. **Model saved** — `vR.1.1_ela_cnn_model.keras` persisted in native format.
6. **Version tracking** — `VERSION` and `CHANGE` strings in config. Ablation comparison table printed at end.
7. **All 19 code cells execute** — Zero crashes, zero errors.
8. **Consistent best epoch** — Epoch 8, same as all prior runs. Confirms reproducibility.

---

## 6. Weaknesses and Issues

### MAJOR

| ID | Issue | Impact |
|----|-------|--------|
| M1 | **FN rate doubled** — Tampered FN rate went from 5.4% (vR.1) to 11.7%. The model now misses 90 of 769 tampered images. This is a real regression in tampered detection sensitivity. | Performance regression |
| M2 | **Val collapse at epochs 12–13** — Worse than any prior run. Val_loss explodes to 0.6360 (2.4× best). The reduced training set makes the model significantly more unstable. | Training instability red flag |
| M3 | **No localization** — Still classification only. | Assignment requirement failure |

### MINOR

| ID | Issue | Impact |
|----|-------|--------|
| m1 | No data augmentation — reduced training set (8,829 vs 10,091) desperately needs augmentation | Contributes to overfitting |
| m2 | Class imbalance (1.46:1) not addressed | Contributes to FP/FN imbalance |
| m3 | `warnings.filterwarnings('ignore')` suppresses all warnings | Too aggressive |

---

## 7. Comparison with Previous Versions

| Metric | vR.ETASR (val) | vR.1 (val) | vR.0 (test) | **vR.1.1 (test)** | Direction |
|--------|----------------|------------|-------------|-------------------|-----------|
| Accuracy | 89.89% | 89.81% | 88.33% | **88.38%** | Expected drop |
| Au Precision | 0.9607 | 0.9593 | 0.9154 | **0.9170** | Expected drop |
| Au Recall | 0.8652 | 0.8652 | 0.8852 | **0.8843** | Improved |
| Au F1 | 0.9104 | 0.9098 | 0.9000 | **0.9004** | Slight drop |
| Tp Precision | 0.8279 | 0.8276 | 0.8400 | **0.8393** | Improved |
| Tp Recall | 0.9483 | 0.9463 | 0.8804 | **0.8830** | **Significant drop** |
| Tp F1 | 0.8840 | 0.8830 | 0.8597 | **0.8606** | Drop |
| Macro F1 | 0.8972 | 0.8964 | 0.8799 | **0.8805** | Expected |
| ROC-AUC | — | — | 0.9600 | **0.9601** | Consistent |
| FP rate | 13.5% | 13.5% | 11.5% | **11.6%** | Improved |
| FN rate | 5.2% | 5.4% | 12.0% | **11.7%** | **Major regression** |

**Key finding:** vR.1.1 and vR.0 produce nearly identical results (88.38% vs 88.33% accuracy, 0.9601 vs 0.9600 AUC). This is expected — they are the same architecture with the same 70/15/15 split. The tiny differences are due to:
1. vR.1.1 ran on P100, vR.0 ran on T4x2 (minor float precision differences)
2. vR.1.1 uses per-class metrics as headline, vR.0 still used weighted

**The accuracy drop from ~89.9% (val) to ~88.4% (test) is NOT regression** — it's the cost of honest evaluation on a held-out test set. However, the **FN rate increase from ~5% to ~12% IS a real concern** that needs to be addressed with augmentation (vR.1.2) and class weights (vR.1.3).

---

## 8. Paper Reproduction Score

**6/10** — Architecture unchanged. Evaluation methodology is now proper (test set, per-class metrics, ROC-AUC), which reveals the model is weaker than previously reported. The 7.83pp accuracy gap vs paper claims is the honest measurement.

---

## 9. Assignment Readiness Score

**5/10** — Significant improvement over vR.ETASR (3/10):
- ✅ Proper evaluation methodology (test set, per-class, macro, ROC-AUC)
- ✅ ELA visualization
- ✅ Model saved
- ✅ Version tracking and ablation comparison
- ❌ No localization (1.5 version away)
- ❌ No data augmentation
- ❌ No class weighting

---

## 10. Ablation Study Verdict

**This is the honest baseline.** vR.1.1 establishes the true performance of the ETASR CNN on CASIA v2.0:

| Metric | Honest Value |
|--------|-------------|
| Test Accuracy | 88.38% |
| Tampered Precision | 0.8393 |
| Tampered Recall | 0.8830 |
| Tampered F1 | 0.8606 |
| Macro F1 | 0.8805 |
| ROC-AUC | 0.9601 |

Every subsequent version in the ablation study will be compared against these numbers. The evaluation methodology is now sound, and improvements can be measured with confidence.

### Next Step: vR.1.2 (Data Augmentation)

The reduced training set (8,829 images, down from 10,091) is the most likely cause of the FN rate regression. Adding horizontal flip, vertical flip, and random rotation (±15°) should:
1. Compensate for the smaller training set
2. Reduce the overfitting gap (train 93% vs val 89%)
3. Stabilize the val_loss collapse at epochs 12–13
4. Potentially push test accuracy above 90%
