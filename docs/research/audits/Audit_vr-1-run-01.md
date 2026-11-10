# Technical Audit: vR.1 (Run 01)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vr-1-etasr-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, 2x Tesla T4 (13,757 MB each, only 1 used) |
| **Cells** | 30 total (19 code, 11 markdown) |
| **Executed** | 19 of 19 code cells (all executed) |
| **Training** | 13 epochs (early stopped), best at epoch 8 |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.1 is the **second implementation** of the ETASR paper baseline. It retains the 80/20 train/val split from vR.ETASR but adds ELA visualization, sample prediction display, model save, and documented bug fixes from the reference code. It sits between vR.ETASR (bare baseline) and vR.0 (which added a 3-way split).

### Configuration

```
IMAGE_SIZE       = (128, 128)
ELA_QUALITY      = 90
BATCH_SIZE       = 32
EPOCHS           = 50
LEARNING_RATE    = 0.0001
VALIDATION_SPLIT = 0.2   (80/20, no test set)
EARLY_STOP       = patience=5, monitor=val_accuracy
SEED             = 42
```

### Key Changes from vR.ETASR

| Change | vR.ETASR | vR.1 |
|--------|----------|------|
| ELA visualization | Not shown | **Shown (3 Au + 3 Tp)** |
| Sample predictions | Not shown | **8 correct + 8 incorrect with confidence** |
| Model save | Commented out | **Active (vR1_ETASR_ela_cnn_model.keras)** |
| Data split | 80/20 train/val | **80/20 train/val (unchanged)** |
| ROC-AUC | Not computed | **Not computed (unchanged)** |
| Metric averaging | `weighted` | **`weighted` (unchanged)** |

---

## 2. Results Summary

### Final Metrics (Validation Set — NO test set)

| Metric | Value | Paper Claims | Gap |
|--------|-------|-------------|-----|
| **Accuracy** | **89.81%** | 96.21% | **-6.40pp** |
| Precision (weighted) | 0.9058 | 98.58% | -7.99pp |
| Recall (weighted) | 0.8981 | 92.36% | -2.55pp |
| F1 (weighted) | 0.8989 | 95.37% | -5.48pp |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Authentic | 0.9593 | 0.8652 | 0.9098 | 1,498 |
| Tampered | 0.8276 | 0.9463 | 0.8830 | 1,025 |
| Macro avg | 0.8935 | 0.9057 | 0.8964 | 2,523 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|---|---|---|
| **True Au** | 1,296 | 202 |
| **True Tp** | 55 | 970 |

- FP rate: 13.5% (202 authentic misclassified as tampered)
- FN rate: 5.4% (55 tampered missed)

---

## 3. Training Dynamics

| Epoch | Train Acc | Val Acc | Val Loss | Notes |
|-------|-----------|---------|----------|-------|
| 1 | 0.7320 | 0.8728 | 0.3381 | Fast initial learning |
| 4 | 0.8890 | 0.8926 | 0.2788 | |
| **8** | **0.9167** | **0.8981** | **0.2463** | **Best epoch (restored)** |
| 9 | 0.9211 | 0.8946 | 0.2434 | Slight val_acc decline |
| **11** | 0.9258 | **0.8514** | **0.3744** | **Val crash — 4.7% drop** |
| 12 | 0.9267 | 0.8886 | 0.2699 | Partial recovery |
| 13 | 0.9313 | 0.8783 | 0.2881 | Early stopping triggered |

**Key observations:**
- Nearly identical training dynamics to vR.ETASR (same seed, same split, same architecture)
- Epoch 11 instability: val accuracy drops from 0.8946 to 0.8514 (4.7% crash)
- Recovery at epoch 12, but never matches epoch 8 peak
- Train-val gap opens after epoch 8 — textbook overfitting
- Keras precision/recall values are identical to accuracy throughout (micro-average artifact)

---

## 4. Strengths

1. **All 19 code cells execute** — Zero crashes, zero errors
2. **ELA visualization present** — Shows 3 authentic + 3 tampered images with ELA maps
3. **Sample predictions shown** — 8 correct + 8 incorrect predictions with confidence scores (useful for failure analysis)
4. **Model saved** — `vR1_ETASR_ela_cnn_model.keras` persisted
5. **Reference bug documentation** — 11 bugs documented and fixed from the original reference code
6. **Clean code** — Centralized config, deterministic seeding, sorted file loading

---

## 5. Weaknesses and Issues

### MAJOR

| ID | Issue | Impact |
|----|-------|--------|
| M1 | **No test set** — Same as vR.ETASR. Val set used for both early stopping AND final metrics. Results are optimistically biased. | Evaluation methodology error |
| M2 | **`weighted` average metrics** — Headline numbers (P=0.9058, R=0.8981, F1=0.8989) inflate performance. Actual tampered precision is 0.8276. | Misleading comparison |
| M3 | **No ROC-AUC** — Standard binary classification metric still missing. | Incomplete evaluation |
| M4 | **No localization** — Classification only. Assignment requires pixel-level masks. | Assignment requirement failure |

### MINOR

| ID | Issue | Impact |
|----|-------|--------|
| m1 | Keras precision/recall (micro-averaged) are identical to accuracy — redundant metrics in training logs | Misleading training curves |
| m2 | No data augmentation | Missed opportunity to reduce overfitting |
| m3 | Class imbalance (1.46:1) not addressed | Contributes to FP/FN imbalance |
| m4 | `warnings.filterwarnings('ignore')` suppresses all warnings | Too aggressive |

---

## 6. Comparison with vR.ETASR Baseline

| Metric | vR.ETASR | vR.1 | Delta | Verdict |
|--------|----------|------|-------|---------|
| Accuracy | 89.89% | 89.81% | -0.08pp | Negligible (within noise) |
| Au Precision | 0.9607 | 0.9593 | -0.0014 | Negligible |
| Au Recall | 0.8652 | 0.8652 | 0.0000 | Identical |
| Au F1 | 0.9104 | 0.9098 | -0.0006 | Negligible |
| Tp Precision | 0.8279 | 0.8276 | -0.0003 | Negligible |
| Tp Recall | 0.9483 | 0.9463 | -0.0020 | Negligible |
| Tp F1 | 0.8840 | 0.8830 | -0.0010 | Negligible |
| FP rate | 13.5% | 13.5% | 0.0pp | Identical |
| FN rate | 5.2% | 5.4% | +0.2pp | Negligible |

**These two runs are essentially identical.** The tiny differences (0.08pp accuracy) are due to the different GPU platform (P100 vs T4) causing minor floating-point differences. Same seed, same split, same architecture, same data → same result. This is **exactly what reproducibility should look like**.

---

## 7. Paper Reproduction Score

**7/10** — Architecture faithfully matches the paper. Same evaluation methodology issues as vR.ETASR (no test set, weighted averaging). Results are consistent across runs.

---

## 8. Assignment Readiness Score

**3/10** — Identical to vR.ETASR. ELA viz and model save are improvements, but still missing localization (the core requirement), test set, and ROC-AUC.

---

## 9. Actionable Fixes

1. **Add test set** — 70/15/15 split. Use val only for early stopping, test for final metrics.
2. **Fix metrics** — Report per-class + macro. Drop `weighted` average from headline.
3. **Add ROC-AUC** — `sklearn.metrics.roc_auc_score`
4. **Add localization component** — ELA thresholding or GradCAM pseudo-masks.
5. **Add data augmentation** — The 80/20 split still only gives 10,091 training images.

---

## 10. Key Takeaway

vR.1 is a **cosmetic upgrade** over vR.ETASR. It adds visualization and model persistence, but makes zero changes to the evaluation methodology or model architecture. The results are within noise of the baseline. The real improvements come in vR.1.1 (eval fix) and beyond.
