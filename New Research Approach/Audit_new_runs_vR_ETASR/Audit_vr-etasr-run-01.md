# Technical Audit: vR.ETASR (Run 01)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vr-etasr-image-detection-and-localisation-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (15,511 MB VRAM) |
| **Cells** | 29 total (18 code, 11 markdown) |
| **Executed** | 17 of 18 code cells (model save commented out) |
| **Training** | 13 epochs (early stopped), best at epoch 8 |
| **Status** | **FULLY EXECUTED — MODEL CONVERGES TO 89.89%** |

---

## 1. Notebook Overview

This is the **original baseline reproduction** of the ETASR paper. It implements ELA preprocessing (JPEG quality 90) followed by a compact 2-layer CNN for binary classification of CASIA v2.0 images as authentic or tampered. This is the first run in the vR series and serves as the reference point for all subsequent ablation experiments.

### Configuration

```
IMAGE_SIZE       = (128, 128)
ELA_QUALITY      = 90
BATCH_SIZE       = 32
EPOCHS           = 50
LEARNING_RATE    = 0.0001
VALIDATION_SPLIT = 0.2  (80/20, no test set)
EARLY_STOP       = patience=5, monitor=val_accuracy
SEED             = 42
```

---

## 2. Results Summary

### Final Metrics (Validation Set — NO test set)

| Metric | Value | Paper Claims | Gap |
|--------|-------|-------------|-----|
| **Accuracy** | **89.89%** | 96.21% | **-6.32pp** |
| Precision (weighted) | 0.9068 | 98.58% | -7.90pp |
| Recall (weighted) | 0.8989 | 92.36% | -2.47pp |
| F1 (weighted) | 0.8997 | 95.37% | -5.40pp |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Authentic | 0.9607 | 0.8652 | 0.9104 | 1,498 |
| Tampered | 0.8279 | 0.9483 | 0.8840 | 1,025 |
| Macro avg | 0.8943 | 0.9067 | 0.8972 | 2,523 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|---|---|---|
| **True Au** | 1,296 | 202 |
| **True Tp** | 53 | 972 |

- FP rate: 13.5% (202 authentic misclassified as tampered)
- FN rate: 5.2% (53 tampered missed)

---

## 3. Training Dynamics

| Epoch | Train Acc | Val Acc | Val Loss | Notes |
|-------|-----------|---------|----------|-------|
| 1 | 0.7320 | 0.8720 | 0.3390 | Fast initial learning |
| 4 | 0.8884 | 0.8910 | 0.2801 | |
| **8** | **0.9160** | **0.8989** | **0.2473** | **Best epoch (restored)** |
| 11 | 0.9259 | 0.8565 | 0.3589 | **Val crash — 4.2% drop** |
| 13 | 0.9323 | 0.8831 | 0.2811 | Early stopping triggered |

**Key observations:**
- Overfitting begins at epoch 8 (train-val gap opens to ~2%)
- Epoch 11 instability: val accuracy drops 4.2% in one epoch, likely caused by the massive Flatten→Dense(256) layer (29.5M params)
- Train accuracy continues climbing while val accuracy stagnates → textbook overfitting

---

## 4. Strengths

1. **Architecture exactly matches paper** — Every layer, activation, and dimension matches Table III
2. **ELA implementation is correct** — In-memory BytesIO, proper PIL-based difference + brightness scaling
3. **Reference code bugs documented and fixed** — 11 bugs identified (3 fatal, 3 critical, 3 medium, 2 low)
4. **Model genuinely learns** — 89.89% accuracy is a real, honest result
5. **Clean code structure** — Centralized config, deterministic seeding, sorted file loading
6. **Zero ELA failures** — All 12,614 images processed successfully

---

## 5. Weaknesses and Issues

### MAJOR

| ID | Issue | Impact |
|----|-------|--------|
| M1 | **No test set** — Validation set used for both early stopping AND final metrics. Metrics are optimistically biased by 1-3%. | Evaluation methodology error |
| M2 | **`weighted` average metrics** — Not comparable to paper's per-class metrics. Tampered precision (0.8279) is hidden behind weighted average (0.9068). | Misleading comparison |
| M3 | **No localization** — Assignment requires pixel-level masks. This is classification only. | Assignment requirement failure |

### MINOR

| ID | Issue | Impact |
|----|-------|--------|
| m1 | No ROC-AUC computed | Missing standard binary classification metric |
| m2 | No ELA visualization | Paper's key contribution not visually demonstrated |
| m3 | Model save commented out | No weights persisted |
| m4 | Section numbering broken — duplicate "Section 7", Section 3 missing from body | Presentation quality |
| m5 | No data augmentation | Missed opportunity to reduce overfitting |
| m6 | Class imbalance (1.46:1) not addressed | Higher FP rate |
| m7 | Keras precision/recall equal accuracy in training logs (micro-average artifact) | Misleading training curves |

---

## 6. Paper Reproduction Score

| Category | Score | Notes |
|----------|-------|-------|
| Architecture fidelity | 10/10 | Exact match to Table III |
| Preprocessing fidelity | 10/10 | ELA Q=90, correct formula |
| Training config fidelity | 9/10 | Adam, lr, batch, loss all match. Patience value unspecified in paper. |
| Metric methodology | 5/10 | Wrong averaging, no test set, no ROC-AUC |
| Results reproduction | 6/10 | 6.32% accuracy gap unexplained |

**Overall: 7/10** — Architecture is perfect. Evaluation methodology has fundamental issues.

---

## 7. Assignment Readiness Score

**3/10** — Missing localization (the core requirement), no test set, no model saved.

---

## 8. Actionable Fixes

1. **Add test set** — 70/15/15 split. Use val only for early stopping, test for final metrics.
2. **Fix metrics** — Report per-class + macro. Drop `weighted` average.
3. **Add ROC-AUC** — `sklearn.metrics.roc_curve` + `auc`
4. **Add ELA visualization** — Show authentic vs tampered ELA maps
5. **Save model** — Uncomment the save cell
6. **Fix section numbering** — Restore Section 3, deduplicate Section 7
