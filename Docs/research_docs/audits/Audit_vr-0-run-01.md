# Technical Audit: vR.0 (Run 01)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vr-0-image-tampering-detection-ela-and-a-cn-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, 2x Tesla T4 (13,757 MB each) |
| **Cells** | 31 total (20 code, 11 markdown) |
| **Executed** | 20 of 20 code cells (all executed) |
| **Training** | 13 epochs (early stopped), best at epoch 8 |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.0 is the **cleaned-up version** of the baseline, adding a proper 3-way train/val/test split, ROC-AUC metric, and ELA visualization. It represents the first methodology fix applied to the vR.ETASR baseline.

### Configuration

```
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

### Key Changes from vR.ETASR

| Change | vR.ETASR | vR.0 |
|--------|----------|------|
| Data split | 80/20 train/val | **70/15/15 train/val/test** |
| Evaluation set | Validation (biased) | **Test set (unbiased)** |
| ROC-AUC | Not computed | **Computed: 0.9600** |
| ELA visualization | Not shown | **Shown (3 Au + 3 Tp)** |
| Model save | Commented out | **Active** |
| Metric averaging | `weighted` only | **weighted** (unchanged — see issue) |

---

## 2. Results Summary

### Final Metrics (Test Set — Proper Hold-Out)

| Metric | Value | Paper Claims | Gap |
|--------|-------|-------------|-----|
| **Accuracy** | **88.33%** | 96.21% | **-7.88pp** |
| Precision (weighted) | 0.8847 | 98.58% | -10.11pp |
| Recall (weighted) | 0.8833 | 92.36% | -4.03pp |
| F1 (weighted) | 0.8836 | 95.37% | -7.01pp |
| **ROC-AUC** | **0.9600** | — | New metric |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Authentic | 0.9154 | 0.8852 | 0.9000 | 1,124 |
| Tampered | 0.8400 | 0.8804 | 0.8597 | 769 |
| Macro avg | 0.8777 | 0.8828 | 0.8799 | 1,893 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|---|---|---|
| **True Au** | 995 | 129 |
| **True Tp** | 92 | 677 |

- FP rate: 11.5% (improved from 13.5% in vR.ETASR)
- FN rate: 12.0% (increased from 5.2% — significant regression)

---

## 3. Training Dynamics

| Epoch | Train Acc | Val Acc | Val Loss | Notes |
|-------|-----------|---------|----------|-------|
| 1 | 0.7530 | 0.8325 | 0.3994 | |
| **8** | **0.9289** | **0.8811** | **0.2669** | **Best epoch** |
| 12 | 0.9313 | 0.8187 | 0.5064 | **Val collapse — 6.2% drop** |
| 13 | 0.9295 | 0.7955 | 0.6387 | **Early stopping, val still declining** |

**Critical observation:**
- Epochs 12-13 show catastrophic val loss explosion (0.2669 → 0.5064 → 0.6387)
- Val accuracy drops from 88.1% to 79.6% in just 2 epochs
- This is much worse instability than vR.ETASR (which only had one bad epoch)
- **Cause:** 10% less training data (70% vs 80%) makes the model more prone to overfitting oscillation

---

## 4. Strengths

1. **Proper test set** — 70/15/15 split fixes the biggest methodological flaw from vR.ETASR
2. **ROC-AUC computed** — 0.9600 is a strong threshold-independent discrimination score
3. **ELA visualization present** — Shows authentic vs tampered ELA maps
4. **Model saved** — `.keras` format, weights persisted
5. **All 20 code cells execute** — Zero crashes, zero errors
6. **Good section structure** — 10 clean sections, proper numbering (Section 3 Reference Code Audit restored)

---

## 5. Weaknesses and Issues

### MAJOR

| ID | Issue | Impact |
|----|-------|--------|
| M1 | **Primary metrics still use `weighted` average** — The headline numbers (P=0.8847, R=0.8833, F1=0.8836) inflate tampered precision. The actual tampered precision is 0.8400. | Same metric methodology problem as vR.ETASR |
| M2 | **FN rate almost tripled** — Tampered FN rate went from 5.2% (vR.ETASR) to 12.0%. The model now misses 92 of 769 tampered images vs 53 of 1,025. This is a real regression in tampered detection. | Performance regression |
| M3 | **Val collapse at epochs 12-13** — Loss explodes from 0.27 to 0.64 in 2 epochs. More unstable than baseline. | Training instability |
| M4 | **No localization** — Still classification only. | Assignment requirement failure |

### MINOR

| ID | Issue | Impact |
|----|-------|--------|
| m1 | Metric comparison table in cell-28 still references `acc`, `prec`, `rec`, `f1` variables — these are NOT defined in this notebook (they were from the old val-only split). The summary table will **crash** if the cell ordering changes. | Fragile code |
| m2 | No data augmentation | Missed opportunity |
| m3 | Class imbalance not addressed | Contributes to FP/FN imbalance |
| m4 | `warnings.filterwarnings('ignore')` suppresses all warnings | Too aggressive |

---

## 6. Comparison with vR.ETASR Baseline

| Metric | vR.ETASR (val) | vR.0 (test) | Delta | Verdict |
|--------|----------------|-------------|-------|---------|
| Accuracy | 89.89% | 88.33% | -1.56pp | Expected (unbiased test) |
| Au Precision | 0.9607 | 0.9154 | -0.0453 | Slight regression |
| Au Recall | 0.8652 | 0.8852 | +0.0200 | Improved |
| Au F1 | 0.9104 | 0.9000 | -0.0104 | Slight regression |
| Tp Precision | 0.8279 | 0.8400 | +0.0121 | Improved |
| Tp Recall | 0.9483 | 0.8804 | **-0.0679** | **Significant regression** |
| Tp F1 | 0.8840 | 0.8597 | -0.0243 | Regression |
| FP rate | 13.5% | 11.5% | -2.0pp | Improved |
| FN rate | 5.2% | 12.0% | **+6.8pp** | **Major regression** |

**The accuracy drop (1.56pp) is expected and acceptable** — it's the cost of honest evaluation on a held-out test set. However, **the tampered recall regression (-6.8pp FN rate increase) is concerning** and suggests the model is less sensitive to tampered images with less training data.

---

## 7. Paper Reproduction Score

**6/10** — Architecture unchanged. Test set fix improves methodology but reveals the model is weaker than the biased vR.ETASR numbers suggested. Metric averaging still uses `weighted`.

---

## 8. Assignment Readiness Score

**4/10** — Better than vR.ETASR (proper eval, ELA viz, model saved) but still missing localization.

---

## 9. Actionable Fixes

1. **Fix headline metrics** — Report per-class and macro averages as primary. Move `weighted` to secondary.
2. **Add data augmentation** — The reduced training set (70% vs 80%) needs augmentation to compensate.
3. **Investigate FN regression** — The tripled FN rate needs analysis. Consider class weights.
4. **Add localization component** — ELA thresholding or GradCAM pseudo-masks.
