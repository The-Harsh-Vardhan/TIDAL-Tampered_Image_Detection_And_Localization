# Technical Audit: vR.P.1 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-1-detection-localisation-dataset-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (17.1 GB VRAM) |
| **Training** | 25 epochs (all ran), best at epoch 18 |
| **Version** | vR.P.1 (labeled vK.P.1 in notebook) |
| **Parent** | vR.P.0 (ResNet-34 UNet baseline, divg07 dataset) |
| **Change** | Fix dataset: use sagnikkayalcse52 with real GT masks (100% coverage) |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.P.1 fixes the dataset by switching from divg07 (142 tampered images without GT masks requiring ELA pseudo-masks) to **sagnikkayalcse52** which provides real annotated GT masks for **all 5,123 tampered images** (100% coverage). This establishes the **proper baseline** for all subsequent pretrained experiments.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | ResNet-34 (ImageNet, FROZEN) |
| Input | RGB 384x384, ImageNet normalization |
| Loss | SoftBCEWithLogitsLoss + DiceLoss |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5, decoder only) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | 25 |
| Early stopping | patience=7, val_loss |
| Seed | 42 |
| Dataset | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |

### Parameters

| Category | Count |
|----------|-------|
| Total | 24,436,369 |
| Trainable (decoder) | 3,151,697 |
| Frozen (encoder) | 21,284,672 |

---

## 3. Strengths

| # | Strength |
|---|----------|
| S1 | 100% GT mask coverage (0 ELA pseudo-masks) — clean training signal |
| S2 | Pixel F1 improved +7.97pp over vR.P.0 (0.4546 vs 0.3749) |
| S3 | Pixel IoU improved +6.35pp (0.2942 vs 0.2307) |
| S4 | Pixel recall improved +8.88pp (0.3545 vs 0.2657) |
| S5 | Longest training (25 epochs, best at 18) — model used all available epochs |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **MAJOR** | Image-level accuracy slightly dropped: 70.15% vs 70.63% (P.0) |
| W2 | **MAJOR** | Image-level FN rate 40.4% — still misses 2 in 5 tampered images |
| W3 | **MAJOR** | Severe overfitting: train loss 0.38 vs val loss 0.88 by epoch 25 |
| W4 | MODERATE | Per-image pixel F1 highly variable (mean=0.2534, std=0.3437) |
| W5 | MODERATE | Pixel recall still low at 0.3545 — misses 65% of tampered pixels |

---

## 5. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | LR |
|-------|-----------|----------|----------|-----|
| 11 | 0.7362 | 0.8689 | 0.3768 | 1e-3 |
| **18** (best) | 0.5022 | **0.8345** | 0.4077 | 5e-4 |
| 25 (final) | 0.3800 | 0.8808 | 0.3831 | 2.5e-4 |

LR reduced twice: epoch 16 (1e-3 -> 5e-4), epoch 23 (5e-4 -> 2.5e-4). All 25 epochs ran.

---

## 6. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.0 |
|--------|-------|----------------|
| Pixel Precision | 0.6335 | -0.0029 |
| Pixel Recall | 0.3545 | **+0.0888** |
| **Pixel F1** | **0.4546** | **+0.0797** |
| **Pixel IoU** | **0.2942** | **+0.0635** |
| **Pixel AUC** | **0.8509** | **+0.0023** |

### Image-Level (Classification)

| Metric | Value | Delta from P.0 |
|--------|-------|----------------|
| **Test Accuracy** | **70.15%** | -0.48pp |
| **Macro F1** | **0.6867** | +0.0053 |
| **ROC-AUC** | **0.7785** | -0.0075 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.7367 | 0.7740 | 0.7549 | 1,124 |
| Tampered | 0.6433 | 0.5956 | 0.6185 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 870 (TN) | 254 (FP) |
| **Tp** | 311 (FN) | 458 (TP) |

---

## 7. Verdict: **PROPER BASELINE**

vR.P.1 is not evaluated for POSITIVE/NEGATIVE — it's a dataset fix that establishes the clean baseline. Key outcomes:
- Pixel F1 improved significantly (+0.08) from proper GT masks
- Image-level accuracy slightly dropped but this is a localization model, not a classifier
- All subsequent pretrained experiments branch from vR.P.1 (not P.0)
- The sagnikkayalcse52 dataset with 100% GT coverage is confirmed as the standard
