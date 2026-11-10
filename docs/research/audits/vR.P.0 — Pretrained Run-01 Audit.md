# Technical Audit: vR.P.0 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-0-dataset-with-no-gt-available-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (17.1 GB VRAM) |
| **Training** | 24 epochs (early stopped), best at epoch 17 |
| **Version** | vR.P.0 — ResNet-34 UNet Baseline |
| **Parent** | None (first pretrained experiment) |
| **Change** | Establish pretrained localization baseline using UNet + frozen ResNet-34 encoder |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.P.0 is the **first pretrained localization experiment**, establishing the baseline for Track 2. It uses a UNet architecture with a frozen ResNet-34 encoder (ImageNet pretrained) and trains only the decoder to produce pixel-level tampering masks on RGB 384x384 input.

**Note:** The filename says "dataset-with-no-gt-available" but the run actually found and used GT masks for 4,981 of 5,123 tampered images. Only 142 images fell back to ELA pseudo-masks.

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
| Dataset | CASIA v2.0 (divg07) — 12,614 images |

### Parameters

| Category | Count |
|----------|-------|
| Total | 24,436,369 |
| Trainable (decoder) | 3,151,697 |
| Frozen (encoder) | 21,284,672 |

**Note:** Documentation estimates "~500K" decoder params but actual count is 3.15M.

---

## 3. Strengths

| # | Strength |
|---|----------|
| S1 | Established the pretrained localization baseline |
| S2 | Pixel AUC 0.8486 shows reasonable discriminative ability |
| S3 | Pixel precision 0.6364 — when it predicts tampering, it's mostly correct |
| S4 | Image-level accuracy 70.63% from pure segmentation (no classifier) |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **MAJOR** | Pixel recall only 0.2657 — misses ~73% of tampered pixels |
| W2 | **MAJOR** | Image-level FN rate 47.5% — misses nearly half of tampered images |
| W3 | **MAJOR** | Severe overfitting: train loss 0.40 vs val loss 0.87 |
| W4 | MODERATE | Volatile val metrics (pixel F1 oscillates 0.19-0.37 across epochs) |
| W5 | MODERATE | 142 tampered images use ELA pseudo-masks (not real GT) |
| W6 | MINOR | Parameter documentation error (~500K claimed vs 3.15M actual) |

---

## 5. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | LR |
|-------|-----------|----------|----------|-----|
| **17** (best) | 0.5928 | **0.8678** | 0.3511 | 1e-3 |
| 22 | 0.4487 | 0.8690 | 0.3667 | 5e-4 |
| 24 (final) | 0.4006 | 0.8714 | 0.3631 | 5e-4 |

LR reduced once (epoch 22: 1e-3 -> 5e-4). Early stopping at epoch 24.

---

## 6. Test Results

### Pixel-Level (Localization)

| Metric | Value |
|--------|-------|
| Pixel Precision | 0.6364 |
| Pixel Recall | 0.2657 |
| **Pixel F1** | **0.3749** |
| **Pixel IoU** | **0.2307** |
| **Pixel AUC** | **0.8486** |

### Image-Level (Classification)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **70.63%** |
| **Macro F1** | **0.6814** |
| **ROC-AUC** | **0.7860** |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.7188 | 0.8301 | 0.7704 | 1,124 |
| Tampered | 0.6790 | 0.5254 | 0.5924 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 933 (TN) | 191 (FP) |
| **Tp** | 365 (FN) | 404 (TP) |

---

## 7. Verdict: **BASELINE**

First pretrained experiment. Establishes the reference point for Track 2. Key takeaways:
- Frozen ResNet-34 + UNet decoder can produce pixel-level predictions but with low recall
- The model is conservative (high precision, low recall) — it prefers not predicting tampering
- Image-level classification from segmentation output is significantly below ETASR track (70.63% vs 88.38%)
- The 142 ELA pseudo-masks introduce noise — fixed in vR.P.1 with sagnikkayalcse52 dataset
