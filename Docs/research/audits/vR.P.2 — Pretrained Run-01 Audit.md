# Technical Audit: vR.P.2 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-2-gradual-encoder-unfreeze-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (17.1 GB VRAM) |
| **Training** | 14 epochs (early stopped), best at epoch 7 |
| **Version** | vR.P.2 — Gradual Encoder Unfreeze |
| **Parent** | vR.P.1 (dataset fix + GT masks, frozen encoder) |
| **Change** | Unfreeze encoder layer3 + layer4 with differential LR (encoder 1e-5, decoder 1e-3) |
| **Status** | **PARTIALLY EXECUTED — Cell 20 KeyError crashed cells 22-27** |

---

## 1. Notebook Overview

vR.P.2 tests **gradual encoder unfreezing** by making the last 2 ResNet-34 blocks (layer3 + layer4) trainable at a 100x lower learning rate than the decoder. The hypothesis: the deep encoder layers can be fine-tuned for forensic features while preserving general visual features in the early layers.

**Critical issue:** A `KeyError: 'lr'` in the training curves plot cell (cell 20) crashed the remaining cells. Prediction visualizations, per-image metrics, results summary, and model save were **never executed**.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | ResNet-34 (ImageNet, PARTIALLY UNFROZEN) |
| Input | RGB 384x384, ImageNet normalization |
| Loss | SoftBCEWithLogitsLoss + DiceLoss |
| Optimizer | Adam with 2 param groups |
| | - Encoder (layer3+4): lr=1e-5, weight_decay=1e-5 |
| | - Decoder: lr=1e-3, weight_decay=1e-5 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3, both groups) |
| Batch size | 16 |
| Max epochs | 25 |
| Early stopping | patience=7, val_loss |
| Seed | 42 |
| AMP | Enabled (from P.1.5) |
| Dataset | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |

### Unfreeze Strategy

| Encoder Layer | Parameters | Status | LR |
|--------------|-----------|--------|-----|
| conv1 | 9,408 | FROZEN | — |
| bn1 | 128 | FROZEN | — |
| layer1 | 221,952 | FROZEN | — |
| layer2 | 1,116,416 | FROZEN | — |
| **layer3** | **6,822,400** | **UNFROZEN** | **1e-5** |
| **layer4** | **13,114,368** | **UNFROZEN** | **1e-5** |

### Parameters

| Category | Count |
|----------|-------|
| Total | 24,436,369 |
| **Trainable** | **23,088,465** (94.5%) |
| — Encoder (layer3+4) | 19,936,768 |
| — Decoder | 3,151,697 |
| Frozen | 1,347,904 (5.5%) |
| Data:param ratio | 1:2,615 |

**Documentation error:** Notebook header claims "~5M trainable" and "1:570 ratio". Actual: **23M trainable**, **1:2,615 ratio** — 4.6x worse than documented.

---

## 3. Strengths

| # | Strength |
|---|----------|
| S1 | **Best pixel F1 in pretrained series: 0.5117** (+0.0571 from P.1) |
| S2 | **Best pixel IoU: 0.3439** (+0.0497 from P.1) |
| S3 | **Best pixel AUC: 0.8688** (+0.0179 from P.1) |
| S4 | **Best pixel recall: 0.4317** (+0.0772 from P.1) — detects more tampered pixels |
| S5 | Fast initial convergence — best epoch at 7, showing encoder adaptation works |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **CRITICAL** | Cells 22-27 never executed — model NOT saved, no visualizations |
| W2 | **MAJOR** | Severe overfitting: train-val gap grew from 0.10 to 0.59 across 14 epochs |
| W3 | **MAJOR** | Image-level accuracy DROPPED: 69.04% vs P.1's 70.15% (-1.11pp) |
| W4 | **MAJOR** | Data:param ratio 1:2,615 — extremely unfavorable for 8,829 training images |
| W5 | MODERATE | Volatile val metrics: pixel F1 swings from 0.32 to 0.49 between epochs |
| W6 | MODERATE | Image-level FN rate 47.5% — nearly half of tampered images missed |
| W7 | MINOR | `KeyError: 'lr'` bug in history plotting (should be `lr_encoder`/`lr_decoder`) |

---

## 5. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | Enc LR | Dec LR |
|-------|-----------|----------|----------|--------|--------|
| 2 | 0.8524 | 0.8525 | 0.3831 | 1e-5 | 1e-3 |
| 4 | 0.6780 | 0.8198 | 0.4300 | 1e-5 | 1e-3 |
| **7** (best) | 0.4711 | **0.7688** | **0.4750** | 1e-5 | 1e-3 |
| 12 | 0.2847 | 0.8519 | 0.4084 | 5e-6 | 5e-4 |
| 14 (final) | 0.2589 | 0.8526 | 0.4269 | 5e-6 | 5e-4 |

LR reduced at epoch 12 (both groups halved). Early stopping at epoch 14.

---

## 6. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.1 |
|--------|-------|----------------|
| Pixel Precision | 0.6283 | -0.0052 |
| **Pixel Recall** | **0.4317** | **+0.0772** |
| **Pixel F1** | **0.5117** | **+0.0571** |
| **Pixel IoU** | **0.3439** | **+0.0497** |
| **Pixel AUC** | **0.8688** | **+0.0179** |

### Image-Level (Classification)

| Metric | Value | Delta from P.1 |
|--------|-------|----------------|
| **Test Accuracy** | **69.04%** | **-1.11pp** |
| **Macro F1** | **0.6673** | **-0.0194** |
| **ROC-AUC** | **0.7196** | **-0.0589** |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.7121 | 0.8034 | 0.7550 | 1,124 |
| Tampered | 0.6464 | 0.5254 | 0.5796 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 903 (TN) | 221 (FP) |
| **Tp** | 365 (FN) | 404 (TP) |

---

## 7. Verdict: **POSITIVE (Pixel-Level) / NEGATIVE (Image-Level)**

A split verdict:

**Pixel metrics (localization — primary task):**
- Pixel F1: +5.71pp over P.1 (0.5117 vs 0.4546) — clear POSITIVE
- Pixel IoU: +4.97pp — clear POSITIVE
- Pixel recall: +7.72pp — substantial improvement

**Image metrics (classification — secondary):**
- Accuracy: -1.11pp — shifted to NEGATIVE
- ROC-AUC: -0.059 — significant regression

**Interpretation:** Encoder unfreezing helps the model localize tampered regions more precisely (better pixel-level predictions) but the mask-to-classification pipeline (>100 pixels threshold) is less calibrated. The model produces more activated pixels overall (higher recall) but this confuses the simple thresholding classifier.

### Key Insight: Unfreezing is Too Aggressive

The 23M trainable parameter count (vs P.1's 3.15M) with only 8,829 training images creates a 1:2,615 data:param ratio that causes severe overfitting. The pixel F1 improvement suggests the concept works, but the implementation needs refinement:
- Consider unfreezing only layer4 (13.1M) instead of layer3+layer4 (19.9M)
- Consider lower encoder LR (1e-6 instead of 1e-5)
- Address the model save bug (KeyError) in future versions
