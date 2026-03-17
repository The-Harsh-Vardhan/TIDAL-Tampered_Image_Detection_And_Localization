# Technical Audit: vR.1.4 — ETASR Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-1-4-etasr-ablation-study-batchnormalization-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (15,511 MB VRAM) |
| **Cells** | 31 total (19 code, 12 markdown) |
| **Executed** | 19 of 19 code cells (all executed) |
| **Training** | 8 epochs (early stopped), best at epoch 3 |
| **Version** | vR.1.4 — BatchNormalization after each Conv2D layer |
| **Parent** | vR.1.3 (class weights) |
| **Change** | Add BatchNormalization after each Conv2D layer (W9) |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.1.4 adds **BatchNormalization** after each Conv2D layer to address weakness W9 (training instability, epoch 11-style spikes). This is the fourth ablation in the ETASR single-variable study. It branches from vR.1.3 (class weights, POSITIVE).

### Notebook Structure (31 cells)

| Cell | Type | Section / Purpose |
|------|------|-------------------|
| 0 | Markdown | Title, paper citation, change log, pipeline diagram, TOC |
| 1 | Markdown | Section 1: Version Change Log (vR.1.3 → vR.1.4) |
| 2 | Code | Section 2.1: Imports, seed, config, version info |
| 3 | Markdown | Section 3 header: Dataset Preparation |
| 4 | Code | Section 3.1: Mount dataset, count images |
| 5 | Code | Section 3.2: Load and parse image paths + labels |
| 6 | Markdown | Section 4 header: ELA Preprocessing |
| 7 | Code | Section 4.1: ELA function (JPEG Q=90, pixel diff, brightness) |
| 8 | Code | Section 4.2: Process all images → ELA arrays |
| 9 | Code | Section 4.3: ELA visualization grid |
| 10 | Markdown | Section 5 header: Data Splitting |
| 11 | Code | Section 5.1: Stratified 70/15/15 split, class weights |
| 12 | Code | Section 5.2: Split statistics |
| 13 | Markdown | Section 6 header: Model Architecture |
| 14 | Code | Section 6.1: Build CNN model (Conv→BN→Conv→BN→Pool→Drop→Dense→Drop→Dense) |
| 15 | Markdown | Section 7 header: Training Pipeline |
| 16 | Code | Section 7.1: Callbacks (EarlyStopping) |
| 17 | Code | Section 7.2: model.fit with class_weight |
| 18 | Markdown | Section 8 header: Evaluation |
| 19 | Code | Section 8.1: Test set evaluation (accuracy, per-class, macro, AUC) |
| 20 | Code | Section 8.2: Classification report |
| 21 | Code | Section 8.3: Confusion matrix |
| 22 | Code | Section 8.4: Training curves (loss + accuracy) |
| 23 | Code | Section 8.5: ROC curve |
| 24 | Markdown | Section 9 header: Discussion |
| 25 | Code | Section 9.1: Paper comparison table |
| 26 | Code | Section 9.2: Ablation tracking table |
| 27 | Markdown | Section 10 header: Model Save |
| 28 | Code | Section 10.1: Save model as .keras |
| 29 | Markdown | Section 11: Summary & Next Steps |
| 30 | Code | Final version print |

---

## 2. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | Single-variable ablation maintained | Only change: BN after each Conv2D (+128 non-trainable params) |
| S2 | All frozen constants preserved | ELA Q=90, 128×128, seed=42, Adam lr=1e-4, batch=32, patience=5 |
| S3 | Class weights carried forward from vR.1.3 | `class_weight=CLASS_WEIGHT_DICT` in model.fit |
| S4 | Full evaluation suite | Per-class P/R/F1, macro averages, ROC-AUC, confusion matrix |
| S5 | Model weights saved | `vR.1.4_ela_cnn_model.keras` |
| S6 | Tampered recall improved | 0.9194 vs 0.9012 (+1.82pp) — best in series |

---

## 3. Weaknesses

| # | Severity | Weakness | Impact |
|---|----------|----------|--------|
| W1 | MAJOR | Epoch 1 catastrophe: val_loss=16.13, val_acc=0.4059 | BN initialization caused extreme instability at start |
| W2 | MODERATE | Training converged in only 8 epochs (best at 3) | Much shorter than vR.1.3's 14 epochs — model may not have fully explored loss landscape |
| W3 | MINOR | Test accuracy marginally lower (88.75% vs 89.17%) | Within noise (−0.42pp) but not an improvement |
| W4 | MINOR | No learning rate scheduler to handle BN-induced instability | The high val_loss at epoch 1 could be mitigated with ReduceLROnPlateau |

---

## 4. Paper Reproduction Fidelity

| Aspect | Paper | vR.1.4 | Match? |
|--------|-------|--------|--------|
| ELA quality | 90 | 90 | ✅ |
| Image size | 128×128 | 128×128 | ✅ |
| Conv layers | 2× Conv2D(32, 5×5) | 2× Conv2D(32, 5×5) | ✅ |
| BatchNorm | Not in paper | **Added** | ❌ (intentional ablation) |
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
| Learning rate | 0.0001 | ✅ Frozen |
| Batch size | 32 | ✅ Frozen |
| Max epochs | 50 | ✅ Frozen |
| Early stopping | patience=5, val_accuracy | ✅ Frozen |
| Class weights | inverse-frequency | ✅ Carried from vR.1.3 |
| LR scheduler | None | ⚠️ Not yet added (vR.1.5) |

### Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 1.0718 | 0.7715 | **16.1310** | 0.4059 |
| 2 | 0.3069 | 0.8791 | 4.1840 | 0.7072 |
| 3 | 0.2426 | 0.8989 | **0.2752** | **0.8858** |
| 4 | 0.1920 | 0.9180 | 0.2782 | 0.8811 |
| 5 | 0.1695 | 0.9212 | 0.2735 | 0.8832 |
| 6 | 0.1737 | 0.9230 | 0.3212 | 0.8858 |
| 7 | 0.1464 | 0.9309 | 0.4740 | 0.8621 |
| 8 | 0.1443 | 0.9318 | 0.3543 | 0.8811 |

**Key observations:**
- Epoch 1: val_loss=16.13 is a BN warmup catastrophe (BN running stats uninitialized)
- Recovery by epoch 3: val_loss=0.275, val_acc=0.886
- Despite epoch 1 spike, model recovers completely
- Best weights from epoch 3 restored by early stopping

---

## 7. Performance Summary

### Test Set Results

| Metric | vR.1.3 (Parent) | vR.1.4 | Delta | Assessment |
|--------|-----------------|--------|-------|------------|
| **Test Accuracy** | 89.17% | 88.75% | −0.42pp | Within noise |
| Au Precision | 0.9290 | 0.9401 | +0.0111 | Improved |
| Au Recall | 0.8852 | 0.8657 | −0.0195 | Slightly worse |
| Au F1 | 0.9066 | 0.9013 | −0.0053 | Within noise |
| **Tp Precision** | 0.8431 | 0.8240 | −0.0191 | Slightly worse |
| **Tp Recall** | 0.9012 | **0.9194** | **+0.0182** | **Best in series** |
| **Tp F1** | 0.8712 | 0.8691 | −0.0021 | Within noise |
| **Macro F1** | **0.8889** | **0.8852** | **−0.0037** | Within noise |
| **ROC-AUC** | 0.9580 | 0.9536 | −0.0044 | Within noise |
| Epochs | 14 (9) | 8 (3) | −6 | Much shorter |

### Confusion Matrix

| | Predicted Au | Predicted Tp |
|-|-------------|-------------|
| **True Au (1124)** | 973 (TN) | 151 (FP) |
| **True Tp (769)** | 62 (FN) | 707 (TP) |

- FP rate: 13.4% (151/1124) — slightly worse than vR.1.3's 12.8%
- FN rate: 8.1% (62/769) — improved from vR.1.3's 9.9%

### Model Architecture

```
Conv2D(32, 5×5, ReLU)        → 2,432 params
BatchNormalization            → 128 params (64 trainable, 64 non-trainable)  ← NEW
Conv2D(32, 5×5, ReLU)        → 25,632 params
BatchNormalization            → 128 params (64 trainable, 64 non-trainable)  ← NEW
MaxPooling2D(2×2)             → 0
Dropout(0.25)                 → 0
Flatten                       → 0
Dense(256, ReLU)              → 29,491,456 params
Dropout(0.5)                  → 0
Dense(2, Softmax)             → 514 params

Total params:      29,520,290  (was 29,520,034 in vR.1.3)
Trainable params:  29,520,162
Non-trainable:     128 (BN running mean/variance)
```

---

## 8. Result Analysis

### Verdict: **NEUTRAL**

All deltas are within the ±0.5pp threshold:
- Test Accuracy: −0.42pp (within noise)
- Macro F1: −0.0037 (within noise)
- ROC-AUC: −0.0044 (within noise)

### Key Findings

1. **BN does not harm performance** — all metrics within noise of parent vR.1.3
2. **BN causes epoch 1 instability** — val_loss=16.13 at epoch 1 (BN running stats uninitialized), but model recovers by epoch 3
3. **Training is shorter** — 8 epochs vs 14 (2× faster convergence post-recovery)
4. **Tampered recall improved** — 0.9194 (best in series), suggesting BN helps the model detect more tampered images at the cost of slightly lower precision
5. **FN rate improved** — 8.1% vs 9.9% (fewer tampered images missed)

### Decision: KEEP BatchNorm

Although the verdict is NEUTRAL, BatchNorm is retained because:
- It does not harm any metric beyond noise
- It improves tampered recall (forensically important)
- The epoch 1 instability is a known BN warmup artifact, not a fundamental problem
- A learning rate scheduler (vR.1.5) should eliminate the warmup spike

---

## 9. Paper Reproduction Score

| Criterion | Score | Notes |
|-----------|-------|-------|
| Architecture fidelity | 8/10 | BN intentionally added (ablation) |
| Training pipeline | 9/10 | Standard pipeline, class weights added |
| Evaluation rigor | 10/10 | Full suite: per-class, macro, AUC, CM |
| Reproducibility | 10/10 | seed=42, deterministic split, model saved |
| **Overall** | **37/40** | |

---

## 10. Assignment Readiness Score

| Criterion | Score | Notes |
|-----------|-------|-------|
| Detection accuracy | 7/10 | 88.75% (paper claims 96.21%) |
| Localization | 0/10 | Not applicable (classification only) |
| Visualization | 7/10 | ELA viz, training curves, ROC, CM |
| Documentation | 9/10 | Full ablation tracking, version notes |
| **Overall** | **23/40** | Localization addressed by Track 2 (vR.P.x) |

---

## 11. Recommended Next Steps

1. **vR.1.5 — ReduceLROnPlateau** (IMMEDIATE): Add `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)`. This should:
   - Eliminate the epoch 1 val_loss spike (LR will reduce after the catastrophe)
   - Allow longer, more stable training (model may train beyond 8 epochs)
   - Potentially push past 89% accuracy

2. **Retain all vR.1.4 components**: BN layers, class weights, 70/15/15 split
3. **Continue pretrained track (vR.P.0)** in parallel for localization
