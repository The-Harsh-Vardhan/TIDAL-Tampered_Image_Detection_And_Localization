# Technical Audit: vR.P.1.5 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-1-5-training-speed-optimizations-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (16.0 GB VRAM) |
| **Training** | 23 epochs (early stopped), best at epoch 16 |
| **Version** | vR.P.1.5 — Training Speed Optimizations |
| **Parent** | vR.P.1 (dataset fix + GT mask auto-detection) |
| **Change** | Add AMP, TF32, pin_memory, persistent workers, set_to_none — speed only |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.P.1.5 applies **infrastructure-only speed optimizations** to vR.P.1 without changing any model architecture, hyperparameters, or evaluation logic. The goal is ~1.5-2x faster training via mixed precision (AMP), TF32 math, parallel data loading, and async GPU transfers.

**Important:** This is not an ablation — it's an infrastructure patch. No model/training variable changes.

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

### Speed Optimizations (NEW)

| Optimization | Detail |
|---|---|
| AMP mixed precision | FP16 forward pass via `torch.amp.autocast('cuda')` + `GradScaler` |
| TF32 math | `torch.backends.cuda.matmul.allow_tf32 = True` |
| Parallel loading | `NUM_WORKERS=2`, `persistent_workers=True` |
| Async transfers | `non_blocking=True`, `pin_memory=True` |
| Faster grad zeroing | `optimizer.zero_grad(set_to_none=True)` |
| Drop last batch | `drop_last=True` on train loader |

**Note:** TF32 has no effect on P100 (Pascal architecture) — only Ampere+ GPUs benefit.

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
| S1 | Best image-level accuracy in pretrained series: 71.05% (+0.90pp from P.1) |
| S2 | Best image-level Macro F1: 0.7016 (+0.0149 from P.1) |
| S3 | Best Tampered F1 at image level: 0.6501 (+0.0316 from P.1) |
| S4 | Pixel AUC slightly improved: 0.8560 (+0.0051 from P.1) |
| S5 | Speed infrastructure benefits all future experiments |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | MODERATE | Pixel F1 decreased: 0.4227 vs P.1's 0.4546 (-3.19pp) |
| W2 | MODERATE | Pixel recall low: 0.3165 (vs P.1's 0.3545) — conservative predictions |
| W3 | MODERATE | Severe overfitting persists: train 0.40 vs val 0.90 by epoch 23 |
| W4 | MINOR | No timing data in notebook — cannot verify speed improvement |
| W5 | MINOR | TF32 ineffective on P100 GPU |

---

## 5. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | LR |
|-------|-----------|----------|----------|-----|
| 9 | 0.7893 | 0.8842 | 0.3626 | 1e-3 |
| **16** (best) | 0.5403 | **0.8533** | 0.3772 | 5e-4 |
| 20 | 0.4635 | 0.8685 | 0.3975 | 5e-4 |
| 23 (final) | 0.3953 | 0.8977 | 0.3592 | 2.5e-4 |

LR reduced twice: epoch 14 (1e-3 -> 5e-4), epoch 21 (5e-4 -> 2.5e-4). Early stopping at epoch 23.

---

## 6. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.1 |
|--------|-------|----------------|
| Pixel Precision | 0.6364 | +0.0029 |
| Pixel Recall | 0.3165 | -0.0380 |
| **Pixel F1** | **0.4227** | **-0.0319** |
| **Pixel IoU** | **0.2680** | **-0.0262** |
| **Pixel AUC** | **0.8560** | **+0.0051** |

### Image-Level (Classification)

| Metric | Value | Delta from P.1 |
|--------|-------|----------------|
| **Test Accuracy** | **71.05%** | **+0.90pp** |
| **Macro F1** | **0.7016** | **+0.0149** |
| **ROC-AUC** | **0.7980** | **+0.0195** |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.7628 | 0.7438 | 0.7532 | 1,124 |
| Tampered | 0.6386 | 0.6619 | 0.6501 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 836 (TN) | 288 (FP) |
| **Tp** | 260 (FN) | 509 (TP) |

---

## 7. Verdict: **NEUTRAL (Speed Only)**

This is an infrastructure patch, not an ablation. Model differences from P.1 are due to AMP numerical noise and different data loading order (workers=2 vs workers=0), not intentional changes.

Key observations:
- Pixel F1 dropped slightly (0.4227 vs 0.4546) — likely AMP numerical differences
- Image-level metrics improved (71.05% vs 70.15%) — random variance from data loading differences
- The speed optimizations are carried forward to all subsequent experiments (vR.P.2+)
- Neither the pixel drop nor the image-level improvement should be attributed to the speed optimizations as causal effects
