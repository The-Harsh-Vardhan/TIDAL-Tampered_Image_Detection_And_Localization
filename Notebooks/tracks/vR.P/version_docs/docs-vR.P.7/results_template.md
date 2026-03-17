# vR.P.7 Results Table Template

Fill in after Kaggle run completes.

---

## Pixel-Level Metrics

| Metric | P.3 (Parent) | P.7 (This Run) | Delta | Assessment |
|--------|-------------|----------------|-------|------------|
| Pixel F1 | 0.6920 | _____ | _____ | |
| Pixel IoU | 0.5291 | _____ | _____ | |
| Pixel AUC | 0.9528 | _____ | _____ | |
| Pixel Precision | 0.8356 | _____ | _____ | |
| Pixel Recall | 0.5905 | _____ | _____ | |

## Image-Level Metrics

| Metric | P.3 (Parent) | P.7 (This Run) | Delta | Assessment |
|--------|-------------|----------------|-------|------------|
| Image Accuracy | 86.79% | _____ | _____ | |
| Image Macro F1 | 0.8560 | _____ | _____ | |
| Image ROC-AUC | 0.9502 | _____ | _____ | |

## Confusion Matrix

| | Predicted Au | Predicted Tp |
|---|---|---|
| **Actual Au** | TN = _____ | FP = _____ |
| **Actual Tp** | FN = _____ | TP = _____ |

| Rate | P.3 | P.7 | Delta |
|------|-----|-----|-------|
| FP Rate | 2.7% | _____ | _____ |
| FN Rate | 28.6% | _____ | _____ |

## Training Summary

| Metric | P.3 | P.7 | Change |
|--------|-----|-----|--------|
| Max Epochs | 25 | 50 | +25 |
| Patience | 7 | 10 | +3 |
| Epochs Trained | 25 | _____ | |
| Best Epoch | 25 | _____ | |
| Best Val Loss | 0.4109 | _____ | |
| Final LR | 2.5e-4 | _____ | |
| LR Reductions | 2 | _____ | |

## LR Schedule

| Reduction | Epoch | LR Before | LR After |
|-----------|-------|-----------|----------|
| 1 | ~9 | 1e-3 | 5e-4 |
| 2 | ~20 | 5e-4 | 2.5e-4 |
| 3 | _____ | 2.5e-4 | 1.25e-4 |
| 4 | _____ | 1.25e-4 | 6.25e-5 |
| 5 | _____ | _____ | _____ |

## Verdict

| Criterion | Threshold | P.7 Value | Met? |
|-----------|-----------|-----------|------|
| STRONG POSITIVE | Pixel F1 >= 0.7500 (+5.8pp) | _____ | |
| POSITIVE | Pixel F1 >= 0.7120 (+2.0pp) | _____ | |
| NEUTRAL | Pixel F1 in [0.6720, 0.7120] | _____ | |
| NEGATIVE | Pixel F1 < 0.6720 (-2.0pp) | _____ | |

**Verdict: _____**

## Key Observations

1. Best epoch position: _____ (was 25 in P.3, extended training was {needed/not needed})
2. LR reductions after epoch 25: _____ (each reduction = new learning region)
3. Train-val gap at best epoch: _____pp (P.3 gap at epoch 25: ~___pp)
4. Pixel F1 trajectory shape: {logarithmic/linear/stepped}
5. Model saved: {yes/no} — filename: _____

## Comparison with P.4 (4ch RGB+ELA)

| Metric | P.4 | P.7 | Winner |
|--------|-----|-----|--------|
| Pixel F1 | 0.7053 | _____ | |
| Image Accuracy | 84.42% | _____ | |
| Complexity | 4ch, dual norm, conv1 unfreeze | 3ch, single norm, BN only | P.7 (simpler) |

If P.7 > P.4 on Pixel F1: **ELA-only + sufficient training beats 4-channel fusion** — a significant finding.

---

## Cross-Run Tracking Table (Copy to ablation_master_plan.md)

| Version | Change | Pixel-F1 | IoU | Pixel-AUC | Tam-F1 (cls) | Macro F1 (cls) | Test Acc | Epochs | Verdict |
|---------|--------|----------|-----|-----------|-------------|----------------|----------|--------|---------|
| vR.P.3 | ELA input (BN unfrozen) | 0.6920 | 0.5291 | 0.9528 | 0.8145 | 0.8560 | 86.79% | 25 (25) | STRONG POSITIVE |
| vR.P.4 | 4ch RGB+ELA (conv1+BN) | 0.7053 | 0.5447 | 0.9433 | 0.7873 | 0.8322 | 84.42% | 25 (24) | NEUTRAL |
| **vR.P.7** | **Extended training (50ep, pat=10)** | **_____** | **_____** | **_____** | **_____** | **_____** | **_____%** | **_____ (_____)** | **_____** |
