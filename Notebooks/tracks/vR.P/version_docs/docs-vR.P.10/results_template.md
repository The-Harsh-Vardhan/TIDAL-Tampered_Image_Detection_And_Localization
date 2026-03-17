# vR.P.10 Results Table Template

Fill in after Kaggle run completes.

---

## Pixel-Level Metrics

| Metric | P.3 (Parent) | P.10 (This Run) | Delta | Assessment |
|--------|-------------|-----------------|-------|------------|
| Pixel F1 | 0.6920 | _____ | _____ | |
| Pixel IoU | 0.5291 | _____ | _____ | |
| Pixel AUC | 0.9528 | _____ | _____ | |
| Pixel Precision | 0.8356 | _____ | _____ | |
| Pixel Recall | 0.5905 | _____ | _____ | |

## Image-Level Metrics

| Metric | P.3 (Parent) | P.10 (This Run) | Delta | Assessment |
|--------|-------------|-----------------|-------|------------|
| Image Accuracy | 86.79% | _____ | _____ | |
| Image Macro F1 | 0.8560 | _____ | _____ | |
| Image ROC-AUC | 0.9502 | _____ | _____ | |

## Confusion Matrix

| | Predicted Au | Predicted Tp |
|---|---|---|
| **Actual Au** | TN = _____ | FP = _____ |
| **Actual Tp** | FN = _____ | TP = _____ |

| Rate | P.3 | P.10 | Delta |
|------|-----|------|-------|
| FP Rate | 2.7% | _____ | _____ |
| FN Rate | 28.6% | _____ | _____ |

## Training Summary

| Metric | P.3 | P.10 | Change |
|--------|-----|------|--------|
| Max Epochs | 25 | 25 | — |
| Patience | 7 | 7 | — |
| Epochs Trained | 25 | _____ | |
| Best Epoch | 25 | _____ | |
| Best Val Loss | 0.4109 | _____ | |
| Final LR | 2.5e-4 | _____ | |
| Attention Type | None | CBAM | NEW |
| Loss Function | BCE+Dice | Focal+Dice | NEW |

## Verdict

| Criterion | Threshold | P.10 Value | Met? |
|-----------|-----------|-----------|------|
| STRONG POSITIVE | Pixel F1 >= 0.7700 (+7.8pp) | _____ | |
| POSITIVE | Pixel F1 >= 0.7120 (+2.0pp) | _____ | |
| NEUTRAL | Pixel F1 in [0.6720, 0.7120] | _____ | |
| NEGATIVE | Pixel F1 < 0.6720 (-2.0pp) | _____ | |

**Verdict: _____**

## Key Observations

1. Attention module parameter count: _____ (expected ~11.2K)
2. Best epoch position: _____ (was 25 in P.3)
3. Training convergence speed: {faster/same/slower} than P.3
4. Spatial attention maps: {correlate/don't correlate} with tampered regions
5. Model saved: {yes/no} — filename: _____

## Comparison with Related Experiments

| Metric | P.3 (ELA) | P.9 (Focal+Dice) | P.10 (CBAM+Focal) | Winner |
|--------|-----------|-------------------|---------------------|--------|
| Pixel F1 | 0.6920 | _____ | _____ | |
| Image Accuracy | 86.79% | _____ | _____ | |
| Loss | BCE+Dice | Focal+Dice | Focal+Dice | — |
| Attention | None | None | CBAM | — |

**If P.10 > P.9:** Attention provides additional benefit beyond loss optimization.
**If P.10 ≈ P.9:** Focal loss captures most improvement; attention is redundant.
**If P.10 < P.9:** Attention interferes with optimization.

---

## Cross-Run Tracking Table (Copy to ablation_master_plan.md)

| Version | Change | Pixel-F1 | IoU | Pixel-AUC | Tam-F1 (cls) | Macro F1 (cls) | Test Acc | Epochs | Verdict |
|---------|--------|----------|-----|-----------|-------------|----------------|----------|--------|---------|
| vR.P.3 | ELA input (BN unfrozen) | 0.6920 | 0.5291 | 0.9528 | 0.8145 | 0.8560 | 86.79% | 25 (25) | STRONG POSITIVE |
| vR.P.9 | Focal + Dice loss | _____ | _____ | _____ | _____ | _____ | _____% | _____ | _____ |
| **vR.P.10** | **Focal+Dice + CBAM attention** | **_____** | **_____** | **_____** | **_____** | **_____** | **_____%** | **_____ (_____)** | **_____** |
