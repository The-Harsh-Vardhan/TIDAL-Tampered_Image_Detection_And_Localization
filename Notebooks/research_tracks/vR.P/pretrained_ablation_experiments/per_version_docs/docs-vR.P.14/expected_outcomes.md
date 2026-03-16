# vR.P.14 — Expected Outcomes

## Baseline
**vR.P.3:** Pixel F1 = 0.6920, IoU = 0.5291, Pixel AUC = 0.9528, Image Acc = 86.79%
**vR.P.10 (best):** Pixel F1 = 0.7277, IoU = 0.5719, Pixel AUC = 0.9573, Image Acc = 87.32%

## Scenarios

### Positive (expected, 65% confidence)
- **Pixel F1: 0.70–0.72** (+1 to +3pp from P.3 baseline)
- Boundary predictions become smoother and more consistent
- IoU improves proportionally

### Neutral (25% confidence)
- **Pixel F1: 0.69–0.70** (within ±1pp of P.3)
- TTA views cancel out rather than reinforce — model already orientation-invariant
- The averaging washes out fine details

### Negative (10% confidence)
- **Pixel F1 < 0.69** (worse than P.3)
- Flipped views produce inconsistent predictions that degrade when averaged
- ELA signal is orientation-sensitive (unlikely given pixel-level diff)

## Success Criteria
- **Strong positive:** Pixel F1 ≥ 0.72
- **Positive:** Pixel F1 ≥ P.3 + 1pp = 0.7020
- **Neutral:** Within ±1pp of P.3
- **Negative:** Pixel F1 < P.3 − 1pp = 0.6820

## Risk Assessment
**VERY LOW** — no retraining, no architecture change, reversible (just remove TTA at inference).

## Note
TTA can also be applied ON TOP of P.13's trained model. This notebook tests it on P.3 to isolate TTA's effect, but the technique transfers to any model.
