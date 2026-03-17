# vR.P.13 — Expected Outcomes

## Baseline
**vR.P.3:** Pixel F1 = 0.6920, IoU = 0.5291, Pixel AUC = 0.9528, Image Acc = 86.79%
**vR.P.10 (best so far):** Pixel F1 = 0.7277, IoU = 0.5719, Pixel AUC = 0.9573, Image Acc = 87.32%

## Scenarios

### Positive (expected, 60% confidence)
- **Pixel F1: 0.74–0.80** (+1.2 to +7.2pp from P.10)
- IoU: 0.58–0.66
- CBAM + augmentation + extended training stack to produce the best model yet
- Augmentation prevents overfitting during extended training, CBAM amplifies forensic features

### Neutral (30% confidence)
- **Pixel F1: 0.72–0.74** (within ±2pp of P.10)
- Changes interfere — augmentation dilutes the signal CBAM learned to attend to
- Extended training compensates for slower convergence but doesn't improve peak

### Negative (10% confidence)
- **Pixel F1 < 0.72** (worse than P.10)
- Combined complexity causes optimization difficulties (too many moving parts)
- Augmentation + CBAM interaction is adversarial (blur confuses attention maps)

## Success Criteria
- **Strong positive:** Pixel F1 ≥ 0.78 (new record by ≥5pp)
- **Positive:** Pixel F1 ≥ P.10 + 2pp = 0.7477
- **Neutral:** Within ±2pp of P.10
- **Negative:** Pixel F1 < P.10 − 2pp = 0.7077

## Risk Assessment
**LOW** — combines only proven-positive changes with known code patterns.
