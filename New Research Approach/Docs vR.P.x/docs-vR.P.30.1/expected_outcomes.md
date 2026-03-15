# vR.P.30.1 -- Expected Outcomes

## Metric Targets

| Metric | Expected Range | Success Threshold |
|--------|---------------|-------------------|
| Pixel F1 | 0.78-0.80 | > 0.7379 (+0.5pp over P.15) |
| Pixel IoU | TBD | > 0.5835 (+0.5pp over P.15) |
| Pixel AUC | > 0.96 | > 0.9608 (match P.15) |
| Image Accuracy | > 87% | > 87.53% (match P.15) |

## Success Criteria

- **POSITIVE:** Pixel F1 > 0.7379 (P.15 + 0.5pp)
- **STRONG POSITIVE:** Pixel F1 > 0.76 (additive combination)
- **NEUTRAL:** Pixel F1 within +/-0.5pp of P.15 (0.7329)
- **NEGATIVE:** Pixel F1 < 0.7279 (P.15 - 0.5pp)

## Failure Modes

1. Diminishing returns; overfitting possible with CBAM extra parameters and more epochs
2. CBAM adds ~50K parameters -- may overfit on small CASIA dataset
3. Frozen encoder conv1 processes decorrelated multi-Q ELA channels suboptimally
