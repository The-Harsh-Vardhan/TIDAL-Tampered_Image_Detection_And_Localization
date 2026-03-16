# vR.P.25 — Expected Outcomes

## Scenarios

### Positive (35% confidence)
- **Pixel F1: 0.70–0.72** (+1-3pp over P.3 baseline 0.6920)
- Edge supervision sharpens forgery boundaries, reducing false negatives at edges
- The model maintains interior fill quality while improving boundary precision
- Boundary IoU (if measured) improves significantly

### Neutral (45% confidence)
- **Pixel F1: 0.68–0.70**
- Edge loss marginally improves boundaries but the effect is absorbed by existing Dice loss
- The Dice loss already provides some boundary signal, so explicit edge supervision is redundant
- No degradation but no clear improvement

### Negative (20% confidence)
- **Pixel F1: < 0.68**
- LAMBDA_EDGE=0.3 is too aggressive, causing the model to focus on edges at the expense of interior regions
- Predictions become "outline-only" with hollow interiors
- The additional loss term destabilizes training convergence

## Primary Metric Targets

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Pixel F1 | > 0.7020 (+1pp) | > 0.7220 (+3pp) |
| Pixel IoU | > 0.57 | > 0.59 |
| Pixel AUC | > 0.89 | > 0.91 |

## Secondary Metrics

- Boundary quality: visual inspection of prediction edges (sharper = better)
- Interior fill ratio: predictions should not become hollow (compare to P.3 overlay plots)
- Training loss curves: edge loss component should decrease steadily
- Image-level accuracy: should remain >= P.3 baseline (~92%)

## Success Criteria

- POSITIVE verdict if Pixel F1 > 0.7020 AND boundary quality visually improves
- NEUTRAL if within +/- 1pp of P.3 (0.6820 < F1 < 0.7020)
- Edge supervision is a lightweight addition — even neutral results validate the approach for combination with other improvements

## Failure Modes

1. **Hollow predictions**: Edge loss overweighted causes model to predict boundary rings instead of filled regions. Mitigation: reduce LAMBDA_EDGE to 0.1.
2. **Training instability**: Edge BCE loss has different scale than Dice+BCE. Mitigation: normalize edge loss by number of edge pixels.
3. **No effect**: Sobel edges on binary masks produce trivially thin edges that the model already handles. Mitigation: dilate edge maps (3x3 kernel) to create wider supervision bands.

## Comparison Baselines

- vR.P.3 (ELA baseline): Pixel F1 = 0.6920
- vR.P.10 (current best): Pixel F1 = 0.7277
- This is a loss-only modification — any improvement is purely from better boundary supervision
- If positive, edge loss can be combined with other improvements (architecture, input, etc.)
