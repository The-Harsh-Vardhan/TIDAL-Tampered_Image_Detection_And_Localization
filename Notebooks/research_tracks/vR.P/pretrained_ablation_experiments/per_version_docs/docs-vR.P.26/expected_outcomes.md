# vR.P.26 — Expected Outcomes

## Scenarios

### Positive (35% confidence)
- **Pixel F1: 0.70–0.71** (+1-2pp over P.3 baseline 0.6920)
- **Image Accuracy: 94–95%** (+2-3pp over P.3 baseline ~92%)
- Classification head acts as a regularizer, forcing the encoder to learn globally discriminative features
- Segmentation benefits from the auxiliary classification signal
- Both tasks improve simultaneously

### Neutral (40% confidence)
- **Pixel F1: 0.68–0.70**
- **Image Accuracy: 93–94%** (+1-2pp)
- Classification task is too easy relative to segmentation; cls head converges quickly and stops providing useful gradients
- Segmentation performance is unchanged, but classification improves slightly
- The dual-task setup adds complexity without clear segmentation benefit

### Negative (25% confidence)
- **Pixel F1: < 0.68**
- **Image Accuracy: < 92%**
- Classification gradient conflicts with segmentation through shared encoder
- CLS_WEIGHT=0.5 causes encoder to prioritize global features over local spatial features
- Both tasks degrade or segmentation degrades while classification stays flat

## Primary Metric Targets

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Pixel F1 | > 0.7020 (+1pp) | > 0.7120 (+2pp) |
| Pixel IoU | > 0.57 | > 0.58 |
| Image Accuracy | > 0.94 (+2pp) | > 0.95 (+3pp) |
| Image F1 | > 0.93 | > 0.94 |

## Secondary Metrics

- Classification loss convergence: should plateau faster than segmentation loss
- Per-class classification precision/recall: balanced or biased toward authentic?
- Segmentation quality on correctly-classified vs misclassified images
- Training time overhead: minimal (cls head is lightweight)

## Success Criteria

- POSITIVE verdict if Pixel F1 > 0.7020 AND Image Accuracy > 0.94
- NEUTRAL if either metric improves but the other stays flat
- The classification head is inherently useful even if segmentation doesn't improve — having a reliable image-level verdict is valuable for deployment

## Failure Modes

1. **Gradient conflict**: Classification and segmentation pull encoder in different directions. Mitigation: reduce CLS_WEIGHT to 0.1 or use gradient reversal on cls branch.
2. **Trivial classification**: CASIA2 image-level classification is already ~92% — the cls head may converge to near-perfect accuracy in 2-3 epochs and stop providing useful gradients. Mitigation: use label smoothing on cls target.
3. **Shape mismatch**: SMP encoder output format varies by backbone — must verify encoder feature list indexing. Mitigation: print feature shapes before building cls head.

## Comparison Baselines

- vR.P.3 (ELA baseline): Pixel F1 = 0.6920, Image Acc ~92%
- vR.P.10 (current best pixel): Pixel F1 = 0.7277
- Multi-task learning literature suggests 1-3pp improvement is typical for auxiliary tasks
- If both metrics improve: dual-task is a strong candidate for the final pipeline
