# vR.P.27 — Expected Outcomes

## Scenarios

### Positive (40% confidence)
- **Pixel F1 (standard test): 0.70–0.72** (+1-3pp over P.3 baseline 0.6920)
- **Pixel F1 (compressed test sets): +5-10pp** over P.3 on Q=70-90 conditions (as measured by P.18-style robustness testing)
- Model learns compression-invariant forensic features
- Augmentation acts as regularization, slightly improving standard test performance too

### Neutral (35% confidence)
- **Pixel F1 (standard test): 0.68–0.70**
- **Pixel F1 (compressed test sets): +2-5pp** improvement on compressed conditions
- Standard performance unchanged but robustness improves moderately
- The augmentation helps with compressed inputs but doesn't transfer to standard evaluation

### Negative (25% confidence)
- **Pixel F1 (standard test): < 0.68**
- JPEG augmentation at low Q levels introduces too much noise into ELA features
- The model is confused by the variety of compression artifacts during training
- Standard performance degrades without sufficient robustness gains to compensate

## Primary Metric Targets

| Metric | Condition | Target | Stretch Goal |
|--------|-----------|--------|-------------|
| Pixel F1 | Standard test | > 0.7020 (+1pp) | > 0.7220 (+3pp) |
| Pixel F1 | Q=90 recompressed | > 0.67 | > 0.69 |
| Pixel F1 | Q=80 recompressed | > 0.60 | > 0.65 |
| Pixel F1 | Q=70 recompressed | > 0.50 | > 0.55 |
| Pixel IoU | Standard test | > 0.57 | > 0.59 |

## Secondary Metrics

- Training convergence speed: may be 2-5 epochs slower than P.3 due to augmentation noise
- Per-epoch training loss variance: expected to be slightly higher (different compression levels per epoch)
- Image-level accuracy: should remain >= 92%
- Robustness curve shape: should be flatter (more graceful degradation) than P.3/P.18 results

## Success Criteria

- POSITIVE verdict if Pixel F1 (standard) > 0.70 AND robustness improves by > 3pp at Q=80
- NEUTRAL if standard performance unchanged but robustness improves by 1-3pp
- The primary value of this experiment is robustness, not standard benchmark performance
- Even neutral standard results are valuable if robustness improves significantly

## Failure Modes

1. **Excessive noise at low Q**: Q=50 compression may destroy manipulation traces entirely, making ground truth masks inaccurate for augmented images. Mitigation: raise JPEG_AUG_MIN_Q to 70.
2. **ELA double-compression artifact**: ELA on pre-compressed images produces a "double compression" pattern that is different from standard ELA. The model may overfit to this artifact. Mitigation: ensure val/test do not use augmentation.
3. **Slower convergence**: Augmentation noise increases variance, requiring more epochs. Mitigation: increase EPOCHS or PATIENCE if needed.

## Comparison Baselines

- vR.P.3 (ELA baseline): Pixel F1 = 0.6920 (standard), robustness unknown until P.18 results
- vR.P.10 (current best): Pixel F1 = 0.7277 (standard)
- vR.P.18 (robustness test): provides degradation curves for P.3 model at various Q levels
- P.27 should be compared against P.3 at every compression level, not just standard test
- If P.27 robustness > P.3 robustness at Q=70-80: strong evidence for deployment-ready model
