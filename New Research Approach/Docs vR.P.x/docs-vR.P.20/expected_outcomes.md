# vR.P.20 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | 0.71–0.72 (+1–3pp over P.3) | 30% |
| Neutral | 0.68–0.71 (within ~1pp of P.3) | 45% |
| Negative | < 0.68 (regression from P.3) | 25% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.56 | 0.52–0.56 | < 0.52 |
| Pixel AUC | > 0.91 | 0.88–0.91 | < 0.88 |
| Image Accuracy | > 92% | 90–92% | < 90% |
| Image F1 | > 0.91 | 0.89–0.91 | < 0.89 |

## Success Criteria

- **POSITIVE** verdict if Pixel F1 > 0.7100 — magnitude decomposition captures forensic signal more efficiently than raw RGB ELA
- **NEUTRAL** if 0.68 < Pixel F1 < 0.7100 — decomposition preserves signal but doesn't improve over correlated RGB channels
- **NEGATIVE** if Pixel F1 < 0.68 — chrominance direction noise degrades the representation

## Failure Modes

1. **Low-magnitude noise** — In authentic regions where ELA magnitude is near zero, chrominance direction channels become pure noise (0/0 stabilized by eps), confusing the encoder.
2. **Information loss** — Decomposition may discard useful correlations between R, G, B error channels that the encoder could exploit in raw form.
3. **Normalization mismatch** — Mapping ChromaDir from [-1, 1] to [0, 255] may not align with ImageNet-pretrained conv1 expectations.
4. **Redundant magnitude** — The magnitude channel is essentially a grayscale version of ELA, which P.3's encoder already extracts internally. No new information for the network.

## Comparison Baselines

| Experiment | Input | Pixel F1 | Relevance |
|------------|-------|----------|-----------|
| vR.P.3 | ELA Q=90 RGB (3ch) | 0.6920 | Direct baseline — same ELA, different channel representation |
| vR.P.10 | ELA Q=90 RGB + augmentation | 0.7277 | Current best — upper reference |
| vR.P.1 | Raw RGB (3ch) | 0.4546 | Non-ELA baseline — lower bound |
| vR.P.19 | Multi-quality RGB ELA (9ch) | pending | Alternative ELA enhancement — compare approaches |

## Key Questions

1. **Does separating magnitude from chrominance improve localization?** — If yes, the network benefits from explicit decomposition of error intensity vs. error color.
2. **Is the magnitude channel alone sufficient?** — If Pixel F1 is similar to a single grayscale ELA channel, chrominance adds no value.
3. **Do chrominance directions capture forgery-type-specific signatures?** — Copy-move vs. splicing may show different chrominance patterns.
