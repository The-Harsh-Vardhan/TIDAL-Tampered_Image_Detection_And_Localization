# vR.P.40.2 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | > 0.80 (new best, beats P.19) | 35% |
| Neutral | 0.77–0.80 (competitive with P.19) | 40% |
| Negative | < 0.77 (worse than P.19) | 25% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.67 | 0.62–0.67 | < 0.62 |
| Pixel AUC | > 0.95 | 0.92–0.95 | < 0.92 |
| Image Accuracy | > 93% | 90–93% | < 90% |

## Success Criteria

- **POSITIVE** if Pixel F1 > 0.7965 — EfficientNet-B4 + Multi-Q RGB ELA is the new best
- **NEUTRAL** if 0.77 < Pixel F1 < 0.7965 — gains don't fully stack
- **NEGATIVE** if Pixel F1 < 0.77 — encoder/input combination has diminishing returns

## Failure Modes

1. **Diminishing returns** — EfficientNet-B4 SE attention may not add value on top of 9-channel input (P.19 already captures multi-scale features)
2. **Conv1 mismatch** — Pretrained 3ch weights tiled to 9ch may not adapt well with frozen encoder
3. **Memory limits** — 9ch + EfficientNet-B4 at batch=8 may require gradient accumulation

## Comparison Baselines

| Experiment | Encoder | Input | Pixel F1 |
|------------|---------|-------|----------|
| vR.P.3 | ResNet-34 | ELA Q=90 (3ch) | 0.6920 |
| vR.P.19 | ResNet-34 | MQ-RGB-ELA (9ch) | 0.7965 |
| vR.P.40.1 | EfficientNet-B4 | ELA Q=90 (3ch) | TBD |
| vR.P.40.2 | EfficientNet-B4 | MQ-RGB-ELA (9ch) | TBD |

## Key Questions

1. **Do encoder and input gains stack?** — Compare (P.40.2 - P.40.1) vs (P.19 - P.3)
2. **Is EfficientNet-B4 the best pretrained encoder for this task?** — Establishes ceiling for pretrained approach
3. **Can we beat 0.80 Pixel F1?** — Psychologically and practically important milestone
