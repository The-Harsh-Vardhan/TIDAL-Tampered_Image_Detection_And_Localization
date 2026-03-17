# vR.P.40.1 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | 0.72–0.75 (+3–6pp over P.3) | 30% |
| Neutral | 0.68–0.72 (marginal gain over P.3) | 45% |
| Negative | < 0.68 (regression from P.3) | 25% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.58 | 0.52–0.58 | < 0.52 |
| Pixel AUC | > 0.92 | 0.88–0.92 | < 0.88 |
| Image Accuracy | > 90% | 87–90% | < 87% |

## Success Criteria

- **POSITIVE** if Pixel F1 > 0.72 — EfficientNet-B4 provides meaningful capacity improvement over ResNet-34
- **NEUTRAL** if 0.68 < Pixel F1 < 0.72 — encoder capacity is not the primary bottleneck
- **NEGATIVE** if Pixel F1 < 0.68 — EfficientNet-B4 with frozen body is worse than ResNet-34

## Failure Modes

1. **Insufficient BN adaptation** — EfficientNet-B4 relies heavily on SE blocks which are frozen, limiting domain adaptation to BN statistics only
2. **Batch size sensitivity** — BATCH_SIZE=8 may produce noisy BN estimates
3. **Decoder mismatch** — EfficientNet-B4 skip connection channels differ from ResNet-34, SMP's decoder may not optimally fuse features

## Comparison Baselines

| Experiment | Encoder | Input | Pixel F1 | Relevance |
|------------|---------|-------|----------|-----------|
| vR.P.3 | ResNet-34 | ELA Q=90 (3ch) | 0.6920 | Direct comparison — same input, different encoder |
| vR.P.19 | ResNet-34 | MQ-RGB-ELA (9ch) | 0.7965 | Best overall — target for P.40.2 |
| vR.P.10 | ResNet-34 | ELA+CBAM | 0.7277 | Attention comparison — SE vs CBAM |

## Key Questions

1. **Does encoder capacity matter more than input preprocessing?** — Compare P.40.1 vs P.19 (encoder vs input)
2. **Is built-in SE attention sufficient?** — Compare P.40.1 vs P.10 (implicit vs explicit attention)
3. **What is the encoder ceiling with single-Q ELA?** — Establishes baseline for P.40.2 ablation
