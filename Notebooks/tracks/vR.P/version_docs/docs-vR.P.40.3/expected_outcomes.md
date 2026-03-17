# vR.P.40.3 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | 0.65–0.72 (competitive with pretrained) | 20% |
| Neutral | 0.55–0.65 (learns useful features) | 50% |
| Negative | < 0.55 (fails to converge) | 30% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.50 | 0.40–0.50 | < 0.40 |
| Pixel AUC | > 0.88 | 0.80–0.88 | < 0.80 |
| Image Accuracy | > 85% | 78–85% | < 78% |

## Success Criteria

- **POSITIVE** if Pixel F1 > 0.65 — from-scratch Inception is viable, multi-scale design helps
- **NEUTRAL** if 0.55 < Pixel F1 < 0.65 — learns something but data-limited
- **NEGATIVE** if Pixel F1 < 0.55 or training diverges — from-scratch approach is insufficient

## Failure Modes

1. **Training instability** — No BatchNorm + small dataset → gradient explosion or oscillation
2. **Underfitting** — 12K images insufficient to train a full encoder from scratch
3. **Overfitting** — 100% trainable params with small dataset → memorization
4. **MaxPool information loss** — MaxPool discards spatial position information critical for segmentation

## Comparison Baselines

| Experiment | Encoder | Pretrained | Input | Pixel F1 |
|------------|---------|------------|-------|----------|
| vR.P.19 | ResNet-34 | ImageNet | MQ-RGB-ELA 9ch | 0.7965 |
| vR.P.40.2 | EfficientNet-B4 | ImageNet | MQ-RGB-ELA 9ch | TBD |
| vR.P.40.3 | InceptionV1 | None | MQ-RGB-ELA 9ch | TBD |
| vR.P.40.4 | InceptionV2 | None | MQ-RGB-ELA 9ch | TBD |
| vR.P.40.5 | InceptionV3 | None | MQ-RGB-ELA 9ch | TBD |

## Key Questions

1. **Can from-scratch training compete with transfer learning on 12K images?** — Fundamental data efficiency question
2. **Does Inception's multi-scale design help for ELA?** — If V1 > simple CNN baseline, multi-scale matters
3. **How much does BN help?** — Compare V1 (no BN) vs V2 (with BN) directly
