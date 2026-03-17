# vR.P.40.4 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | 0.65–0.72 (strong from-scratch result) | 30% |
| Neutral | 0.58–0.65 (BN helps but data-limited) | 45% |
| Negative | < 0.58 (marginal improvement over V1) | 25% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.52 | 0.42–0.52 | < 0.42 |
| Pixel AUC | > 0.90 | 0.82–0.90 | < 0.82 |
| Image Accuracy | > 87% | 80–87% | < 80% |

## Success Criteria

- **POSITIVE** if Pixel F1 > P.40.3 + 5pp — BN and factorized convolutions provide clear benefit
- **NEUTRAL** if F1 within 5pp of P.40.3 — improvements are marginal
- **NEGATIVE** if F1 < P.40.3 — BN introduces overfitting or factorization loses signal

## Failure Modes

1. **BN overfitting** — BN with 100% trainable params on small dataset → memorization
2. **AvgPool blurring** — Average pooling may smooth out sharp tamper boundaries
3. **Factorization loss** — Two 3x3 may not perfectly replicate 5x5 receptive field for artifact detection

## Comparison Baselines

| Experiment | Key Difference | Pixel F1 |
|------------|---------------|----------|
| vR.P.40.3 | InceptionV1 (no BN, MaxPool) | TBD |
| vR.P.40.4 | InceptionV2 (BN, AvgPool, factorized) | TBD |
| vR.P.40.5 | InceptionV3 (asymmetric factorization) | TBD |

## Key Questions

1. **How much does BatchNorm improve from-scratch training?** — V2 vs V1 isolates BN contribution
2. **Does AvgPool preserve segmentation quality?** — Compare boundary sharpness with V1's MaxPool
3. **Is factorized 5x5 equivalent to full 5x5 for forensic features?** — Check if artifact patterns require specific kernel sizes
