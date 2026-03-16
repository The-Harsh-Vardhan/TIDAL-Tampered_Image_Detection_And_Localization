# vR.P.40.5 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | 0.65–0.72 (best from-scratch result) | 30% |
| Neutral | 0.58–0.65 (comparable to V2) | 45% |
| Negative | < 0.58 (asymmetric factorization hurts) | 25% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.52 | 0.42–0.52 | < 0.42 |
| Pixel AUC | > 0.90 | 0.82–0.90 | < 0.82 |
| Image Accuracy | > 87% | 80–87% | < 80% |

## Success Criteria

- **POSITIVE** if Pixel F1 >= P.40.4 with fewer parameters — asymmetric factorization provides regularization benefit
- **NEUTRAL** if F1 within 2pp of P.40.4 — comparable performance, validates the approach
- **NEGATIVE** if F1 < P.40.4 by > 3pp — asymmetric kernels lose important forensic features

## Inception Series Summary (Expected Ranking)

| Version | Key Feature | Expected F1 | Rationale |
|---------|------------|-------------|-----------|
| V1 (P.40.3) | No BN, MaxPool | 0.55–0.65 | Unstable training |
| V2 (P.40.4) | BN, AvgPool, factorized 5x5 | 0.60–0.70 | BN stabilizes |
| V3 (P.40.5) | BN, AvgPool, asymmetric 1xn+nx1 | 0.60–0.72 | Best regularization |

## Failure Modes

1. **Directional bias** — 1xn + nx1 captures horizontal/vertical edges but misses diagonal tamper boundaries
2. **Over-regularization** — Too few parameters may underfit on complex tamper patterns
3. **n selection sensitivity** — Wrong n for a given stage may miss relevant scale features

## Full Phase 4 Comparison Matrix

| Exp | Encoder | Pretrained | Input | Expected F1 |
|-----|---------|------------|-------|-------------|
| P.40.1 | EfficientNet-B4 | ImageNet | ELA Q=90 3ch | 0.68–0.75 |
| P.40.2 | EfficientNet-B4 | ImageNet | MQ-RGB-ELA 9ch | 0.77–0.82 |
| P.40.3 | InceptionV1 | None | MQ-RGB-ELA 9ch | 0.55–0.65 |
| P.40.4 | InceptionV2 | None | MQ-RGB-ELA 9ch | 0.60–0.70 |
| P.40.5 | InceptionV3 | None | MQ-RGB-ELA 9ch | 0.60–0.72 |

## Key Questions

1. **Is V3 the best from-scratch Inception?** — Validates progressive improvement hypothesis
2. **How large is the pretrained vs from-scratch gap?** — Compare P.40.2 vs best of P.40.3-5
3. **Does directional decomposition help for tamper detection?** — If V3 > V2, directional features matter
4. **What is the minimum dataset size for from-scratch encoder training?** — Informs future data collection strategy
