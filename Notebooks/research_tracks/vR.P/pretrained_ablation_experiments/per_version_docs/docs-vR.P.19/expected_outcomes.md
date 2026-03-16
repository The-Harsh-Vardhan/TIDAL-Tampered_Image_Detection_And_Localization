# vR.P.19 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | 0.72–0.74 (+2–5pp over P.3) | 35% |
| Neutral | 0.68–0.72 (within ~1pp of P.3) | 40% |
| Negative | < 0.68 (regression from P.3) | 25% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.58 | 0.52–0.58 | < 0.52 |
| Pixel AUC | > 0.92 | 0.88–0.92 | < 0.88 |
| Image Accuracy | > 93% | 90–93% | < 90% |
| Image F1 | > 0.92 | 0.89–0.92 | < 0.89 |

## Success Criteria

- **POSITIVE** verdict if Pixel F1 > 0.7277 (exceeds current best P.10) — multi-quality ELA provides genuinely complementary forensic signal
- **NEUTRAL** if 0.68 < Pixel F1 < 0.7277 — additional quality levels add marginal signal but don't clearly surpass single-quality ELA
- **NEGATIVE** if Pixel F1 < 0.68 — 9-channel input destabilizes training or adds redundancy/noise

## Failure Modes

1. **Correlated channels** — Q=75, Q=85, Q=95 ELA may be too similar, making 6 of 9 channels redundant. The model learns the same features 3x instead of complementary features.
2. **conv1 instability** — 9-channel initialization from 3-channel weights may cause gradient explosion in early training despite 1/3 scaling.
3. **Preprocessing bottleneck** — 3x ELA computation per image may slow training enough to require reduced epochs, hurting convergence.
4. **Q=75 noise dominance** — Low-quality ELA introduces heavy noise on authentic regions, overwhelming the subtler Q=95 signal.

## Comparison Baselines

| Experiment | Input | Pixel F1 | Relevance |
|------------|-------|----------|-----------|
| vR.P.3 | ELA Q=90 (3ch) | 0.6920 | Direct baseline — same pipeline, single quality |
| vR.P.10 | ELA Q=90 (3ch) + augmentation | 0.7277 | Current best — target to beat |
| vR.P.4 | RGB+ELA (4ch) | 0.7053 | Prior multi-channel experiment — adding channels was marginal |
| vR.P.17 | ELA+DCT (6ch) | pending | Other multi-channel fusion — compare channel utility |

## Key Questions

1. **Does multi-quality ELA outperform single-quality ELA?** — If yes, the compression quality dimension contains meaningful forensic variation.
2. **Which quality level contributes most?** — Per-channel gradient analysis or ablation could reveal the most informative quality.
3. **Is 9-channel better than 3-channel grayscale multi-quality?** — Full RGB may be redundant per quality level; grayscale multi-quality (3ch) might be more efficient.
