# vR.P.22 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | 0.71–0.73 (+1–4pp over P.3) | 25% |
| Neutral | 0.60–0.71 (moderate performance) | 45% |
| Negative | < 0.60 (significant regression) | 30% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.56 | 0.45–0.56 | < 0.45 |
| Pixel AUC | > 0.91 | 0.82–0.91 | < 0.82 |
| Image Accuracy | > 92% | 82–92% | < 82% |
| Image F1 | > 0.91 | 0.82–0.91 | < 0.82 |

## Success Criteria

- **POSITIVE** verdict if Pixel F1 > 0.7100 — SRM filters capture manipulation traces that ELA misses, potentially complementary
- **NEUTRAL** if 0.60 < Pixel F1 < 0.7100 — SRM has some localization signal but is inferior to ELA for JPEG forgery detection
- **NEGATIVE** if Pixel F1 < 0.60 — SRM noise maps fail to provide actionable localization on CASIA2 dataset

## Failure Modes

1. **Domain mismatch** — SRM filters were designed for steganographic embedding detection (subtle bit-level changes), not for JPEG forgery localization (block-level compression artifacts). The signal type may not transfer.
2. **Uniform noise amplification** — High-pass filters amplify sensor noise and compression artifacts across the entire image, not just in forged regions, drowning out the manipulation signal.
3. **Grayscale limitation** — Converting to grayscale before filtering discards chrominance artifacts that are diagnostic of splicing (color temperature mismatch, chroma subsampling differences).
4. **Filter redundancy** — The three SRM kernels may produce highly correlated responses, effectively providing a single-channel signal spread across 3 channels.

## Comparison Baselines

| Experiment | Input | Pixel F1 | Relevance |
|------------|-------|----------|-----------|
| vR.P.3 | ELA Q=90 RGB (3ch) | 0.6920 | Direct baseline — ELA vs SRM comparison |
| vR.P.16 | DCT features (3ch) | pending | Another non-ELA representation — compare approaches |
| vR.P.21 | ELA Laplacian (3ch) | pending | Similar high-pass concept applied to ELA — compare |
| vR.P.1 | Raw RGB (3ch) | 0.4546 | Non-forensic baseline — SRM should exceed this |

## Key Questions

1. **Can SRM filters localize JPEG forgeries without explicit JPEG re-compression (ELA)?** — If yes, SRM captures more general manipulation traces.
2. **Which SRM filter is most informative?** — Per-channel analysis reveals whether 1st, 2nd, or 3rd order residuals are most diagnostic.
3. **Is SRM complementary to ELA?** — Even if SRM alone is weaker, it may detect different forgery types, motivating an ELA+SRM fusion experiment.
4. **Does SRM perform differently on copy-move vs. splicing?** — Copy-move preserves noise statistics; SRM may struggle. Splicing introduces noise mismatch; SRM may excel.
