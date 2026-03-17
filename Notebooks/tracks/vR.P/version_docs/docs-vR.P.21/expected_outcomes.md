# vR.P.21 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | 0.72–0.73 (+2–4pp over P.3) | 35% |
| Neutral | 0.68–0.72 (within ~1pp of P.3) | 35% |
| Negative | < 0.68 (regression from P.3) | 30% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.57 | 0.52–0.57 | < 0.52 |
| Pixel AUC | > 0.92 | 0.88–0.92 | < 0.88 |
| Image Accuracy | > 93% | 90–93% | < 90% |
| Image F1 | > 0.92 | 0.89–0.92 | < 0.89 |

## Success Criteria

- **POSITIVE** verdict if Pixel F1 > 0.7277 (exceeds current best P.10) — Laplacian residual captures boundary information that raw ELA misses
- **NEUTRAL** if 0.68 < Pixel F1 < 0.7277 — high-pass filtering preserves forensic signal but doesn't clearly outperform raw ELA
- **NEGATIVE** if Pixel F1 < 0.68 — noise amplification from Laplacian overwhelms the forensic signal

## Failure Modes

1. **Noise amplification** — The Laplacian is a second-order derivative operator that amplifies high-frequency noise. If the ELA signal-to-noise ratio is already low, Laplacian filtering may destroy the useful signal.
2. **Texture false positives** — High-frequency textures (foliage, fabric, hair) produce strong Laplacian responses even in authentic regions, potentially degrading precision.
3. **Loss of magnitude information** — By discarding the smooth, low-frequency ELA component, the network loses absolute error intensity, which is itself a strong indicator of tampering.
4. **Per-image normalization instability** — Dividing by per-image max creates inconsistent scaling: an image with one extreme pixel dominates the normalization.

## Comparison Baselines

| Experiment | Input | Pixel F1 | Relevance |
|------------|-------|----------|-----------|
| vR.P.3 | ELA Q=90 RGB (3ch) | 0.6920 | Direct baseline — raw ELA without high-pass filtering |
| vR.P.10 | ELA Q=90 RGB + augmentation | 0.7277 | Current best — target to exceed |
| vR.P.20 | ELA Magnitude (3ch) | pending | Alternative ELA decomposition — compare signal extraction strategies |
| vR.P.1 | Raw RGB (3ch) | 0.4546 | Non-ELA baseline — lower bound |

## Key Questions

1. **Does high-pass filtering of ELA improve boundary localization?** — If Pixel F1 improves while IoU also improves, the Laplacian successfully sharpens forgery boundaries.
2. **Does the network already learn high-pass features internally?** — If P.21 matches P.3, the pretrained encoder already extracts edge information from raw ELA, making explicit filtering redundant.
3. **Is the noise-signal tradeoff favorable?** — Laplacian amplifies both signal and noise; the experiment determines which dominates in practice.
