# vR.P.23 — Expected Outcomes

## Primary Metric Targets

| Scenario | Pixel F1 | Confidence |
|----------|----------|------------|
| Positive | 0.70–0.72 (+0–3pp over P.3) | 20% |
| Neutral | 0.55–0.70 (moderate but below P.3) | 45% |
| Negative | < 0.55 (significant regression) | 35% |

## Secondary Metrics

| Metric | Positive | Neutral | Negative |
|--------|----------|---------|----------|
| Pixel IoU | > 0.55 | 0.40–0.55 | < 0.40 |
| Pixel AUC | > 0.90 | 0.80–0.90 | < 0.80 |
| Image Accuracy | > 90% | 80–90% | < 80% |
| Image F1 | > 0.90 | 0.80–0.90 | < 0.80 |

## Success Criteria

- **POSITIVE** verdict if Pixel F1 > 0.6920 (matches or exceeds P.3) — chrominance alone contains comparable forensic signal to ELA, a surprising and valuable finding
- **NEUTRAL** if 0.55 < Pixel F1 < 0.6920 — chrominance has some localization capability but is weaker than ELA
- **NEGATIVE** if Pixel F1 < 0.55 — YCbCr channels provide insufficient forensic signal for localization, similar to or worse than RGB baseline

## Failure Modes

1. **RGB equivalence** — YCbCr is a linear transform of RGB. The pretrained encoder may extract identical features, making P.23 results equivalent to P.1 (RGB baseline, Pixel F1 = 0.4546). If so, the experiment confirms that color space alone does not provide forensic signal.
2. **No ELA amplification** — Without JPEG re-compression and differencing, compression artifacts are not amplified. The network must learn forensic features from raw pixel intensities, which is fundamentally harder.
3. **Chroma resolution loss** — JPEG 4:2:0 chroma subsampling halves Cb/Cr resolution. Resizing to 384x384 further smooths these channels, potentially destroying the subsampling discontinuities at splice boundaries.
4. **Dataset variability** — CASIA2 images have heterogeneous source cameras, resolutions, and compression histories. Chroma patterns vary so widely that the network cannot learn a consistent forgery signature.

## Comparison Baselines

| Experiment | Input | Pixel F1 | Relevance |
|------------|-------|----------|-----------|
| vR.P.1 | Raw RGB (3ch) | 0.4546 | Critical comparison — if P.23 matches P.1, YCbCr adds nothing over RGB |
| vR.P.3 | ELA Q=90 RGB (3ch) | 0.6920 | Primary baseline — ELA vs raw chrominance |
| vR.P.10 | ELA Q=90 RGB + augmentation | 0.7277 | Current best — upper reference |
| vR.P.22 | SRM noise maps (3ch) | pending | Another non-ELA approach — compare non-ELA strategies |

## Key Questions

1. **Does YCbCr outperform RGB (P.1)?** — If P.23 > P.1, chrominance separation provides the encoder with a more useful representation than raw RGB, even without ELA.
2. **How large is the gap between YCbCr and ELA?** — This quantifies how much forensic signal comes from ELA amplification vs. color space transformation.
3. **Do Cb/Cr channels show splice-specific signatures?** — Per-channel analysis may reveal that chrominance is diagnostic for splicing but not copy-move, informing future fusion strategies.
4. **Would ELA in YCbCr space outperform ELA in RGB space?** — If P.23 shows chrominance value, a follow-up experiment applying ELA in YCbCr space could combine both signals.
