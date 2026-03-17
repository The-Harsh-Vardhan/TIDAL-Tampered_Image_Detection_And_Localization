# vR.P.15 — Expected Outcomes

## Baseline
**vR.P.3:** Pixel F1 = 0.6920, IoU = 0.5291, Pixel AUC = 0.9528, Image Acc = 86.79%
**vR.P.10 (best):** Pixel F1 = 0.7277, IoU = 0.5719, Pixel AUC = 0.9573, Image Acc = 87.32%

## Scenarios

### Positive (expected, 50% confidence)
- **Pixel F1: 0.72–0.80** (+3 to +11pp from P.3)
- Multi-quality channels expose different artifact levels the single-quality ELA misses
- The encoder's conv1 (trained on RGB) learns to distinguish the three quality-level channels
- Potential for another breakthrough if multi-quality captures fundamentally richer signal

### Neutral (30% confidence)
- **Pixel F1: 0.67–0.72** (within ±2pp of P.3)
- The three quality levels are too correlated — similar information in each channel
- The BN adaptation is sufficient for single-quality, and multi-quality doesn't add value

### Negative (20% confidence)
- **Pixel F1 < 0.67** (worse than P.3)
- Grayscale channels lose color information that single-quality RGB ELA preserves
- The frozen encoder's conv1 weights (trained on RGB) cannot interpret the quality-level channels
- ELA normalization statistics are fundamentally different, confusing the BN adaptation

## Success Criteria
- **Strong positive:** Pixel F1 ≥ 0.78 (would indicate multi-quality is a major leap)
- **Positive:** Pixel F1 ≥ P.3 + 2pp = 0.7120
- **Neutral:** Within ±2pp of P.3
- **Negative:** Pixel F1 < P.3 − 2pp = 0.6720

## Risk Assessment
**MODERATE** — fundamentally changes the input signal (from correlated RGB ELA to independent quality-level channels). The encoder's conv1 was trained on RGB, not grayscale quality maps. BN adaptation may or may not be sufficient.

## Key Insight
If this works, it confirms that **quality-level diversity is more valuable than color information** in ELA maps. If it fails, it suggests the encoder's RGB features depend on inter-channel color correlation.
