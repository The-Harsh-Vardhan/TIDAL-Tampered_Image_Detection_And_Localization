# vR.P.16 — Expected Outcomes

## Scenarios

### Positive (30% confidence)
- **Pixel F1: 0.55–0.65**
- DCT features capture compression artifact patterns that are spatially localized
- The model learns to distinguish single-compressed from double-compressed blocks
- AC energy and HF energy channels provide meaningful gradients for the decoder

### Neutral (40% confidence)
- **Pixel F1: 0.40–0.55**
- DCT maps are too low-resolution (48x48 upsampled to 384x384) for precise boundaries
- The encoder can extract some signal but block-boundary artifacts dominate
- Classification accuracy may be reasonable (~75-80%) but localization is imprecise

### Negative (30% confidence)
- **Pixel F1: < 0.40**
- Blockwise DCT statistics are too coarse for pixel-level localization
- The bilinear upsampling creates artificial patterns that mislead the encoder
- The model essentially fails to localize

## Comparison Baseline

- vR.P.1 (RGB baseline): Pixel F1 = 0.4546
- vR.P.3 (ELA input): Pixel F1 = 0.6920
- If P.16 exceeds P.1 but falls below P.3: DCT has signal but ELA is superior
- If P.16 exceeds P.3: DCT may be a better representation (unexpected but valuable)

## Success Criteria

- POSITIVE verdict if Pixel F1 > P.3 + 0.005 (> 0.697)
- NEUTRAL if within +/- 0.005 of P.3
- Any Pixel F1 > 0.45 indicates DCT has localization signal (even if below P.3)
- Classification accuracy > 75% indicates DCT has detection signal
