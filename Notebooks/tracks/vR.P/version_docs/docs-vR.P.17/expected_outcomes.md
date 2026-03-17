# vR.P.17 — Expected Outcomes

## Scenarios

### Positive (40% confidence)
- **Pixel F1: 0.72–0.78**
- DCT provides genuinely complementary information to ELA
- The model learns to weight both channels appropriately
- Fusion exceeds the best single-input result (P.10 at 0.7277)

### Neutral (40% confidence)
- **Pixel F1: 0.68–0.72**
- DCT adds marginal signal that doesn't overcome the noise from 6-channel input
- Similar to P.4 (4-channel RGB+ELA) which scored 0.7053 — competitive but not clearly better

### Negative (20% confidence)
- **Pixel F1: < 0.68**
- 6-channel input confuses the encoder (conv1 instability)
- DCT channels introduce systematic artifacts from upsampling

## Historical Parallel

vR.P.4 (4-channel RGB+ELA) achieved Pixel F1 = 0.7053 vs P.3's 0.6920 (+0.0133). The fusion was NEUTRAL. This experiment (ELA+DCT) may follow a similar pattern — adding channels doesn't guarantee improvement when the primary signal (ELA) is already strong.

## Success Criteria

- POSITIVE if Pixel F1 > 0.7277 (exceeds current best P.10)
- NEUTRAL if 0.68 < F1 < 0.7277
- Any result provides valuable data on whether DCT complements ELA for localization
