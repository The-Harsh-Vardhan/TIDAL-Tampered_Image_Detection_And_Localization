# vR.P.15 — Experiment Description

## Multi-Quality ELA (3-Channel: Q=75, Q=85, Q=95)

### Hypothesis
Stacking ELA maps at three different JPEG quality factors as a 3-channel input provides **richer forensic signal** than single-quality ELA (Q=90), because different quality levels expose different artifact magnitudes — potentially yielding another breakthrough comparable to the RGB→ELA switch.

### Motivation
The #1 lesson from the entire ablation study: **input representation matters 10× more than anything else** (ELA switch produced +23.74pp Pixel F1). Currently all ELA experiments use a single quality factor (Q=90). However:

- **Q=75** (aggressive compression): Large ELA residuals — highlights strong manipulations, may wash out subtle ones
- **Q=85** (medium compression): Balanced residual magnitudes
- **Q=95** (light compression): Small residuals — sensitive to subtle edits, preserves fine detail

Each quality level acts as a different "lens" on the compression artifacts. Stacking them as 3 channels (like RGB) gives the encoder three complementary forensic signals instead of three correlated channels from a single ELA.

### Single Variable Changed from vR.P.3
**Input representation** — Replace 3-channel single-quality ELA (Q=90) with 3-channel multi-quality ELA (Q=75, Q=85, Q=95 as grayscale channels). Everything else unchanged.

### Key Configuration

| Parameter | P.3 (parent) | P.15 (this) |
|-----------|-------------|-------------|
| ELA input | Q=90 (3 channels, RGB) | Q=75/85/95 (3 channels, grayscale per quality) |
| IN_CHANNELS | 3 | 3 (same — compatible with pretrained encoder) |
| Image size | 384 | 384 |
| Encoder | ResNet-34 frozen + BN | ResNet-34 frozen + BN |
| Everything else | Same | Same |

### Why 3 Channels (Not 4+)?
The pretrained ResNet-34 expects 3-channel input. Using exactly 3 quality levels maps naturally to the encoder's `conv1` without architecture changes. Each channel carries an independent quality-level signal.
