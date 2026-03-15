# vR.P.19 -- Experiment Description

## Multi-Quality RGB ELA (Q=75/85/95 Full-Color)

### Hypothesis

Multi-quality ELA using full RGB channels (instead of P.15's grayscale) preserves chrominance artifact information that is lost in luminance-only analysis. Tampering often introduces chrominance discontinuities visible across quality levels, providing 9 information-rich channels (3 qualities x 3 RGB channels).

### Motivation

P.15 uses grayscale ELA at 3 quality levels (Q=75/85/95) to fit 3 channels. However, color information carries forensic signal -- splicing from different-quality sources creates chrominance boundary artifacts that grayscale ELA discards. By using a 9-channel input (or selecting the 3 most informative channels via PCA/learned projection), we can capture both luminance and chrominance artifacts across quality levels.

### Single Variable Changed from vR.P.3

**Input representation** -- Replace 3-channel single-quality ELA (Q=90 RGB) with 9-channel multi-quality RGB ELA (Q=75/85/95 full-color). Conv1 modified to accept 9 channels.

### Key Configuration

| Parameter | P.3 (parent) | P.19 (this) |
|-----------|-------------|-------------|
| ELA input | Q=90 (3ch RGB) | Q=75/85/95 (9ch RGB, 3 per quality) |
| IN_CHANNELS | 3 | 9 |
| conv1 | Frozen (pretrained 3ch) | Unfrozen (9ch, initialized from 3x pretrained) |
| Everything else | Same | Same |

### Pipeline

```
Image -> ELA(Q=75) RGB -> 3ch
      -> ELA(Q=85) RGB -> 3ch  -> Concatenate -> 9ch input
      -> ELA(Q=95) RGB -> 3ch
      -> UNet (conv1 unfrozen, 9ch) -> 384x384 binary mask
```

### Expected Impact

+2-5pp Pixel F1 (input representation experiment -- historically highest impact category)

### Risk

9-channel conv1 initialization may destabilize training. Mitigated by scaling pretrained weights by 1/3.
