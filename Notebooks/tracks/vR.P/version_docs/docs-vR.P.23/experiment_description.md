# vR.P.23 -- Experiment Description

## Chrominance Channel Analysis (YCbCr Cb/Cr)

### Hypothesis

Chrominance channels (Cb, Cr) in the YCbCr color space reveal tampering artifacts that are invisible in luminance (Y) or RGB representations. JPEG compression applies stronger quantization to chrominance than luminance, making chrominance discontinuities at splice boundaries more pronounced. Combining Y + Cb + Cr as 3-channel input exploits this asymmetry.

### Motivation

JPEG compression uses 4:2:0 chroma subsampling and coarser quantization for Cb/Cr channels. When a tampered region is spliced from a differently-compressed source, the chrominance quantization grid misaligns at the boundary. This creates visible "grid artifacts" in Cb/Cr that don't appear in the luminance channel.

Previous work: YCbCr-based forensic analysis is well-established in the literature. Our contribution is testing it within the pretrained UNet pipeline to assess complementarity with ELA.

### Single Variable Changed from vR.P.3

**Input representation** -- Replace ELA with YCbCr (Y, Cb, Cr) channels. Architecture unchanged.

### Key Configuration

| Parameter | P.3 (parent) | P.23 (this) |
|-----------|-------------|-------------|
| Input | ELA (Q=90) RGB | YCbCr (Y, Cb, Cr) |
| Preprocessing | JPEG resave + diff + scale | RGB -> YCbCr conversion, normalize per-channel |
| IN_CHANNELS | 3 | 3 |
| Everything else | Same | Same |

### Pipeline

```
Image -> cv2.cvtColor(RGB, COLOR_RGB2YCrCb)
    -> Y channel (luminance)
    -> Cb channel (blue chrominance)
    -> Cr channel (red chrominance)
    -> Stack (Y, Cb, Cr) -> normalize -> UNet -> mask
```

### Expected Impact

+0-3pp Pixel F1 as standalone. More valuable as a complementary signal for future fusion experiments.

### Risk

Chrominance signal is weaker than ELA for CASIA v2.0 (many tamperings are copy-move within same image, so chrominance characteristics may match). Best suited for cross-source splicing detection.
