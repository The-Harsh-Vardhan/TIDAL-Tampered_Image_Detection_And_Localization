# vR.P.17 — ELA + DCT Fusion

## Experiment Description

**Version:** vR.P.17
**Track:** Pretrained Localization (Track 2)
**Parent:** vR.P.3 (ELA input, frozen body + BN unfrozen)
**Single Variable Changed:** Input representation — replace 3-channel ELA with 6-channel ELA+DCT

### Hypothesis

ELA and DCT capture complementary tampering signals. ELA highlights pixel-level recompression artifacts (spatial domain), while DCT captures block-level compression statistics (frequency domain). Fusing both should improve localization by providing the model with two independent views of the same tampering evidence.

### Pipeline

```
Raw Image
 ├── ELA (Q=90) → 3-channel ELA map (384x384)
 └── DCT → 3-channel spatial DCT map (48x48 → upsampled to 384x384)
         |
         v
    Concatenate → 6-channel input (384x384)
         |
         v
    UNet (ResNet-34 encoder, conv1 modified for 6ch)
    - conv1 UNFROZEN (learns 6-channel input)
    - body FROZEN + BN UNFROZEN
    - decoder TRAINABLE
         |
         v
    384x384 binary mask → Pixel F1, IoU, AUC + Image-level accuracy
```

### Architecture

UNet with ResNet-34 encoder. conv1 accepts 6 channels (pretrained 3ch weights duplicated and scaled by 0.5). Body frozen, BN unfrozen, conv1 unfrozen. ~3.18M trainable parameters.

### What Changes from P.3

| Aspect | P.3 | P.17 |
|--------|-----|------|
| Input | 3ch ELA (Q=90) | 6ch ELA (Q=90) + DCT spatial map |
| IN_CHANNELS | 3 | 6 |
| conv1 | Frozen (pretrained) | Unfrozen (modified for 6ch) |
| Normalization | ELA mean/std only | Separate ELA + DCT mean/std |
| Everything else | Same | Same |
