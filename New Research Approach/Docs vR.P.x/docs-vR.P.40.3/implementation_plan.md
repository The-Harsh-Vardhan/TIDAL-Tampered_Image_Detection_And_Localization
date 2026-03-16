# vR.P.40.3 — Implementation Plan

## Core Implementation: InceptionV1 Custom Encoder for SMP

### Custom Encoder Architecture

```python
class InceptionV1Module(nn.Module):
    # Parallel branches: 1x1, 3x3, 5x5, MaxPool
    # No BatchNorm (original 2014 design)
    # Output = f1 + f3 + f5 + fpool channels

class InceptionV1Encoder(nn.Module):
    # 5-stage encoder compatible with SMP's UNet decoder
    # out_channels = [9, 32, 128, 240, 336, 432]
    # Each stage: InceptionV1 -> 1x1 projection -> MaxPool 2x2
```

### SMP Registration

```python
smp.encoders.encoders['inception-v1-custom'] = {
    'encoder': InceptionV1Encoder,
    'pretrained_settings': {},
    'params': {'in_channels': 9, 'depth': 5}
}
```

### Stage Architecture Detail

| Stage | Input ch | Inception branches (f1/f3r/f3/f5r/f5/fpool) | Inception out | Proj out | Downsample |
|-------|----------|----------------------------------------------|---------------|----------|------------|
| Stem | 9 | Conv2d(9, 32, 3, stride=2) | 32 | 32 | stride 2 |
| 1 | 32 | 16/16/32/8/16/8 | 72 | 128 | MaxPool 2x2 |
| 2 | 128 | 48/32/80/16/32/16 | 176 | 240 | MaxPool 2x2 |
| 3 | 240 | 64/48/128/24/48/24 | 264 | 336 | MaxPool 2x2 |
| 4 | 336 | 80/64/160/32/80/32 | 352 | 432 | MaxPool 2x2 |

### Training Strategy

- All parameters trainable (no pretrained weights)
- Single Adam optimizer, LR=1e-3
- ReduceLROnPlateau (factor=0.5, patience=3)
- BCEDice loss (same as all other experiments)
- 30 epochs, patience=10

### Cell Modification Map

| Cell | Action |
|------|--------|
| 0 | Title: InceptionV1 custom encoder |
| 2 | ENCODER='inception-v1-custom', encoder_weights=None |
| 12 | Full InceptionV1 encoder class + SMP registration |
| 14 | All params trainable (no freeze) |

### Risks

- No BatchNorm makes training from scratch unstable
- 12K training images may be insufficient for learning from scratch
- MaxPool may lose fine-grained spatial information needed for pixel-level segmentation

### Verification Checklist

- [ ] InceptionV1Encoder.forward() returns list of 6 feature tensors
- [ ] out_channels matches [9, 32, 128, 240, 336, 432]
- [ ] SMP registration works: smp.Unet(encoder_name='inception-v1-custom') builds
- [ ] All parameters are trainable (no frozen layers)
- [ ] Training loss decreases over first 5 epochs
- [ ] No NaN/Inf (critical without BatchNorm)
