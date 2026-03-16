# vR.P.17 — Implementation Plan

## Core Implementation: 6-Channel Fusion

### Dataset Class

```python
def __getitem__(self, idx):
    # ELA channels (3)
    ela = compute_ela_image(path, quality=90)
    ela_tensor = normalize_ela(to_tensor(resize(ela)))  # (3, 384, 384)

    # DCT channels (3)
    dct_map = compute_dct_feature_map(path, target_size=384)
    dct_tensor = normalize_dct(to_tensor(dct_map))  # (3, 384, 384)

    # Fuse: (6, 384, 384)
    fused = torch.cat([ela_tensor, dct_tensor], dim=0)
    return fused, mask, label
```

### Model Build (conv1 modification)

1. Create UNet with `in_channels=6`
2. Load pretrained 3-channel conv1 weights
3. Duplicate: `cat([pretrained_3ch, pretrained_3ch.clone()], dim=1) / 2.0`
4. Freeze all encoder except conv1 and BN layers

### Cell Modification Map

| Cell | Action |
|------|--------|
| 0 | Title: "vR.P.17 — ELA + DCT Spatial Fusion" |
| 1 | Changelog with P.3 → P.17 diff |
| 2 | VERSION='vR.P.17', IN_CHANNELS=6, INPUT_TYPE='ELA+DCT' |
| 7 | Explain fusion strategy and dual normalization |
| 8 | Add DCT extraction + fusion dataset class (both ELA and DCT) |
| 9 | Compute BOTH ELA stats and DCT stats from training set |
| 10 | Visualize ELA and DCT channels side-by-side (6 panels) |
| 11 | Architecture: "6ch input, conv1 unfrozen for fusion" |
| 12 | 6-channel model build with conv1 weight initialization |
| 22 | Denormalize both ELA and DCT for display |
| 25 | Results: "ELA+DCT 384sq" input label |
| 26 | Discussion: fusion hypothesis, whether DCT adds signal or noise |
| 27 | Config: input_type='ELA+DCT', channels=6 |

### Risks

- 6-channel conv1 initialization may destabilize early training (mitigated by 0.5 scaling)
- DCT channels may add noise rather than signal (P.4 showed RGB added noise to ELA)
- Doubled preprocessing time (both ELA and DCT computed per image)
