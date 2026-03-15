# vR.P.19 — Implementation Plan

## Core Implementation: Multi-Quality RGB ELA (9-Channel)

### Hypothesis

Standard ELA uses a single JPEG re-save quality (Q=90). Different quality levels emphasize different artifact patterns: Q=75 reveals coarse compression block boundaries, Q=85 captures mid-frequency artifacts, and Q=95 highlights subtle, high-frequency discrepancies. By concatenating full-color ELA at three quality levels (Q=75, Q=85, Q=95), we provide the encoder with a richer 9-channel forensic representation that captures artifact information across the compression spectrum.

### `compute_multi_quality_rgb_ela(image_path, qualities=[75, 85, 95], target_size=384)`

1. Open image with PIL
2. For each quality Q in `qualities`:
   - Re-save image as JPEG at quality Q into BytesIO buffer
   - Re-open compressed image
   - Compute pixel-wise absolute difference: `|original - compressed|`
   - Scale by 20x and clip to [0, 255] (standard ELA amplification)
3. Stack the three 3-channel ELA images along the channel axis: (9, H, W)
4. Resize to `target_size` with bilinear interpolation
5. Return 9-channel numpy array

### `compute_multi_quality_ela_statistics(paths, qualities=[75, 85, 95], n_samples=500)`

1. Sample 500 images from training set
2. Compute 9-channel multi-quality ELA for each
3. Convert to tensor, compute per-channel mean and std (9 values each)
4. Return (mean_9ch, std_9ch) for normalization

### Model Build (conv1 modification for 9 channels)

1. Create UNet with `in_channels=9`
2. Load pretrained 3-channel conv1 weights
3. Tile: `cat([pretrained_3ch, pretrained_3ch.clone(), pretrained_3ch.clone()], dim=1) / 3.0`
4. Unfreeze conv1 so it can learn quality-specific filters
5. Freeze remaining encoder layers (standard P.3 strategy)

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.19 — Multi-Quality RGB ELA (9ch, Q=75/85/95)" |
| 1 | Changelog | Add P.19 entry: 9-channel multi-quality full-color ELA |
| 2 | Setup | VERSION='vR.P.19', IN_CHANNELS=9, ELA_QUALITIES=[75, 85, 95] |
| 7 | Data prep header | Explain multi-quality ELA strategy and channel layout |
| 8 | Dataset class | Replace `compute_ela_image()` with `compute_multi_quality_rgb_ela()` |
| 9 | Splitting / stats | Call `compute_multi_quality_ela_statistics()` for 9-channel normalization |
| 11 | Architecture header | Note "9ch input (3 ELA qualities x 3 RGB channels)" |
| 12 | Model build | 9-channel conv1 initialization with 3x tiling, unfreeze conv1 |
| 25 | Results table | "Multi-Q RGB ELA 9ch 384sq" in input column |
| 26 | Discussion | Multi-quality hypothesis, per-quality contribution analysis |
| 27 | Save model | Config includes in_channels=9, ela_qualities=[75,85,95] |

### Unchanged Cells

Cells 3, 4, 5, 6, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 remain identical to P.3.

### Key New Code

```python
ELA_QUALITIES = [75, 85, 95]
IN_CHANNELS = 9

def compute_multi_quality_rgb_ela(image_path, qualities=ELA_QUALITIES, target_size=384):
    original = Image.open(image_path).convert('RGB')
    ela_channels = []
    for q in qualities:
        buffer = BytesIO()
        original.save(buffer, 'JPEG', quality=q)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        ela = ImageChops.difference(original, compressed)
        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        ela = ela.resize((target_size, target_size), Image.BILINEAR)
        ela_channels.append(np.array(ela))  # (H, W, 3)
    # Stack along channel axis: (H, W, 9)
    return np.concatenate(ela_channels, axis=2)
```

### Risks

- 3x preprocessing time per image (three JPEG re-saves instead of one)
- 9-channel conv1 may be unstable even with 1/3 scaling initialization
- The three quality levels may produce highly correlated ELA maps, adding redundancy rather than new signal
- Q=75 ELA may be noisy on authentic regions, increasing false positives

### Verification Checklist

- [ ] `compute_multi_quality_rgb_ela()` returns shape (384, 384, 9) for valid images
- [ ] Statistics computation returns mean/std of shape (9,) each
- [ ] conv1 weight shape is (64, 9, 7, 7) after initialization
- [ ] conv1 gradients are non-zero (unfrozen)
- [ ] DataLoader yields batches of shape (B, 9, 384, 384)
- [ ] Training loss decreases over first 3 epochs
- [ ] No NaN/Inf in loss or predictions
- [ ] Model checkpoint saves and loads correctly with 9-channel config
