# vR.P.21 — Implementation Plan

## Core Implementation: ELA Residual Learning (Laplacian High-Pass)

### Hypothesis

Standard ELA captures compression artifacts as pixel-wise differences. However, these differences include both forensically meaningful high-frequency residuals and low-frequency intensity variations (e.g., brightness shifts due to JPEG quantization). By applying a Laplacian high-pass filter to the ELA image, we isolate edge-like and texture-like residuals that are more diagnostic of tampering boundaries. This "residual of residuals" approach suppresses smooth, low-frequency ELA components and amplifies the high-frequency discontinuities at forgery boundaries.

### `compute_ela_laplacian_residual(image_path, quality=90, target_size=384)`

1. Open image with PIL, compute standard ELA (Q=90)
2. Convert ELA to numpy float32 array
3. Convert to grayscale (or process each channel independently)
4. Apply Laplacian filter: `cv2.Laplacian(ela_channel, cv2.CV_64F, ksize=3)`
5. Take absolute value of Laplacian response
6. Normalize each channel to [0, 255]
7. Stack into 3-channel output: (H, W, 3)
8. Resize to target_size
9. Return 3-channel numpy array

### `compute_ela_laplacian_statistics(paths, n_samples=500)`

1. Sample 500 images from training set
2. Compute Laplacian ELA residual for each
3. Return per-channel mean and std for normalization

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.21 — ELA Residual Learning (Laplacian High-Pass)" |
| 1 | Changelog | Add P.21 entry: Laplacian high-pass on ELA |
| 2 | Setup | VERSION='vR.P.21', CHANGE='Laplacian ELA residual' |
| 7 | Data prep header | Explain Laplacian residual strategy and frequency-domain rationale |
| 8 | Dataset class | Replace `compute_ela_image()` with `compute_ela_laplacian_residual()` |
| 9 | Splitting / stats | Call `compute_ela_laplacian_statistics()` for 3-channel normalization |
| 25 | Results table | "ELA-Laplacian 3ch 384sq" in input column |
| 26 | Discussion | Whether high-pass filtering improves boundary localization |
| 27 | Save model | Config includes input_type='ELA_LAPLACIAN' |

### Unchanged Cells

Cells 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 remain identical to P.3. No architecture change since input remains 3-channel.

### Key New Code

```python
import cv2

def compute_ela_laplacian_residual(image_path, quality=90, target_size=384):
    original = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert('RGB')
    ela = np.array(ImageChops.difference(original, compressed), dtype=np.float32)

    channels = []
    for c in range(3):
        lap = cv2.Laplacian(ela[:, :, c], cv2.CV_64F, ksize=3)
        lap = np.abs(lap)
        lap_max = lap.max()
        if lap_max > 0:
            lap = (lap / lap_max * 255).astype(np.uint8)
        else:
            lap = np.zeros_like(lap, dtype=np.uint8)
        channels.append(lap)

    result = np.stack(channels, axis=2)  # (H, W, 3)
    result = np.array(Image.fromarray(result).resize(
        (target_size, target_size), Image.BILINEAR))
    return result
```

### Risks

- Laplacian is a second-order derivative: it amplifies noise, which may overwhelm the forensic signal
- Authentic regions with complex textures (e.g., foliage, hair) produce strong Laplacian responses, potentially increasing false positives
- Per-image normalization (dividing by max) may cause inconsistent scaling across the dataset
- The approach discards absolute ELA magnitude information, which is itself a strong forgery indicator

### Verification Checklist

- [ ] `compute_ela_laplacian_residual()` returns shape (384, 384, 3) for valid images
- [ ] Laplacian response is non-negative after absolute value
- [ ] Output does not contain NaN/Inf values
- [ ] Statistics computation returns mean/std of shape (3,) each
- [ ] DataLoader yields batches of shape (B, 3, 384, 384)
- [ ] Training loss decreases over first 3 epochs
- [ ] Visual inspection: Laplacian residual highlights forgery boundaries more sharply than raw ELA
- [ ] No NaN/Inf in loss or predictions
