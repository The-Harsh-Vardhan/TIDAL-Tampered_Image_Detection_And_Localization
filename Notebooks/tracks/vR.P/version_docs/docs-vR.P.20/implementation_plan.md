# vR.P.20 — Implementation Plan

## Core Implementation: ELA Magnitude Channel Decomposition

### Hypothesis

Standard RGB ELA distributes forensic signal across three correlated color channels. By decomposing the ELA image into a magnitude channel (overall error intensity) and two chrominance direction channels (ChromaDir1, ChromaDir2), we separate "how much error" from "what kind of error." The magnitude channel directly measures tampering intensity regardless of color, while chrominance directions capture color-specific compression artifacts that differ between authentic and forged regions.

### `compute_ela_magnitude_decomposition(image_path, quality=90, target_size=384)`

1. Open image with PIL, compute standard ELA (Q=90)
2. Convert ELA to numpy float32 array: (H, W, 3) — channels are R, G, B error
3. Compute magnitude channel: `Mag = sqrt(R^2 + G^2 + B^2)` — scalar error intensity
4. Normalize magnitude to [0, 255]
5. Compute chrominance directions:
   - `ChromaDir1 = (R - G) / (Mag + eps)` — red-green error ratio
   - `ChromaDir2 = (B - (R + G) / 2) / (Mag + eps)` — blue vs. red-green error ratio
6. Normalize ChromaDir1 and ChromaDir2 to [0, 255]
7. Stack: (H, W, 3) = [Mag, ChromaDir1, ChromaDir2]
8. Resize to target_size
9. Return 3-channel numpy array

### `compute_ela_mag_statistics(paths, n_samples=500)`

1. Sample 500 images from training set
2. Compute magnitude decomposition for each
3. Return per-channel mean and std (3 values each)

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.20 — ELA Magnitude Channel (Mag/ChromaDir1/ChromaDir2)" |
| 1 | Changelog | Add P.20 entry: magnitude decomposition of ELA |
| 2 | Setup | VERSION='vR.P.20', CHANGE='ELA magnitude decomposition' |
| 7 | Data prep header | Explain magnitude vs chrominance decomposition rationale |
| 8 | Dataset class | Replace `compute_ela_image()` with `compute_ela_magnitude_decomposition()` |
| 9 | Splitting / stats | Call `compute_ela_mag_statistics()` for 3-channel normalization |
| 25 | Results table | "ELA-Mag 3ch 384sq" in input column |
| 26 | Discussion | Whether decomposition improves over raw RGB ELA |
| 27 | Save model | Config includes input_type='ELA_MAG' |

### Unchanged Cells

Cells 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 remain identical to P.3. No architecture change since input remains 3-channel.

### Key New Code

```python
def compute_ela_magnitude_decomposition(image_path, quality=90, target_size=384):
    original = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert('RGB')
    ela = np.array(ImageChops.difference(original, compressed), dtype=np.float32)

    R, G, B = ela[:,:,0], ela[:,:,1], ela[:,:,2]
    mag = np.sqrt(R**2 + G**2 + B**2)
    eps = 1e-6
    chroma_dir1 = (R - G) / (mag + eps)
    chroma_dir2 = (B - (R + G) / 2.0) / (mag + eps)

    # Normalize each to [0, 255]
    mag = np.clip(mag / (mag.max() + eps) * 255, 0, 255).astype(np.uint8)
    chroma_dir1 = np.clip((chroma_dir1 + 1) * 127.5, 0, 255).astype(np.uint8)
    chroma_dir2 = np.clip((chroma_dir2 + 1) * 127.5, 0, 255).astype(np.uint8)

    result = np.stack([mag, chroma_dir1, chroma_dir2], axis=2)
    result = np.array(Image.fromarray(result).resize(
        (target_size, target_size), Image.BILINEAR))
    return result
```

### Risks

- Chrominance direction channels may be extremely noisy in low-magnitude regions (where ELA error is near zero, direction is undefined)
- The eps-stabilized division may produce plateau values that confuse the encoder
- Magnitude alone may carry most of the signal, making ChromaDir channels wasted capacity
- Normalization to [0, 255] may clip meaningful variation in chrominance directions

### Verification Checklist

- [ ] `compute_ela_magnitude_decomposition()` returns shape (384, 384, 3) for valid images
- [ ] Magnitude channel is non-negative and correlates with tampering
- [ ] ChromaDir channels are bounded and don't contain NaN/Inf
- [ ] Statistics computation returns mean/std of shape (3,) each
- [ ] DataLoader yields batches of shape (B, 3, 384, 384)
- [ ] Training loss decreases over first 3 epochs
- [ ] No NaN/Inf in loss or predictions
- [ ] Visual inspection: magnitude channel highlights forged regions
