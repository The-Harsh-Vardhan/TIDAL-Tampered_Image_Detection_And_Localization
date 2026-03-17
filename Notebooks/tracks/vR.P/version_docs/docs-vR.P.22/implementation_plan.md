# vR.P.22 — Implementation Plan

## Core Implementation: SRM Noise Maps (3 SRM High-Pass Filter Channels)

### Hypothesis

Steganalysis Rich Model (SRM) filters are hand-crafted high-pass kernels originally designed for detecting steganography. These filters suppress image content and expose low-level noise patterns that differ between authentic and manipulated regions. Unlike ELA (which depends on JPEG re-compression), SRM filters operate directly on pixel values and can detect manipulation traces from any source — including copy-move, splicing, and inpainting — regardless of the compression history. By replacing ELA with three complementary SRM noise maps, we may capture a broader range of forgery artifacts.

### SRM Filter Kernels

Three standard SRM high-pass filters:

```python
# Filter 1: 1st-order edge residual (horizontal)
SRM_FILTER_1 = np.array([[ 0,  0,  0],
                          [ 0, -1,  1],
                          [ 0,  0,  0]], dtype=np.float32)

# Filter 2: 2nd-order edge residual
SRM_FILTER_2 = np.array([[ 0,  1,  0],
                          [ 1, -4,  1],
                          [ 0,  1,  0]], dtype=np.float32)

# Filter 3: 3rd-order square residual
SRM_FILTER_3 = np.array([[-1,  2, -1],
                          [ 2, -4,  2],
                          [-1,  2, -1]], dtype=np.float32)
```

### `compute_srm_noise_maps(image_path, target_size=384)`

1. Open image with cv2, convert BGR to grayscale (float32)
2. Apply each SRM filter via `cv2.filter2D(gray, cv2.CV_64F, kernel)`
3. Take absolute value of each filter response
4. Truncate to 3 sigma (clip outliers) and normalize to [0, 255]
5. Stack 3 filter responses: (H, W, 3)
6. Resize to target_size
7. Return 3-channel numpy array

### `compute_srm_statistics(paths, n_samples=500)`

1. Sample 500 images from training set
2. Compute SRM noise maps for each
3. Return per-channel mean and std for normalization

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.22 — SRM Noise Maps (3 SRM High-Pass Channels)" |
| 1 | Changelog | Add P.22 entry: SRM noise map input |
| 2 | Setup | VERSION='vR.P.22', INPUT_TYPE='SRM' |
| 7 | Data prep header | Explain SRM filters, steganalysis background, and rationale |
| 8 | Dataset class | Replace `compute_ela_image()` with `compute_srm_noise_maps()` |
| 9 | Splitting / stats | Call `compute_srm_statistics()` for 3-channel normalization |
| 10 | Visualization | Show 3 SRM channels separately + combined visualization |
| 11 | Architecture header | Note "SRM noise map input (3ch)" |
| 25 | Results table | "SRM 3ch 384sq" in input column |
| 26 | Discussion | SRM vs ELA comparison, complementarity analysis |
| 27 | Save model | Config includes input_type='SRM' |

### Unchanged Cells

Cells 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 remain identical to P.3. No architecture change since input remains 3-channel.

### Key New Code

```python
SRM_FILTER_1 = np.array([[ 0,  0,  0],
                          [ 0, -1,  1],
                          [ 0,  0,  0]], dtype=np.float32)
SRM_FILTER_2 = np.array([[ 0,  1,  0],
                          [ 1, -4,  1],
                          [ 0,  1,  0]], dtype=np.float32)
SRM_FILTER_3 = np.array([[-1,  2, -1],
                          [ 2, -4,  2],
                          [-1,  2, -1]], dtype=np.float32)

def compute_srm_noise_maps(image_path, target_size=384):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    channels = []
    for kernel in [SRM_FILTER_1, SRM_FILTER_2, SRM_FILTER_3]:
        response = cv2.filter2D(gray, cv2.CV_64F, kernel)
        response = np.abs(response)
        # Truncate at 3 sigma for robust normalization
        mu, sigma = response.mean(), response.std()
        response = np.clip(response, 0, mu + 3 * sigma)
        r_max = response.max()
        if r_max > 0:
            response = (response / r_max * 255).astype(np.uint8)
        else:
            response = np.zeros_like(response, dtype=np.uint8)
        channels.append(response)

    result = np.stack(channels, axis=2)  # (H, W, 3)
    result = np.array(Image.fromarray(result).resize(
        (target_size, target_size), Image.BILINEAR))
    return result
```

### Risks

- SRM filters are designed for steganography detection, not forgery localization — the signal may not transfer well
- Grayscale conversion discards chrominance information that may contain forgery traces
- SRM noise maps may be too similar to Laplacian-based approaches (P.21), offering limited novelty
- High-pass filtering amplifies sensor noise and JPEG artifacts uniformly, potentially flooding the signal
- 3-sigma truncation may clip genuine forgery signal in extreme cases

### Verification Checklist

- [ ] `compute_srm_noise_maps()` returns shape (384, 384, 3) for valid images
- [ ] Each SRM channel is non-negative after absolute value
- [ ] No NaN/Inf in output
- [ ] 3-sigma truncation does not clip more than ~1% of pixels
- [ ] Statistics computation returns mean/std of shape (3,) each
- [ ] DataLoader yields batches of shape (B, 3, 384, 384)
- [ ] Cell 10 visualization shows distinct patterns for each SRM filter
- [ ] Training loss decreases over first 3 epochs
- [ ] No NaN/Inf in loss or predictions
