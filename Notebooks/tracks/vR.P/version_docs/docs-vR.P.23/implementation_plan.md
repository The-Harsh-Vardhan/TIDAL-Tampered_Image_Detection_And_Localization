# vR.P.23 — Implementation Plan

## Core Implementation: Chrominance Analysis (YCbCr Channels)

### Hypothesis

Image forgeries often leave traces in the chrominance domain that are invisible in the luminance channel. When a forged region is spliced from a different source image, the chroma subsampling pattern (4:2:0 in JPEG) creates discontinuities at splice boundaries. The Cb and Cr channels capture color temperature differences between source and target images, while the Y channel provides structural context. By converting the input image to YCbCr color space and feeding all three channels directly (without ELA processing), we test whether raw chrominance information contains sufficient forensic signal for forgery localization.

### `compute_ycbcr_channels(image_path, target_size=384)`

1. Open image with PIL, convert to 'YCbCr' mode
2. Convert to numpy array: (H, W, 3) = [Y, Cb, Cr]
3. Resize to target_size with bilinear interpolation
4. Return 3-channel numpy array (values in [0, 255])

### `compute_ycbcr_statistics(paths, n_samples=500)`

1. Sample 500 images from training set
2. Convert each to YCbCr
3. Compute per-channel mean and std (3 values each)
4. Return (ycbcr_mean, ycbcr_std) for normalization

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.23 — Chrominance Analysis (YCbCr Y/Cb/Cr)" |
| 1 | Changelog | Add P.23 entry: YCbCr color space input |
| 2 | Setup | VERSION='vR.P.23', INPUT_TYPE='YCbCr' |
| 7 | Data prep header | Explain YCbCr forensic rationale, chroma subsampling artifacts |
| 8 | Dataset class | Replace `compute_ela_image()` with `compute_ycbcr_channels()` |
| 9 | Splitting / stats | Call `compute_ycbcr_statistics()` for 3-channel normalization |
| 25 | Results table | "YCbCr 3ch 384sq" in input column |
| 26 | Discussion | Chrominance vs. ELA comparison, chroma subsampling analysis |
| 27 | Save model | Config includes input_type='YCbCr' |

### Unchanged Cells

Cells 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 remain identical to P.3. No architecture change since input remains 3-channel.

### Key New Code

```python
def compute_ycbcr_channels(image_path, target_size=384):
    img = Image.open(image_path).convert('YCbCr')
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)  # (H, W, 3) = [Y, Cb, Cr]

def compute_ycbcr_statistics(paths, n_samples=500):
    indices = random.sample(range(len(paths)), min(n_samples, len(paths)))
    all_tensors = []
    for i in indices:
        ycbcr = compute_ycbcr_channels(paths[i])
        tensor = torch.from_numpy(ycbcr).permute(2, 0, 1).float() / 255.0
        all_tensors.append(tensor)
    stacked = torch.stack(all_tensors)
    return stacked.mean(dim=[0, 2, 3]), stacked.std(dim=[0, 2, 3])
```

### Risks

- YCbCr is essentially a linear transformation of RGB — the pretrained encoder may already extract similar features from RGB input, making this experiment equivalent to P.1 (RGB baseline)
- Without ELA processing, there is no explicit amplification of compression artifacts; the network must learn forensic features from raw pixel values
- Chroma subsampling artifacts are subtle (sub-pixel level) and may be destroyed by the resize to 384x384
- CASIA2 dataset images vary in original resolution and compression; chroma artifacts may be inconsistent

### Verification Checklist

- [ ] `compute_ycbcr_channels()` returns shape (384, 384, 3) for valid images
- [ ] Y channel ranges [0, 255], Cb/Cr channels range [0, 255] (PIL YCbCr convention)
- [ ] Statistics computation returns mean/std of shape (3,) each
- [ ] DataLoader yields batches of shape (B, 3, 384, 384)
- [ ] Training loss decreases over first 3 epochs
- [ ] No NaN/Inf in loss or predictions
- [ ] Visual inspection: Cb/Cr channels show color differences in spliced regions
- [ ] Model checkpoint saves and loads correctly with YCbCr config
