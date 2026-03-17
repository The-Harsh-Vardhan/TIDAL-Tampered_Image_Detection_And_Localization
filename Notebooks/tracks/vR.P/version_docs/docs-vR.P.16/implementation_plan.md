# vR.P.16 — Implementation Plan

## Core Implementation: DCT Feature Extraction

### `compute_dct_feature_map(image_path, block_size=8, target_size=384)`

1. Read image with cv2, convert BGR → YCbCr, extract Y channel
2. Crop to block-aligned dimensions (h - h%8, w - w%8)
3. Reshape into (n_blocks_h, n_blocks_w, 8, 8) grid
4. Apply `cv2.dct()` to each 8x8 block
5. Compute 3 statistics per block:
   - AC energy: sum of squared coefficients excluding DC (position [0,0])
   - DC value: the [0,0] coefficient (block mean)
   - HF energy: sum of squares in bottom-right 4x4 quadrant
6. Normalize each channel to [0, 255]
7. Stack into 3-channel map, resize to target_size with bilinear interpolation

### `compute_dct_statistics(paths, n_samples=500)`

1. Sample 500 images from training set
2. Compute DCT feature map for each
3. Convert to tensor, compute per-channel mean and std
4. Return (dct_mean, dct_std) for normalization

### Dataset Class Changes

Replace `compute_ela_image()` → `compute_dct_feature_map()` in `__getitem__()`.

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | Update version, title, pipeline diagram |
| 1 | Changelog | Add P.16 entry |
| 2 | Setup | VERSION='vR.P.16', remove ELA_QUALITY, add INPUT_TYPE='DCT' |
| 7 | Data prep header | Describe DCT feature strategy |
| 8 | Dataset class | Replace ELA with DCT extraction + new dataset class |
| 9 | Splitting | Call compute_dct_statistics instead of compute_ela_statistics |
| 10 | Visualization | Show DCT channels separately + combined |
| 11 | Architecture header | Note "DCT spatial map input" |
| 22 | Prediction grid | Use denormalize_dct instead of denormalize_ela |
| 25 | Results table | "DCT 384sq" in input column |
| 26 | Discussion | DCT hypothesis, comparison framework |
| 27 | Save model | Config includes input_type='DCT' |

### Risks

- DCT map resolution (48x48 upsampled to 384x384) may be too coarse for precise localization
- Bilinear upsampling creates block-boundary artifacts that could confuse the encoder
- cv2.dct requires float32 input — ensure proper dtype handling
