# vR.P.27 — Implementation Plan

## Core Implementation: JPEG Compression Augmentation (Training-Time)

### JPEG Compression Augmentation Function

```python
def jpeg_compression_augmentation(image_pil, min_quality=50, max_quality=95, p=0.5):
    """Apply random JPEG compression as data augmentation before ELA computation."""
    if random.random() > p:
        return image_pil  # no augmentation
    quality = random.randint(min_quality, max_quality)
    buffer = io.BytesIO()
    image_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()
```

### Modified Dataset `__getitem__`

```python
def __getitem__(self, idx):
    image = Image.open(self.paths[idx]).convert('RGB')
    # Apply JPEG augmentation BEFORE ELA computation (training only)
    if self.split == 'train':
        image = jpeg_compression_augmentation(image, JPEG_AUG_MIN_Q, JPEG_AUG_MAX_Q, JPEG_AUG_P)
    ela = compute_ela_image(image, quality=ELA_QUALITY)
    ...
```

The key insight: JPEG compression augmentation happens **before** ELA computation, so the model sees ELA patterns from images at various compression levels, making it robust to real-world recompression.

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.27 — JPEG Compression Augmentation (Training-Time)" |
| 1 | Changelog | Add P.27 entry: JPEG aug with Q=[50,95], p=0.5 |
| 2 | Setup | VERSION='vR.P.27', JPEG_AUG_MIN_Q=50, JPEG_AUG_MAX_Q=95, JPEG_AUG_P=0.5 |
| 7 | Data prep header | Explain JPEG augmentation rationale: real-world images are recompressed when shared, augmentation simulates this |
| 8 | Dataset class | Add `jpeg_compression_augmentation()` function; modify `__getitem__` to apply augmentation before ELA for training split only |
| 25 | Results table | Note "JPEG Aug Q=[50,95] p=0.5" in config column |
| 26 | Discussion | Robustness hypothesis, comparison to P.18 robustness test results |
| 27 | Save model | Config includes jpeg_aug_min_q, jpeg_aug_max_q, jpeg_aug_p |

### Unchanged Cells

Cells 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 remain unchanged from P.3. This is a data-pipeline-only modification — no changes to model architecture, loss function, training loop, or evaluation.

### Key New Code

- `jpeg_compression_augmentation()` (~10 lines): PIL-based random JPEG recompression
- Modified `__getitem__` (~3 lines changed): conditional augmentation call before ELA
- Total new code is minimal (~13 lines), making this a clean augmentation-only ablation

### Verification Checklist

- [ ] JPEG augmentation only applied during training (not val/test)
- [ ] Augmented images are valid PIL images (no corruption from BytesIO round-trip)
- [ ] ELA computation on augmented images produces valid outputs (no all-zero or all-saturated maps)
- [ ] Augmentation probability ~50% verified by logging (half of training images should be augmented per epoch)
- [ ] Training convergence is comparable to P.3 (augmentation should not slow convergence dramatically)
- [ ] Visualization cell shows ELA differences between augmented and non-augmented versions
- [ ] All metric cells execute and produce valid Pixel F1 / IoU / AUC values
- [ ] Memory usage unchanged (BytesIO buffer is ephemeral)

### Risks

- JPEG augmentation at Q=50 may destroy too much forensic signal, making training labels noisy
- ELA computed on already-compressed augmented images may look different from standard ELA, confusing the model
- Training may take slightly longer due to PIL encode/decode overhead per image
