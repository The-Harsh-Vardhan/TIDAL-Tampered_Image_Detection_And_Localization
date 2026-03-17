# Robustness Testing

Evaluate model robustness against post-processing degradations common in real-world image sharing pipelines.

---

## Purpose

Image tampering detection models degrade under post-processing operations that destroy forensic traces. Controlled degradation testing measures how much performance drops under realistic conditions.

Key real-world scenarios:
- **Social media re-encoding** destroys subtle splicing boundaries (typical JPEG QF 70–85)
- **Sensor noise** obscures statistical inconsistencies between tampered and authentic regions
- **Blur** destroys high-frequency forensic cues the model may rely on
- **Resize** introduces interpolation artifacts that compete with tampering artifacts

---

## Protocol

1. Apply degradations to **test images only**. Ground-truth masks remain clean and unchanged.
2. Use the trained model with **no retraining** or fine-tuning.
3. Reuse the **validation-selected threshold** — no per-degradation threshold tuning.
4. Report Pixel-F1 (mean ± std) for each condition alongside the clean baseline.
5. AMP is applied consistently via `autocast('cuda', enabled=config['use_amp'])`.

---

## Degradation Suite

| Condition | Transform | Rationale |
|---|---|---|
| Clean (baseline) | Resize(384) + Normalize | Reference performance |
| JPEG QF=70 | `ImageCompression(quality_lower=70, quality_upper=70)` | Social media re-encoding |
| JPEG QF=50 | `ImageCompression(quality_lower=50, quality_upper=50)` | Heavy compression |
| Gaussian noise (light) | `GaussNoise(var_limit=(10.0, 50.0))` | Sensor noise |
| Gaussian noise (heavy) | `GaussNoise(var_limit=(100.0, 100.0))` | Severe noise |
| Gaussian blur | `GaussianBlur(blur_limit=(5, 5))` | Post-processing blur |
| Resize 0.75× | Custom resize-restore function | Resolution loss |
| Resize 0.50× | Custom resize-restore function | Severe resolution loss |

---

## Implementation

### Standard Degradation Transforms (Albumentations)

```python
NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

robustness_transforms = {
    'clean': A.Compose([
        A.Resize(384, 384), NORMALIZE, ToTensorV2(),
    ]),
    'jpeg_qf70': A.Compose([
        A.Resize(384, 384),
        A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'jpeg_qf50': A.Compose([
        A.Resize(384, 384),
        A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'noise_light': A.Compose([
        A.Resize(384, 384),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    # ... similar for other conditions
}
```

### Resize Degradation (Custom Wrapper)

```python
class ResizeDegradationDataset(Dataset):
    """Applies resize degradation to images only. Masks stay clean."""

    def __getitem__(self, idx):
        # Load image, downscale by scale_factor, upscale back to original size
        # Load mask with binarization > 0 (consistent with training)
        # Apply normalize + ToTensorV2
        return image_t, mask_t, label
```

Mask binarization in `ResizeDegradationDataset` uses `> 0`, consistent with the training `TamperingDataset`.

---

## Results Reporting

| Degradation | Pixel-F1 (mean ± std) | Δ from clean |
|---|---|---|
| Clean (baseline) | measured | — |
| JPEG QF=70 | measured | measured |
| JPEG QF=50 | measured | measured |
| Gaussian noise (light) | measured | measured |
| Gaussian noise (heavy) | measured | measured |
| Gaussian blur (5×5) | measured | measured |
| Resize 0.75× | measured | measured |
| Resize 0.50× | measured | measured |

A bar chart comparing all conditions is generated and saved as `robustness_chart.png`.

---

## Expected Behavior

Based on research literature (EMT-Net, FENet, ME-Net):

- **JPEG QF=70**: Moderate degradation expected. Social media re-encoding is the most common real-world threat.
- **JPEG QF=50**: Significant degradation. Heavy compression destroys most forensic traces.
- **Gaussian noise**: Performance depends on whether the model relies on noise-domain features. A model using only RGB features should be moderately robust to light noise.
- **Gaussian blur**: Destroys high-frequency boundary artifacts. Models that rely on edge sharpness at manipulation boundaries will suffer most.
- **Resize**: Interpolation artifacts from downscale–upscale cycle partially mask tampering artifacts. More severe at 0.50× than 0.75×.

---

## Albumentations Compatibility

Pin `albumentations>=1.3.1,<2.0` to ensure parameters like `quality_lower`, `var_limit`, and `blur_limit` work as documented. Newer versions may rename these parameters.

---

## Interview: "What if the model completely fails under JPEG compression?"

A large drop under JPEG QF=70 would suggest the model relies heavily on compression-inconsistency artifacts (JPEG ghost detection). This would be expected for a model trained on uncompressed CASIA images — the tampering signal partially overlaps with JPEG artifacts. Mitigations include:
1. Training with JPEG augmentation (adding `ImageCompression` to training transforms)
2. Using frequency-domain features (ELA, SRM) that are more robust to re-compression
3. Multi-domain fusion (as in EMT-Net) to avoid over-reliance on any single forensic trace

These are documented as future work rather than current limitations of the baseline.
