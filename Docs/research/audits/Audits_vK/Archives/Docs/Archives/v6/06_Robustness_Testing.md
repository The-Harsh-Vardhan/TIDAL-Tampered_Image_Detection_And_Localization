# Robustness Testing

Evaluate model robustness against post-processing degradations that are common in real-world image sharing pipelines.

---

## Purpose

Image tampering detection models are expected to degrade under post-processing operations that destroy forensic traces. Controlled degradation testing provides evidence of robustness beyond clean-image performance.

Key real-world scenarios:
- **Social media re-encoding** destroys subtle splicing boundaries (typical QF 70–85).
- **Sensor noise** obscures statistical inconsistencies between tampered and authentic regions.
- **Blur** destroys high-frequency forensic cues the segmentation model may rely on.
- **Resize** introduces interpolation artifacts that compete with tampering artifacts.

---

## Protocol

1. Apply degradations to **test images only**. Ground-truth masks remain clean and unchanged.
2. Use the trained model with **no retraining** or fine-tuning.
3. Reuse the **validation-selected threshold** — no per-degradation threshold tuning.
4. Report Pixel-F1 (mean ± std) for each degradation condition alongside the clean baseline.

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

### Standard degradation transforms (albumentations-based)

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
    # ... similar for other albumentations-based degradations
}
```

### Resize degradation (custom wrapper)

```python
class ResizeDegradationDataset(Dataset):
    '''Applies resize degradation to images only. Masks stay clean.'''

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

A bar chart comparing all conditions is generated in the notebook and saved as `robustness_chart.png`.

---

## albumentations Compatibility

Pin `albumentations>=1.3.1,<2.0` to ensure parameters like `quality_lower`, `var_limit`, and `blur_limit` work as documented. Newer versions may rename these parameters.
