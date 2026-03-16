# Robustness Testing

Phase 3 bonus. Run only after core pipeline is complete and evaluated.

---

## Protocol

1. Apply degradations to **test images only**. Ground-truth masks remain clean and unchanged.
2. Use the trained model with **no retraining** or fine-tuning.
3. Reuse the **validation-selected threshold** — no per-degradation threshold tuning.
4. Report Pixel-F1 (mean ± std) for each degradation condition alongside the clean baseline.
5. Robustness evaluation is **segmentation-focused**. Image-level robustness behavior is not separately analyzed.

---

## Degradation Suite

| Condition | Transform | Rationale |
|---|---|---|
| Clean (baseline) | Resize + Normalize | Reference performance |
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

Requires `albumentations>=1.3.1,<2.0` for parameter compatibility.

```python
NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

robustness_transforms = {
    'clean': A.Compose([
        A.Resize(512, 512), NORMALIZE, ToTensorV2(),
    ]),
    'jpeg_qf70': A.Compose([
        A.Resize(512, 512),
        A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'jpeg_qf50': A.Compose([
        A.Resize(512, 512),
        A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'gaussian_noise_light': A.Compose([
        A.Resize(512, 512),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'gaussian_noise_heavy': A.Compose([
        A.Resize(512, 512),
        A.GaussNoise(var_limit=(100.0, 100.0), p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'gaussian_blur': A.Compose([
        A.Resize(512, 512),
        A.GaussianBlur(blur_limit=(5, 5), p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
}
```

### Resize degradation (custom wrapper)

Resize degrades image resolution by downscaling then upscaling. The mask must **not** go through this degradation — only the image changes.

```python
class ResizeDegradationDataset(Dataset):
    """Wraps test pairs with resize degradation applied to images only."""

    def __init__(self, pairs, scale_factor, image_size=512):
        self.pairs = pairs
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.normalize = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]

        # Load image
        image = cv2.imread(entry['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply resize degradation to image only
        h, w = image.shape[:2]
        small_h = max(1, int(h * self.scale_factor))
        small_w = max(1, int(w * self.scale_factor))
        degraded = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_LINEAR)

        # Load mask (clean path — no degradation)
        if entry['mask_path'] is not None:
            mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 128).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # Normalize and tensorize
        augmented = self.normalize(image=degraded, mask=mask)
        image_t = augmented['image']
        mask_t = augmented['mask'].unsqueeze(0).float()
        label = torch.tensor(entry['label'], dtype=torch.float32)
        return image_t, mask_t, label
```

### Evaluation loop (handles both albumentations and resize)

```python
def evaluate_robustness(model, test_pairs, device, threshold, transforms_dict, resize_scales):
    results = {}

    # Albumentations-based degradations
    for name, transform in transforms_dict.items():
        dataset = TamperingDataset(test_pairs, transform=transform)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
        f1_scores = run_eval_loop(model, loader, device, threshold)
        results[name] = {'f1_mean': np.mean(f1_scores), 'f1_std': np.std(f1_scores)}

    # Resize degradations
    for scale in resize_scales:
        dataset = ResizeDegradationDataset(test_pairs, scale_factor=scale)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
        f1_scores = run_eval_loop(model, loader, device, threshold)
        name = f'resize_{scale}x'
        results[name] = {'f1_mean': np.mean(f1_scores), 'f1_std': np.std(f1_scores)}

    return results
```

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

A bar chart comparing all conditions is generated in the notebook.

---

## Research Context

Image tampering detection models are expected to degrade under post-processing operations that destroy forensic traces. Controlled degradation testing provides evidence of model robustness beyond clean-image performance.

Relevant degradation categories and their expected effects:
- **JPEG compression** is the most common real-world degradation; social media platforms typically re-encode at QF 70–85, which can destroy subtle splicing boundaries.
- **Gaussian noise** simulates sensor noise and can obscure the statistical inconsistencies that distinguish tampered from authentic regions.
- **Blur** destroys high-frequency forensic cues that segmentation models may rely on.
- **Resize** causes interpolation artifacts that compete with the tampering artifacts the model was trained to detect.

These observations are consistent with general findings in the image forensics literature, though specific degradation thresholds vary by model and dataset.

---

## albumentations Compatibility

Pin `albumentations>=1.3.1,<2.0` to ensure parameters like `quality_lower`, `var_limit`, and `blur_limit` work as documented. Newer versions may rename these parameters.
