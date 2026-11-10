# 08 — Robustness Testing

## Purpose

Define the bonus robustness evaluation protocol. This is **Phase 3** work — only attempt after the core model is fully trained and evaluated.

## Protocol

Apply degradations to **test images only**. Ground truth masks remain clean and unchanged. Run the trained model (no retraining) on degraded images and report Pixel-F1.

**Critical:** The robustness transforms must apply to images only. Masks must follow a clean path (resize only, nearest-neighbor interpolation). If using `albumentations`, ensure the degradation transforms are pixel-level only (not spatial) so the mask is not affected.

**Threshold reuse:** Use the same threshold selected on the clean validation set. Do not retune per degradation.

## Degradation Transforms

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
}
```

For resize degradation, apply the downscale-upscale to images separately, then pair with the cleanly resized mask:

```python
def apply_resize_degradation(image, scale_factor):
    """Degrade image by downscaling then upscaling. Apply to image only."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    restored = cv2.resize(small, (w, h))
    return restored
```

This avoids accidentally degrading masks through `albumentations` spatial transforms.

## Evaluation

```python
def evaluate_robustness(model, test_pairs, device, threshold=0.5):
    results = {}
    for name, transform in robustness_transforms.items():
        dataset = TamperingDataset(test_pairs, transform=transform)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

        f1_scores = []
        model.eval()
        with torch.no_grad():
            for images, masks, labels in loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > threshold).float()
                for pred, gt in zip(preds, masks):
                    f1_scores.append(compute_pixel_f1(pred, gt))

        results[name] = {
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
        }
    return results
```

## Results Table Format

| Degradation | Pixel-F1 (mean +/- std) | Delta from clean |
|---|---|---|
| Clean (baseline) | measured | — |
| JPEG QF=70 | measured | measured |
| JPEG QF=50 | measured | measured |
| Gaussian noise (light) | measured | measured |
| Gaussian noise (heavy) | measured | measured |
| Resize 0.75x | measured | measured |
| Resize 0.50x | measured | measured |

Do not fill in expected values. Report only what the notebook measures.

## Optional: Forgery Type Breakdown

Report separate Pixel-F1 for splicing vs. copy-move images from the test set:

```python
splicing = [p for p in test_pairs if p['forgery_type'] == 'splicing']
copymove = [p for p in test_pairs if p['forgery_type'] == 'copy-move']
```

## Related Documents

- [06_Evaluation_Methodology.md](06_Evaluation_Methodology.md) — Core metrics
- [10_Project_Timeline.md](10_Project_Timeline.md) — This is Phase 3 work
