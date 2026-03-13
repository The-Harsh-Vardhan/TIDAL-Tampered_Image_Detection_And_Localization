# 08 — Robustness Testing

## Purpose

This document defines the bonus robustness evaluation protocol. Robustness testing is **Stage 3** work and is only attempted after the core model is trained and evaluated.

## Motivation

Real-world tampered images are often post-processed before distribution (JPEG compression on social media, resizing for storage, noise from re-capture). A robust model should maintain localization quality under these degradations.

## Robustness Test Protocol

Apply each degradation to the **test set** independently. Run the trained model (without retraining) on the degraded images and report Pixel-F1.

### Degradation Transforms

Each transform is applied to test images only. Ground truth masks are unchanged.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

robustness_transforms = {
    'clean': A.Compose([
        A.Resize(512, 512),
        NORMALIZE,
        ToTensorV2(),
    ]),
    'jpeg_qf70': A.Compose([
        A.Resize(512, 512),
        A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),
        NORMALIZE,
        ToTensorV2(),
    ]),
    'jpeg_qf50': A.Compose([
        A.Resize(512, 512),
        A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),
        NORMALIZE,
        ToTensorV2(),
    ]),
    'gaussian_noise_light': A.Compose([
        A.Resize(512, 512),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        NORMALIZE,
        ToTensorV2(),
    ]),
    'gaussian_noise_heavy': A.Compose([
        A.Resize(512, 512),
        A.GaussNoise(var_limit=(100.0, 100.0), p=1.0),
        NORMALIZE,
        ToTensorV2(),
    ]),
    'resize_075x': A.Compose([
        A.Resize(384, 384),
        A.Resize(512, 512),
        NORMALIZE,
        ToTensorV2(),
    ]),
    'resize_050x': A.Compose([
        A.Resize(256, 256),
        A.Resize(512, 512),
        NORMALIZE,
        ToTensorV2(),
    ]),
}
```

### Evaluation Loop

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

## Expected Results Format

| Degradation | Pixel-F1 (mean ± std) | Δ from clean |
|---|---|---|
| Clean (baseline) | e.g., 0.62 ± 0.18 | — |
| JPEG QF=70 | e.g., 0.58 ± 0.19 | -0.04 |
| JPEG QF=50 | e.g., 0.50 ± 0.21 | -0.12 |
| Gaussian noise (light) | e.g., 0.56 ± 0.20 | -0.06 |
| Gaussian noise (heavy) | e.g., 0.42 ± 0.23 | -0.20 |
| Resize 0.75× | e.g., 0.55 ± 0.20 | -0.07 |
| Resize 0.50× | e.g., 0.45 ± 0.22 | -0.17 |

Moderate degradation (JPEG QF 70, light noise) should show a 5–10% F1 drop. Heavy degradation (JPEG QF 50, heavy noise, 0.5× resize) may show 15–35% drops.

## Visualization

```python
def plot_robustness_results(results):
    names = list(results.keys())
    means = [results[n]['f1_mean'] for n in names]
    stds = [results[n]['f1_std'] for n in names]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(names)), means, yerr=stds, capsize=5, alpha=0.8)
    bars[0].set_color('green')  # Highlight clean baseline
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Pixel-F1')
    plt.title('Robustness Evaluation')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('robustness_results.png', dpi=150, bbox_inches='tight')
    plt.show()
```

## Optional: Forgery Type Breakdown

If time permits, report separate results for splicing vs. copy-move images from the CASIA test set.

```python
def evaluate_by_forgery_type(model, test_pairs, device, threshold=0.5):
    splicing = [p for p in test_pairs if p['forgery_type'] == 'splicing']
    copymove = [p for p in test_pairs if p['forgery_type'] == 'copy-move']
    
    # Evaluate each subset separately using the same evaluate() function
    # Report: splicing F1 mean ± std, copy-move F1 mean ± std
```

## What Not to Include

- Do not re-train the model on augmented data for robustness testing. Use the same trained model.
- Do not report COVERAGE dataset evaluation unless all other bonus work is complete.
- Do not implement adversarial attacks or complex perturbation schemes.

## Related Documents

- [06_Evaluation_Methodology.md](06_Evaluation_Methodology.md) — Core metrics
- [10_Project_Timeline.md](10_Project_Timeline.md) — This is Stage 3 work
