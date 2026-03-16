# 10. Bonus Points — Robustness & Extended Evaluation Guide

## 10.1 Assignment Requirement

> *"Points for robustness testing."*
> *"Testing on the COVERAGE dataset is a plus."*
> *"Copy-move detection evaluation is a plus."*

These are the differentiators that separate a good submission from a great one.

---

## 10.2 Bonus Opportunity Map

| Bonus | Effort | Impact | Recommended? |
|-------|--------|--------|-------------|
| **Robustness Testing** (JPEG, noise, resize) | Low (~30 min) | High | **Yes — essential** |
| **COVERAGE Dataset Evaluation** | Medium (~1 hr) | High | **Yes — strong signal** |
| **Copy-Move Breakdown** | Low (~15 min) | Medium | **Yes — already have data** |
| **SRM Ablation** (RGB-only vs. RGB+SRM) | Medium (~2 hr) | Very High | **If time permits** |

---

## 10.3 Bonus 1: Robustness Testing

### Concept
Real-world tampered images are often post-processed — shared on social media (JPEG compression), saved multiple times, slightly cropped or resized. A good forensic model should maintain performance under these degradations.

### Attack Definitions

| Attack | Severity | Parameters | Real-World Scenario |
|--------|----------|-----------|-------------------|
| **JPEG Compression** | Moderate | QF=70 | Social media sharing (Instagram, Twitter) |
| **JPEG Compression** | Severe | QF=50 | Aggressive web compression |
| **Gaussian Noise** | Moderate | σ=5 (0.02 normalized) | Sensor noise, low-light capture |
| **Gaussian Noise** | Severe | σ=15 (0.06 normalized) | Heavy noise / deliberate anti-forensics |
| **Resize** | Moderate | 0.75× then back to 512 | Moderate downsampling |
| **Resize** | Severe | 0.5× then back to 512 | Heavy downsampling (thumbnail recovery) |

### Implementation

```python
import albumentations as A
import cv2
import io
from PIL import Image as PILImage

def apply_jpeg_compression(image_np, quality=70):
    """
    Apply JPEG compression to a numpy image and decode back.
    
    Args:
        image_np: (H, W, 3) uint8 numpy array
        quality: JPEG quality factor (1-100)
    Returns:
        compressed: (H, W, 3) uint8 numpy array
    """
    # Encode to JPEG bytes
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), encode_param)
    # Decode back
    compressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)


def apply_gaussian_noise(image_np, sigma=5):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, image_np.shape).astype(np.float32)
    noisy = np.clip(image_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_resize_attack(image_np, scale=0.5):
    """Downscale then upscale back to original resolution."""
    h, w = image_np.shape[:2]
    small = cv2.resize(image_np, (int(w * scale), int(h * scale)), 
                       interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored
```

### Evaluation Framework

```python
def evaluate_robustness(model, test_dataset, device, threshold=0.5):
    """
    Run model on clean and degraded versions of test set.
    Report F1 drop for each attack.
    """
    attacks = {
        'Clean (baseline)': lambda img: img,
        'JPEG QF=70': lambda img: apply_jpeg_compression(img, quality=70),
        'JPEG QF=50': lambda img: apply_jpeg_compression(img, quality=50),
        'Gaussian σ=5': lambda img: apply_gaussian_noise(img, sigma=5),
        'Gaussian σ=15': lambda img: apply_gaussian_noise(img, sigma=15),
        'Resize 0.75×': lambda img: apply_resize_attack(img, scale=0.75),
        'Resize 0.5×': lambda img: apply_resize_attack(img, scale=0.5),
    }
    
    results = {}
    
    for attack_name, attack_fn in attacks.items():
        f1_scores = []
        
        model.eval()
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                # Get raw image (before normalization) and mask
                image_np, mask = test_dataset.get_raw(idx)  # (H,W,3) uint8, (H,W) float
                
                # Apply attack to the raw image
                attacked = attack_fn(image_np)
                
                # Apply standard preprocessing (normalize + to tensor)
                transformed = test_dataset.transform(image=attacked, mask=mask.numpy())
                image_tensor = transformed['image'].unsqueeze(0).to(device)
                mask_tensor = transformed['mask']
                
                # Predict
                with autocast('cuda'):
                    logits = model(image_tensor)
                pred = torch.sigmoid(logits).squeeze().cpu()
                
                f1, _, _ = compute_pixel_f1_safe(pred, mask_tensor, threshold)
                f1_scores.append(f1)
        
        mean_f1 = np.mean(f1_scores)
        results[attack_name] = mean_f1
    
    return results


def print_robustness_table(results):
    """Pretty-print robustness results with F1 drop."""
    baseline = results.get('Clean (baseline)', 0)
    
    print(f"\n{'='*60}")
    print(f"{'ROBUSTNESS ANALYSIS':^60}")
    print(f"{'='*60}")
    print(f"{'Attack':<25} {'F1':>8} {'Drop':>8} {'Relative':>10}")
    print(f"{'-'*60}")
    
    for attack, f1 in results.items():
        drop = baseline - f1
        rel_drop = (drop / baseline * 100) if baseline > 0 else 0
        marker = '' if attack == 'Clean (baseline)' else f'{rel_drop:>+.1f}%'
        print(f"  {attack:<23} {f1:>8.4f} {drop:>+8.4f} {marker:>10}")
    
    print(f"{'='*60}")
```

### Expected Results Pattern

| Attack | Expected F1 | Expected Drop | Concern Level |
|--------|------------|---------------|---------------|
| Clean | 0.60–0.70 | — | Baseline |
| JPEG QF=70 | 0.55–0.65 | -5–8% | Acceptable |
| JPEG QF=50 | 0.45–0.55 | -15–25% | Expected |
| Gaussian σ=5 | 0.55–0.65 | -5–10% | Acceptable |
| Gaussian σ=15 | 0.40–0.50 | -20–35% | Expected |
| Resize 0.75× | 0.50–0.60 | -10–15% | Acceptable |
| Resize 0.5× | 0.35–0.50 | -25–40% | Expected |

> Moderate degradation (<15% drop) on JPEG QF=70 and Gaussian σ=5 indicates a robust model. Severe degradation under heavy attacks is normal.

---

## 10.4 Bonus 2: COVERAGE Dataset Evaluation

### Why COVERAGE
COVERAGE is a specialized copy-move forgery dataset (100 original + 100 forged images) with precise ground truth. Evaluating on it demonstrates **cross-dataset generalization**.

### Download and Setup

```python
# COVERAGE dataset — download from the original source
# Small dataset: only 200 images, can be uploaded manually to Colab
# or downloaded from a direct link if available

COVERAGE_DIR = '/content/coverage'
# structure expected:
# coverage/
# ├── image/        (forged images)
# ├── mask/         (ground truth masks)
# └── original/     (original images)
```

### Evaluation

```python
def evaluate_coverage(model, coverage_dir, transform, device, threshold=0.5):
    """
    Evaluate model on COVERAGE dataset (copy-move only).
    
    Note: Model was NOT trained on COVERAGE — this tests generalization.
    """
    image_dir = os.path.join(coverage_dir, 'image')
    mask_dir = os.path.join(coverage_dir, 'mask')
    
    f1_scores = []
    iou_scores = []
    
    model.eval()
    with torch.no_grad():
        for img_file in sorted(os.listdir(image_dir)):
            # Load image
            img_path = os.path.join(image_dir, img_file)
            image = np.array(PILImage.open(img_path).convert('RGB'))
            
            # Find corresponding mask
            mask_name = img_file.replace('.png', '_mask.png')  # Adjust naming
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                continue
            mask = np.array(PILImage.open(mask_path).convert('L'))
            mask = (mask > 128).astype(np.float32)
            
            # Preprocess
            transformed = transform(image=image, mask=mask)
            img_tensor = transformed['image'].unsqueeze(0).to(device)
            mask_tensor = transformed['mask']
            
            # Predict
            with autocast('cuda'):
                logits = model(img_tensor)
            pred = torch.sigmoid(logits).squeeze().cpu()
            
            f1, _, _ = compute_pixel_f1_safe(pred, mask_tensor, threshold)
            iou = compute_pixel_iou(pred, mask_tensor, threshold)
            
            f1_scores.append(f1)
            iou_scores.append(iou)
    
    print(f"\nCOVERAGE Dataset Results ({len(f1_scores)} images):")
    print(f"  Pixel-F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"  Pixel-IoU: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    
    return f1_scores, iou_scores
```

### Expected Results
Since the model is trained on CASIA (mostly splicing) and COVERAGE is entirely copy-move:
- **Expected F1 on COVERAGE**: 0.30–0.50 (cross-domain generalization is hard)
- **Anything above 0.40** is impressive for a cross-dataset evaluation
- **The point isn't high numbers** — it's showing you thought to test generalization

---

## 10.5 Bonus 3: Copy-Move vs. Splicing Breakdown

Already available from the CASIA test set — just split results by filename pattern.

```python
def analyze_by_tampering_type(test_dataset, pred_masks, gt_masks, threshold=0.5):
    """Print separate results for splicing and copy-move."""
    splicing_f1, copymove_f1 = [], []
    
    for idx, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
        filename = test_dataset.filenames[idx]
        f1, _, _ = compute_pixel_f1_safe(pred, gt, threshold)
        
        if gt.sum() == 0:
            continue  # Skip authentic images
        
        if '_S_' in filename:
            splicing_f1.append(f1)
        elif '_C_' in filename:
            copymove_f1.append(f1)
    
    print(f"\n{'Tampering Type Analysis':=^50}")
    print(f"  Splicing  ({len(splicing_f1):>3} images): "
          f"F1 = {np.mean(splicing_f1):.4f} ± {np.std(splicing_f1):.4f}")
    print(f"  Copy-Move ({len(copymove_f1):>3} images): "
          f"F1 = {np.mean(copymove_f1):.4f} ± {np.std(copymove_f1):.4f}")
    
    # Discussion point for notebook markdown
    if np.mean(splicing_f1) > np.mean(copymove_f1):
        print(f"\n  → Splicing outperforms copy-move by "
              f"{np.mean(splicing_f1) - np.mean(copymove_f1):.4f}")
        print(f"  → This is expected: splicing introduces cross-camera artifacts "
              f"(noise mismatch, JPEG grid mismatch) that SRM detects well.")
        print(f"  → Copy-move uses source material from the same image, "
              f"so noise statistics are naturally matched — harder to detect.")
```

---

## 10.6 Bonus 4: SRM Ablation Study (If Time Permits)

This is the **highest-impact bonus** but requires retraining:

```python
# Train a second model WITHOUT SRM (RGB-only, 3 channels)
model_rgb_only = smp.Unet(
    encoder_name='efficientnet-b1',
    encoder_weights='imagenet',
    in_channels=3,         # RGB only — no SRM
    classes=1,
    activation=None
).to(device)

# Train with identical hyperparameters, same splits, same seed
# Compare: SRM model F1 vs. RGB-only F1
```

### How to Present
```
| Model Variant          | Pixel-F1 | Pixel-IoU | Image-AUC |
|------------------------|----------|-----------|-----------|
| RGB Only (baseline)    | 0.XX     | 0.XX      | 0.XX      |
| RGB + SRM (ours)       | 0.XX     | 0.XX      | 0.XX      |
| Improvement            | +X.XX    | +X.XX     | +X.XX     |
```

> **Expected result**: SRM should improve F1 by 5-15%. If it doesn't, check that the SRM filters are producing meaningful non-zero output (call `print_gpu_memory()` and inspect SRM output visually).

---

## 10.7 Notebook Integration

### Where These Go in the Notebook

After Section 8 (Visual Results), add:

```markdown
# Cell 30 (Markdown)
## 9. Robustness & Extended Analysis

### 9.1 Robustness Under Post-Processing Attacks
We test model performance under realistic degradations that tampered 
images might undergo during distribution (social media compression, 
re-saving, screenshot capture).
```

```python
# Cell 31 (Code) — Robustness table
robustness_results = evaluate_robustness(model, test_dataset, device, threshold)
print_robustness_table(robustness_results)
```

```markdown
# Cell 32 (Markdown)
### 9.2 Copy-Move vs. Splicing Analysis
CASIA v2.0 contains both splicing and copy-move tampering. 
We analyze performance separately to understand model behavior 
across forgery types.
```

```python
# Cell 33 (Code) — Type breakdown
analyze_by_tampering_type(test_dataset, all_preds, all_gts, threshold)
```

```markdown
# Cell 34 (Markdown) — optional
### 9.3 Cross-Dataset Evaluation: COVERAGE
To test generalization, we evaluate our CASIA-trained model 
on the COVERAGE dataset (100 copy-move forgeries with ground truth).
```

---

## 10.8 What Evaluators Are Looking For

The bonus section demonstrates **engineering maturity**:

1. **Robustness testing** shows you understand that lab performance ≠ real-world performance
2. **Cross-dataset evaluation** shows you think about generalization, not just benchmark numbers
3. **Per-category analysis** shows you understand the problem domain (splicing ≠ copy-move)
4. **Ablation study** shows you can design controlled experiments to validate design decisions

Even if the numbers aren't spectacular, **the act of testing and analyzing** is what earns points.
