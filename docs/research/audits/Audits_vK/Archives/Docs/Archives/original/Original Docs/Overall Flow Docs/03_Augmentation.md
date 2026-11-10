# 3. Augmentation — Implementation Guide

## 3.1 Assignment Requirement

> *"Apply relevant data augmentation techniques to ensure model robustness."*

---

## 3.2 Why Augmentation is Critical for Forensics

Augmentation in image forensics serves a **dual purpose**:

1. **Standard ML purpose**: Regularization — prevent overfitting on ~4,200 training images (small by deep learning standards)
2. **Forensic-specific purpose**: Teach the model to recognize tampering artifacts that survive common image processing operations (JPEG re-compression, resizing, brightness shifts)

Without augmentation, the model will:
- Memorize specific noise patterns in the training set
- Fail on images that have been re-saved, shared on social media, or screenshot-captured
- Overfit to CASIA-specific camera artifacts rather than learning general tampering signals

---

## 3.3 Why `albumentations` (Not `torchvision`)

| Feature | `albumentations` | `torchvision.transforms` |
|---------|-----------------|-------------------------|
| **Image + Mask sync** | ✅ Built-in (`image=..., mask=...`) | ❌ Manual; error-prone |
| **Speed** | ⚡ 10-30× faster (OpenCV backend) | 🐢 PIL backend |
| **Transform variety** | 70+ transforms | ~25 transforms |
| **JPEG compression sim** | ✅ `ImageCompression` | ❌ Not available |
| **Segmentation support** | ✅ First-class citizen | ⚠️ Requires workarounds |
| **Industry adoption** | Kaggle standard; used in SMP docs | Legacy in research |

**Critical**: For segmentation tasks, spatial transforms (flip, rotate, crop) must be applied identically to both the image and mask. `albumentations` guarantees this automatically. With `torchvision`, you'd need to manually manage random states or use functional transforms — error-prone and unnecessary.

---

## 3.4 Augmentation Strategy

### Design Principles
1. **Geometric transforms**: Make the model orientation-invariant (tampering can appear in any location/orientation)
2. **Photometric transforms**: Make the model robust to brightness/contrast variations (different capture conditions)
3. **Forensic-relevant distortions**: JPEG compression, Gaussian noise — these are the exact distortions the bonus criteria tests for
4. **No destructive transforms**: Avoid heavy blur, elastic deformation, or cutout — these destroy the forensic artifacts we're trying to detect

### Train vs. Validation/Test Transforms

**Training augmentations** introduce controlled randomness to expand the effective dataset.  
**Validation/Test transforms** only resize and normalize — no randomness, so metrics are deterministic and comparable.

---

## 3.5 Training Transform Pipeline

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    # 1. Resize to fixed input size
    A.Resize(512, 512),
    
    # 2. Geometric transforms (spatial — applied to image AND mask)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    
    # 3. Photometric transforms (pixel — applied to image ONLY, not mask)
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.2
    ),
    
    # 4. Forensic-relevant distortions
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.ImageCompression(
        quality_lower=50,
        quality_upper=100,
        compression_type=A.ImageCompression.ImageCompressionType.JPEG,
        p=0.3
    ),
    
    # 5. Normalize + Convert to tensor
    A.Normalize(
        mean=(0.485, 0.456, 0.406),   # ImageNet statistics
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),  # HWC numpy → CHW tensor
])
```

### Transform-by-Transform Justification

| # | Transform | Type | p | Justification |
|---|-----------|------|---|---------------|
| 1 | `Resize(512, 512)` | Spatial | 1.0 | Standardize input; 512×512 preserves high-frequency forensic artifacts while fitting in T4 VRAM |
| 2a | `HorizontalFlip` | Spatial | 0.5 | Tampering has no preferred horizontal orientation |
| 2b | `VerticalFlip` | Spatial | 0.5 | Tampering has no preferred vertical orientation |
| 2c | `RandomRotate90` | Spatial | 0.5 | Orientation invariance; 90° rotations preserve pixel grid (no interpolation artifacts) |
| 3a | `RandomBrightnessContrast` | Pixel | 0.3 | Simulates different lighting/exposure conditions |
| 3b | `HueSaturationValue` | Pixel | 0.2 | Color variation robustness; conservative limits to avoid unrealistic shifts |
| 4a | `GaussNoise` | Pixel | 0.2 | Simulates sensor noise; trains the model to detect tampering despite noise overlay |
| 4b | `ImageCompression` | Pixel | 0.3 | **Forensic-critical**: simulates JPEG re-compression that tampered images undergo when shared online |
| 5a | `Normalize` | Pixel | 1.0 | Required for pre-trained ImageNet encoders; standardizes input distribution |
| 5b | `ToTensorV2` | Format | 1.0 | Converts HWC numpy array to CHW PyTorch tensor |

---

## 3.6 Validation/Test Transform Pipeline

```python
val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),
])
```

**No augmentation** — validation and test metrics must be deterministic and reproducible.

---

## 3.7 Spatial vs. Pixel Transforms — How `albumentations` Handles Them

Understanding this is important for the notebook narrative:

| Transform Type | Applied To | Mask Affected? | Example |
|---------------|-----------|---------------|---------|
| **Spatial** | Image + Mask | ✅ Yes (identically) | Flip, Rotate, Resize, Crop |
| **Pixel** | Image only | ❌ No | Brightness, Noise, JPEG, Hue |

**Spatial transforms** change pixel positions — so the mask must be transformed identically to maintain alignment.  
**Pixel transforms** change pixel values — the mask should NOT be affected (it's a binary label, not an image).

`albumentations` handles this automatically when you pass `image=..., mask=...`.

---

## 3.8 Transforms We Deliberately Do NOT Use

| Transform | Why Excluded |
|-----------|-------------|
| `RandomCrop` / `CenterCrop` | Can crop out the tampered region entirely, creating wrong label (tampered image → all-zero mask → looks authentic) |
| `ElasticTransform` | Distorts pixel grid in ways that don't occur naturally; destroys CFA/demosaicing patterns that are forensic signals |
| `GaussianBlur` (heavy) | Destroys the noise residuals that SRM filters depend on; light blur from JPEG is already covered by `ImageCompression` |
| `CoarseDropout` / `Cutout` | Removes random patches; would artificially create mask-like patterns, confusing the learning signal |
| `GridDistortion` | Unrealistic distortion that adds noise to the training signal |
| `CLAHE` (aggressive) | Local histogram equalization can create artifacts that mimic tampering boundaries |
| `ColorJitter` (extreme) | Extreme color shifts make the model invariant to color — but color inconsistency IS a forensic signal |

**Principle**: Forensic augmentation must not destroy the very artifacts the model is trying to learn. Conservative transforms that preserve noise structure are preferred.

---

## 3.9 Advanced: Augmentation for Robustness Testing (Bonus)

During **test-time robustness evaluation** (not training), apply distortions to the test set to measure degradation:

```python
# These are applied AFTER training, during robustness evaluation only
robustness_transforms = {
    'JPEG_QF50': A.Compose([
        A.Resize(512, 512),
        A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]),
    'JPEG_QF70': A.Compose([
        A.Resize(512, 512),
        A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]),
    'GaussNoise_s01': A.Compose([
        A.Resize(512, 512),
        A.GaussNoise(var_limit=(100, 100), p=1.0),  # σ ≈ 0.04
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]),
    'Resize_0.5x': A.Compose([
        A.Resize(256, 256),  # Downscale first
        A.Resize(512, 512),  # Then upscale back (quality loss)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]),
}
```

This maps directly to the assignment bonus criteria:
> *"Testing robustness against distortions such as JPEG compression, resizing, cropping, and noise."*

---

## 3.10 Augmentation Verification

Include a visual check in the notebook showing the same image before and after augmentation:

```python
# Show original vs. augmented (5 random augmentations of the same image)
sample_image = np.array(Image.open(train_pairs[0]['image_path']).convert('RGB'))
sample_mask = np.array(Image.open(train_pairs[0]['mask_path']).convert('L'))
sample_mask = (sample_mask > 128).astype(np.float32)

fig, axes = plt.subplots(2, 6, figsize=(24, 8))
axes[0, 0].imshow(sample_image)
axes[0, 0].set_title("Original")
axes[1, 0].imshow(sample_mask, cmap='gray')
axes[1, 0].set_title("Original Mask")

for i in range(1, 6):
    aug = train_transform(image=sample_image, mask=sample_mask)
    img_aug = aug['image'].permute(1, 2, 0).numpy()
    img_aug = img_aug * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_aug = np.clip(img_aug, 0, 1)
    axes[0, i].imshow(img_aug)
    axes[0, i].set_title(f"Aug #{i}")
    axes[1, i].imshow(aug['mask'].numpy(), cmap='gray')
    axes[1, i].set_title(f"Mask #{i}")

for ax in axes.flat:
    ax.axis('off')
plt.suptitle("Augmentation Verification: Image-Mask Alignment", fontsize=14)
plt.tight_layout()
plt.show()
```

**What to verify**: Masks must perfectly align with the corresponding augmented image in every variant.
