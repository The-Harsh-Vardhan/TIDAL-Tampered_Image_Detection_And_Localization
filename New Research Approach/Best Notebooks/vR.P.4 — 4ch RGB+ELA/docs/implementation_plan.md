# Implementation Plan — vR.P.4: 4-Channel Input (RGB + ELA)

## 1. Cell-by-Cell Changes from vR.P.3

| Cell | Type | Change |
|------|------|--------|
| 0 | Markdown | Title → "4-Channel Input (RGB + ELA)", pipeline diagram for dual-path 4ch |
| 1 | Markdown | Add vR.P.4 to changelog, update comparison table |
| 2 | Code | `VERSION='vR.P.4'`, `IN_CHANNELS=4`, add `IMAGENET_MEAN/STD`, update prints |
| 7 | Markdown | Describe 4ch input pipeline, why RGB+ELA |
| 8 | Code | Dataset `__getitem__` loads RGB+ELA gray→concat 4ch. `compute_ela_gray_statistics` (1ch). |
| 9 | Code | Compute ELA gray stats (1 value), pass both ImageNet + ELA gray normalization to datasets |
| 10 | Code | Viz: `denormalize_rgb()` + `denormalize_ela_gray()`, show RGB/ELA gray/Mask/Overlay |
| 11 | Markdown | Architecture diagram with 4ch input, conv1 unfrozen |
| 12 | Code | `in_channels=4`, freeze all → unfreeze `model.encoder.conv1` → unfreeze BN |
| 13 | Markdown | Explain conv1 unfreezing rationale |
| 14 | Code | Print update: "Includes: decoder + encoder BN + conv1 (4ch)" |
| 15 | Code | Print: "4ch RGB+ELA" |
| 22 | Code | Fix P.3 denormalize bug, use `denormalize_rgb(tensor[:3])` for display |
| 25 | Code | Update version references |
| 26 | Markdown | Discussion: 4ch experiment rationale |
| 27 | Code | Config: `input_type='RGB+ELA_4ch'`, `in_channels=IN_CHANNELS` |

Cells 3-6, 16-21, 23-24 remain unchanged.

---

## 2. Key Implementation Details

### 2.1 Dataset `__getitem__` (4-Channel)

```python
def __getitem__(self, idx):
    path = self.image_paths[idx]
    label = self.labels[idx]

    # Load RGB
    img = Image.open(path).convert('RGB')
    img = self.resize(img)
    rgb_tensor = self.to_tensor(img)  # [3, H, W]

    # Compute ELA and convert to grayscale
    ela = compute_ela_image(path, quality=self.ela_quality)
    ela = self.resize(ela)
    ela_gray = ela.convert('L')
    ela_tensor = self.to_tensor(ela_gray)  # [1, H, W]

    # Concatenate: [4, H, W]
    combined = torch.cat([rgb_tensor, ela_tensor], dim=0)

    # Normalize: ch 0-2 ImageNet, ch 3 ELA gray stats
    combined[:3] = self.normalize_rgb(combined[:3])
    combined[3:] = self.normalize_ela(combined[3:])

    mask = get_gt_mask(path, self.mask_size)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    return combined, mask, label
```

### 2.2 ELA Grayscale Statistics

```python
def compute_ela_gray_statistics(image_paths, n_samples=500, size=384):
    """Compute mean and std of ELA grayscale from a sample."""
    # Returns: ([mean], [std]) — single values in list form
```

### 2.3 Normalization Strategy

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# ELA_GRAY_MEAN and ELA_GRAY_STD computed from training set (single values)
```

### 2.4 Encoder Freeze with Conv1 + BN Unfrozen

```python
# Step 1: Freeze ALL encoder parameters
for param in model.encoder.parameters():
    param.requires_grad = False

# Step 2: Unfreeze conv1 (4th channel needs training)
for param in model.encoder.conv1.parameters():
    param.requires_grad = True

# Step 3: Unfreeze BatchNorm layers (domain adaptation)
for module in model.encoder.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        for param in module.parameters():
            param.requires_grad = True
        module.track_running_stats = True
```

---

## 3. Verification Checklist

1. `VERSION = 'vR.P.4'`
2. `IN_CHANNELS = 4`
3. `IMAGENET_MEAN` defined
4. `compute_ela_image` function exists
5. ELA grayscale conversion (`convert('L')`)
6. `torch.cat` for channel concatenation
7. `in_channels=IN_CHANNELS` in smp.Unet call
8. `model.encoder.conv1` unfreezing
9. `nn.BatchNorm2d` unfreezing
10. No `ENCODER_LR` (single LR)
11. Title mentions P.4 + 4-channel
12. `SoftBCEWithLogitsLoss` + `DiceLoss`
13. `ReduceLROnPlateau`
14. 28 cells total
15. Valid JSON

---

## 4. Runtime Estimate

| Component | Time |
|-----------|------|
| RGB loading + ELA computation per image | ~55ms |
| ELA grayscale stats computation (500 images) | ~30s |
| Training per epoch (8,829 images) | ~3.5-4.5 min |
| Total (25 epochs max) | ~85-115 min |

4-channel input adds ~33% more data per batch vs 3ch ELA. VRAM impact is modest and fits on T4/P100 with batch_size=16.
