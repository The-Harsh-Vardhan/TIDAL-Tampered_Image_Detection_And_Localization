# Experiment Description — vR.P.4: 4-Channel Input (RGB + ELA)

| Field | Value |
|-------|-------|
| **Version** | vR.P.4 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.3 (ELA as input) |
| **Change** | Replace 3ch ELA input with 4ch RGB+ELA (3 RGB + 1 ELA grayscale) |
| **Encoder** | ResNet-34 (ImageNet, frozen body, conv1 unfrozen, BatchNorm unfrozen) |
| **Input** | RGB+ELA 384×384×4 (3 RGB channels + 1 ELA grayscale channel) |

---

## 1. Motivation

vR.P.3 tested ELA as the sole input, discarding RGB information entirely. While ELA provides direct forensic signal (compression artifacts), it loses the visual context that RGB provides — lighting, texture, semantic boundaries. The pretrained encoder's features were optimized for RGB statistics, making ELA a domain-shift challenge.

vR.P.4 tests the hypothesis that **combining both signals** gives the model the best of both worlds: ImageNet-compatible RGB features in channels 0-2, plus forensic ELA signal in channel 3.

The master plan identifies this as a key ablation point. A previous attempt (vK.11-12) tried 4-channel input but failed catastrophically because it changed 5 variables simultaneously. vR.P.4 isolates the 4-channel change as the **single variable**.

---

## 2. What Changed from vR.P.3

| Aspect | vR.P.3 | vR.P.4 (This Version) |
|--------|--------|----------------------|
| **Input type** | ELA only (3ch RGB ELA map) | **RGB + ELA grayscale (4ch)** |
| **IN_CHANNELS** | 3 | **4** |
| **Dataset `__getitem__`** | Compute ELA, return 3ch tensor | **Load RGB (3ch) + compute ELA gray (1ch), concatenate** |
| **Normalization** | ELA-specific mean/std (3 values) | **ImageNet for ch 0-2, ELA gray mean/std for ch 3** |
| **Encoder conv1** | Standard 3ch (frozen) | **Modified 4ch (conv1 UNFROZEN)** |
| **Encoder BN** | Unfrozen | Unfrozen (same) |
| **Trainable params** | ~500K + BN params | **~500K + BN params + conv1 (~12.5K)** |

---

## 3. What DID NOT Change (Frozen)

- Architecture: UNet + ResNet-34 (SMP)
- Loss: BCEDiceLoss
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=7, monitor=val_loss
- Optimizer: Single LR (1e-3) for all trainable params
- Batch size: 16
- Max epochs: 25
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling (same 3-tier fallback)
- AMP + TF32 enabled
- Checkpoint save/resume
- Evaluation: pixel-level + image-level metrics
- ELA preprocessing: JPEG Q=90, brightness-scaled

---

## 4. 4-Channel Preprocessing

```
For each image:
  1. Load as RGB → resize to 384×384 → ToTensor [3, H, W] in [0, 1]
  2. Compute ELA (JPEG Q=90, brightness-scaled) → convert to grayscale ('L')
  3. Resize ELA grayscale to 384×384 → ToTensor [1, H, W] in [0, 1]
  4. Concatenate: torch.cat([rgb_tensor, ela_tensor], dim=0) → [4, H, W]
  5. Normalize channels 0-2 with ImageNet mean/std
  6. Normalize channel 3 with ELA grayscale mean/std (computed from training set)
```

### Why ELA Grayscale (1ch) Instead of ELA RGB (3ch)?

- The master plan explicitly specifies "4-channel" (3+1), not "6-channel" (3+3)
- ELA RGB channels are highly correlated (compression artifacts affect all channels similarly)
- Grayscale ELA captures the essential forensic signal (brightness = artifact magnitude)
- Fewer additional parameters to learn (1 conv1 filter column vs 3)

### Conv1 Modification

SMP handles 4-channel input natively:
```python
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=4)
# SMP: conv1 weights [64, 3, 7, 7] → [64, 4, 7, 7]
# Channels 0-2: pretrained RGB weights (copied)
# Channel 3: average of 3 RGB channel weights (initialization)
```

The conv1 layer must be unfrozen because the 4th channel weights are only initialized (averaged), not trained. They need gradient updates to learn ELA-specific edge/artifact filters.
