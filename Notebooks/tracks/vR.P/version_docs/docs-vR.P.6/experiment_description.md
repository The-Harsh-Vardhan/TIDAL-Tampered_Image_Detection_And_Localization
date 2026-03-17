# Experiment Description — vR.P.6: EfficientNet-B0 Encoder

| Field | Value |
|-------|-------|
| **Version** | vR.P.6 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.1 (ResNet-34 frozen, RGB, proper dataset) |
| **Change** | Replace ResNet-34 encoder with EfficientNet-B0 |
| **Encoder** | EfficientNet-B0 (ImageNet, frozen) |
| **Input** | RGB 384x384x3 (ImageNet normalized) |

---

## 1. Motivation

The pretrained track has used **ResNet-34** as the encoder backbone for all experiments (vR.P.0 through vR.P.4). vR.P.6 tests a fundamentally different encoder architecture: **EfficientNet-B0**.

EfficientNet-B0 is interesting because:
- **4.2x fewer total parameters** than ResNet-34 (5.3M vs 21.8M)
- **Higher ImageNet accuracy** (77.1% vs 73.3%) — more efficient feature extraction
- **Squeeze-excite attention** — learns channel-wise importance, may help focus on forensically relevant features
- **MBConv blocks** — mobile inverted bottleneck convolutions, a modern architecture design

By branching from **vR.P.1** (the clean baseline with frozen encoder, RGB input, proper GT masks), this experiment isolates the encoder change as the **single variable**. All other settings (input, freezing strategy, loss, optimizer, scheduler, data split) remain identical.

---

## 2. What Changed from vR.P.1

| Aspect | vR.P.1 | vR.P.6 (This Version) |
|--------|--------|----------------------|
| **Encoder** | ResNet-34 | **EfficientNet-B0** |
| **ENCODER constant** | `'resnet34'` | **`'efficientnet-b0'`** |
| **Total params** | ~21.8M | **~5.3M** |
| **Trainable params** | ~500K (decoder) | **~400K (decoder)** |
| **Data:param ratio** | 1:57 | **1:45** |
| **Feature channels** | [64, 64, 128, 256, 512] | **[16, 24, 40, 112, 320]** |
| **Attention mechanism** | None | **Squeeze-excite** |
| **Skip connection sizes** | Larger (more spatial detail) | **Smaller (more compressed)** |

---

## 3. What DID NOT Change (Frozen)

- Input: RGB 384x384, ImageNet normalization
- IN_CHANNELS = 3
- Encoder state: Fully frozen
- Encoder weights: ImageNet pretrained
- Architecture: UNet decoder (SMP)
- Loss: BCEDiceLoss
- Optimizer: Adam (lr=1e-3, decoder only)
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=7, monitor=val_loss
- Batch size: 16
- Max epochs: 25
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling (same 3-tier fallback)
- Evaluation: pixel-level + image-level metrics
- Checkpoint save/resume

---

## 4. EfficientNet-B0 Architecture

```
Input: 384x384x3
|
+-- stem:     3x3, 32, stride 2       -> 192x192x32
|
+-- block1:   1x MBConv1, k3x3, 16    -> 192x192x16
+-- block2:   2x MBConv6, k3x3, 24    -> 96x96x24
+-- block3:   2x MBConv6, k5x5, 40    -> 48x48x40
+-- block4:   3x MBConv6, k3x3, 80    -> 24x24x80
+-- block5:   3x MBConv6, k5x5, 112   -> 24x24x112
+-- block6:   4x MBConv6, k5x5, 192   -> 12x12x192
+-- block7:   1x MBConv6, k3x3, 320   -> 12x12x320
|
+-- [UNet Decoder reconstructs to 384x384x1]
```

### Key Differences from ResNet-34

| Feature | ResNet-34 | EfficientNet-B0 |
|---------|-----------|-----------------|
| Building block | BasicBlock (3x3 conv + 3x3 conv) | MBConv (expand + depthwise + SE + project) |
| Attention | None | Squeeze-excite per block |
| Skip channels | [64, 64, 128, 256, 512] | [16, 24, 40, 112, 320] |
| Depth scaling | Fixed | Compound-scaled |
| Parameter count | 21.8M | 5.3M |
| ImageNet Top-1 | 73.3% | 77.1% |

### SMP Compatibility

SMP natively supports EfficientNet-B0 (`encoder_name='efficientnet-b0'`). The UNet decoder automatically adapts to the different skip connection channel sizes. No code changes are needed beyond changing the `ENCODER` constant.
