# Experiment Description -- vR.P.5: ResNet-50 Encoder

| Field | Value |
|-------|-------|
| **Version** | vR.P.5 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.1.5 (ResNet-34, RGB, frozen encoder, speed-optimized) |
| **Change** | Swap encoder from ResNet-34 to ResNet-50 |
| **Encoder** | ResNet-50 (ImageNet pretrained, frozen) |
| **Input** | RGB 384x384 (ImageNet normalization) |

---

## 1. Motivation

All pretrained track experiments so far use **ResNet-34** as the UNet encoder. ResNet-34 uses basic residual blocks (two 3x3 convolutions) and produces 512-dimensional features at its deepest layer. This is a reasonable starting point, but deeper encoders may capture more nuanced features.

**ResNet-50** replaces basic blocks with **bottleneck blocks** (1x1 -> 3x3 -> 1x1 convolution). This design:
- Produces **2048-dimensional features** at the final layer (4x wider than ResNet-34's 512)
- Enables **richer skip connections** at every resolution level (256, 512, 1024, 2048 vs 64, 128, 256, 512)
- Has only marginally more encoder parameters (~23.5M vs ~21.3M) despite much wider feature maps

The hypothesis is that these richer features will help the decoder produce more accurate tampered-region masks, especially at fine boundaries where higher-dimensional representations can encode more subtle differences.

This experiment directly tests: **does encoder depth/width matter for forensic localization with frozen features?**

---

## 2. What Changed from vR.P.1.5

| Aspect | vR.P.1.5 (Parent) | vR.P.5 (This Version) |
|--------|-------------------|----------------------|
| Encoder | **ResNet-34** (ImageNet, frozen) | **ResNet-50** (ImageNet, frozen) |
| Block type | BasicBlock (3x3, 3x3) | **Bottleneck** (1x1, 3x3, 1x1) |
| Encoder depth | 34 layers | **50 layers** |
| Final feature dim | 512 | **2048** |
| Skip connection channels | 64, 128, 256, 512 | **256, 512, 1024, 2048** |
| Encoder params | ~21.3M | **~23.5M** |
| Decoder params | ~3.1M | **~8.7M** (wider skip inputs) |
| Total params | ~24.4M | **~32.2M** |
| Trainable params (decoder) | ~3.1M | **~8.7M** |

**Note:** The decoder is automatically adapted by SMP to match the encoder's output channel sizes. This means the decoder becomes larger when using ResNet-50, which is a side effect of the encoder swap -- not a manual change.

---

## 3. What DID NOT Change (Frozen from vR.P.1.5)

- Input type: RGB (raw image)
- Normalization: ImageNet mean/std
- Image size: 384x384
- Architecture: UNet (SMP)
- Encoder state: Frozen (all encoder parameters)
- Loss: BCEDiceLoss (SoftBCEWithLogitsLoss + DiceLoss)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=7, monitor=val_loss
- Batch size: 16
- Max epochs: 25
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling: auto-detect from sibling MASK directory
- AMP mixed precision: Enabled
- TF32 math: Enabled
- Data loading: NUM_WORKERS=2, pin_memory, persistent_workers
- Checkpoint save/resume: Enabled
- Evaluation: pixel-level (F1, IoU, Dice, AUC) + image-level (accuracy, F1, ROC)

---

## 4. Technical Details: ResNet-34 vs ResNet-50

### Block Architecture

```
ResNet-34 BasicBlock:         ResNet-50 Bottleneck:
  input (C channels)            input (C channels)
    |                             |
  3x3 conv, C                  1x1 conv, C/4  (reduce)
  BN + ReLU                    BN + ReLU
  3x3 conv, C                  3x3 conv, C/4  (process)
  BN                           BN + ReLU
    |                           1x1 conv, C    (expand)
  + residual                   BN
  ReLU                           |
                               + residual
                               ReLU
```

### Layer Channel Sizes (used by UNet decoder as skip connections)

| Layer | ResNet-34 | ResNet-50 |
|-------|-----------|-----------|
| conv1 + pool | 64 | 64 |
| layer1 | 64 | 256 |
| layer2 | 128 | 512 |
| layer3 | 256 | 1024 |
| layer4 | 512 | 2048 |

### VRAM Considerations

ResNet-50 uses more memory due to wider intermediate activations. With the encoder frozen (no gradient storage for encoder params), batch_size=16 at 384x384 with AMP should fit within T4's 16GB VRAM. If OOM occurs, the fallback is to reduce batch size to 8.
