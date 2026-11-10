# Architecture Deep Dive: ResNet-34, ResNet-50, EfficientNet-B0

---

## 1. ResNet-34 (Primary Recommendation)

### Architecture

```
Input: 384×384×3
│
├─ conv1:    7×7, 64, stride 2    → 192×192×64
├─ maxpool:  3×3, stride 2        → 96×96×64
│
├─ layer1:   3×[3×3, 64]          → 96×96×64     (residual blocks, identity shortcuts)
├─ layer2:   4×[3×3, 128]         → 48×48×128    (first block: stride 2 downsample)
├─ layer3:   6×[3×3, 256]         → 24×24×256    (first block: stride 2 downsample)
├─ layer4:   3×[3×3, 512]         → 12×12×512    (first block: stride 2 downsample)
│
└─ [U-Net decoder reconstructs to 384×384×1 using skip connections from each layer]
```

### Key Numbers

| Property | Value |
|----------|-------|
| Total encoder params | 21.3M |
| Trainable (frozen encoder) | ~500K (decoder only) |
| ImageNet Top-1 accuracy | 73.3% |
| Feature channels per stage | [64, 64, 128, 256, 512] |
| Downsampling factor | 32× (384→12) |
| Memory (batch=16, 384×384) | ~3.0 GB GPU |
| Training speed | ~45 sec/epoch on T4 |

### Why It Works for Forensics

1. **Identity skip connections** — preserve low-level features (edges, noise patterns) through the entire network. Critical for detecting subtle tampering artifacts.
2. **Multi-scale features** — The U-Net decoder can use features from 4 different resolutions (96×96, 48×48, 24×24, 12×12), capturing both fine-grained pixel artifacts and global context.
3. **Proven in this project** — v6.5 achieved Tam-F1 = 0.41 with exactly this encoder.
4. **Sweet spot of complexity** — Deep enough for meaningful features, small enough to train efficiently on 8,829 images.

### SMP Configuration

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation='sigmoid'
)
```

### Keras Configuration

```python
from tensorflow.keras.applications import ResNet50  # No Keras ResNet34 natively
# For ResNet34 in Keras: use classification_models_keras or tf-hub
# Alternatively, use ResNet50 (see below)
```

**Note:** Keras does not have a native ResNet34 implementation. PyTorch (via SMP or torchvision) is the standard way to use ResNet34. For Keras-only workflows, use ResNet50 instead.

---

## 2. ResNet-50

### Architecture

```
Input: 384×384×3
│
├─ conv1:    7×7, 64, stride 2       → 192×192×64
├─ maxpool:  3×3, stride 2           → 96×96×64
│
├─ layer1:   3×[1×1,64 → 3×3,64 → 1×1,256]      → 96×96×256    (bottleneck blocks)
├─ layer2:   4×[1×1,128 → 3×3,128 → 1×1,512]     → 48×48×512
├─ layer3:   6×[1×1,256 → 3×3,256 → 1×1,1024]    → 24×24×1024
├─ layer4:   3×[1×1,512 → 3×3,512 → 1×1,2048]    → 12×12×2048
│
└─ [U-Net decoder reconstructs to 384×384×1]
```

### Key Numbers

| Property | Value |
|----------|-------|
| Total encoder params | 23.5M |
| Trainable (frozen encoder) | ~600K (decoder only) |
| ImageNet Top-1 accuracy | 76.1% |
| Feature channels per stage | [64, 256, 512, 1024, 2048] |
| Downsampling factor | 32× |
| Memory (batch=16, 384×384) | ~4.0 GB GPU |
| Training speed | ~65 sec/epoch on T4 |

### Differences from ResNet-34

| Aspect | ResNet-34 | ResNet-50 |
|--------|-----------|-----------|
| Block type | Basic (2×3×3 conv) | Bottleneck (1×1 → 3×3 → 1×1) |
| Final feature channels | 512 | 2048 |
| Encoder params | 21.3M | 23.5M |
| ImageNet accuracy | 73.3% | 76.1% |
| Decoder complexity | Lower (fewer skip channels) | Higher (4× more skip channels) |
| Training speed | ~45 sec/epoch | ~65 sec/epoch |

### When to Use ResNet-50 Over ResNet-34

- When ResNet-34's feature representation is insufficient (accuracy plateaus)
- When you need richer skip connection features (2048 vs 512 channels at deepest level)
- When using Keras (native `tf.keras.applications.ResNet50` available)
- When targeting state-of-the-art (ME-Net uses ResNet-50)

### SMP Configuration

```python
model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation='sigmoid'
)
```

### Keras Configuration

```python
from tensorflow.keras.applications import ResNet50

base = ResNet50(weights='imagenet', include_top=False, input_shape=(384, 384, 3))
base.trainable = False

# Build decoder from base.output
# Skip connections from: base.get_layer('conv2_block3_out'), etc.
```

---

## 3. EfficientNet-B0

### Architecture

```
Input: 384×384×3
│
├─ stem:     3×3, 32, stride 2       → 192×192×32
│
├─ block1:   1× MBConv1, k3×3, 16    → 192×192×16
├─ block2:   2× MBConv6, k3×3, 24    → 96×96×24
├─ block3:   2× MBConv6, k5×5, 40    → 48×48×40
├─ block4:   3× MBConv6, k3×3, 80    → 24×24×80
├─ block5:   3× MBConv6, k5×5, 112   → 24×24×112
├─ block6:   4× MBConv6, k5×5, 192   → 12×12×192
├─ block7:   1× MBConv6, k3×3, 320   → 12×12×320
│
├─ head:     1×1, 1280               → 12×12×1280
│
└─ [Decoder reconstructs to 384×384×1]
```

### Key Numbers

| Property | Value |
|----------|-------|
| Total encoder params | 5.3M |
| Trainable (frozen encoder) | ~400K (decoder only) |
| ImageNet Top-1 accuracy | 77.1% |
| Feature channels per stage | [16, 24, 40, 112, 320] |
| Downsampling factor | 32× |
| Memory (batch=16, 384×384) | ~2.5 GB GPU |
| Training speed | ~55 sec/epoch on T4 |

### Key Innovations

1. **Compound scaling** — Width, depth, and resolution are scaled together with a fixed ratio, not independently.
2. **MBConv blocks** — Mobile inverted bottleneck convolutions with squeeze-excite attention. Each block has a channel expansion → depthwise conv → squeeze-excite → pointwise projection pipeline.
3. **Squeeze-excite attention** — Learns channel-wise importance weights. May help focus on forensically relevant features.
4. **4.2× fewer parameters** than ResNet-34 with **higher ImageNet accuracy** (77.1% vs 73.3%).

### When to Use EfficientNet-B0

- When GPU memory is constrained (smallest footprint)
- When you want modern attention mechanisms
- When parameter efficiency matters
- **Concern:** Skip connection structure differs from ResNet; SMP may use different integration points

### SMP Configuration

```python
model = smp.Unet(
    encoder_name='efficientnet-b0',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation='sigmoid'
)
```

### Keras Configuration

```python
from tensorflow.keras.applications import EfficientNetB0

base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(384, 384, 3))
base.trainable = False
```

---

## 4. Decision Matrix

| Criterion (Weight) | ResNet-34 | ResNet-50 | EfficientNet-B0 |
|---------------------|-----------|-----------|-----------------|
| Project evidence (30%) | **10** (v6.5 proven) | 3 (untested) | 3 (untested) |
| Literature support (20%) | 8 | **9** (ME-Net) | 6 (survey mention) |
| Parameter efficiency (10%) | 6 | 5 | **10** |
| T4 compatibility (10%) | 9 | 8 | **10** |
| Feature quality (15%) | 7 | **9** | 8 |
| Implementation risk (15%) | **10** (SMP native) | **10** (SMP native) | 8 (SMP native) |
| **Weighted Score** | **8.4** | **7.0** | **6.5** |

### Final Ranking

1. **ResNet-34** — Start here. Proven, fast, efficient, well-understood.
2. **ResNet-50** — Test if ResNet-34 plateaus. More features, modest cost increase.
3. **EfficientNet-B0** — Test as an efficiency experiment. Interesting but unproven.

---

## 5. Scaling Guidance

### If ResNet-34 is Not Enough

| Current Issue | Next Step | Expected Improvement |
|---------------|-----------|---------------------|
| Accuracy plateau | Try ResNet-50 | +1-3% from richer features |
| Still not enough | Try EfficientNet-B4 | Higher capacity, attention |
| Overfitting | Stay with ResNet-34 but add augmentation | Augmentation may work with pretrained |
| Underfitting | Unfreeze more encoder layers | More model capacity |
| Memory constrained | Try EfficientNet-B0 | Smaller footprint |

### Resolution Scaling

| Resolution | ResNet-34 Batch | ResNet-50 Batch | EfficientNet-B0 Batch | Expected Impact |
|------------|----------------|----------------|----------------------|-----------------|
| 256×256 | 32 | 24 | 32 | Baseline |
| 384×384 | 16 | 12 | 16 | +2-5% (v6.5 used this) |
| 512×512 | 8 | 6 | 10 | Diminishing returns |
