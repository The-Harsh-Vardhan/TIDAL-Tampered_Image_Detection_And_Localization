# Implementation Details

| Field | Value |
|-------|-------|
| **Notebook** | vR.0 Image Detection and Localisation.ipynb |
| **Framework** | TensorFlow / Keras (Sequential API) |
| **Based on** | ETASR_9593 paper + reconstructed reference code |

---

## 1. Reference Code Reconstruction

### Sources Used
1. `CASIA2code.py` — Kaggle-sourced ELA + CNN implementation
2. `code.py` — Second Kaggle-sourced implementation
3. ETASR_9593 paper (Table III, Section III)
4. GitHub repos: agusgun/FakeImageDetector, Divyansh-git10/image-forgery-detection

### What Was Kept
- ELA algorithm (PIL-based JPEG recompression + brightness scaling)
- General pipeline structure (load -> preprocess -> train -> evaluate)

### What Was Rebuilt from Paper
- CNN architecture (exact layer specifications from Table III)
- Training hyperparameters (Adam lr=0.0001, categorical cross-entropy)
- Input dimensions (128x128x3)
- Output layer (Dense(2, Softmax))

### 11 Bugs Fixed from Reference Code

| Bug | Impact | Fix |
|-----|--------|-----|
| Image size 150x150 | Wrong architecture | 128x128 per paper |
| Dense(150) | Wrong capacity | Dense(256) per paper |
| Sigmoid output | Wrong activation for 2-class | Softmax per paper |
| X shuffled without Y | Destroyed label correspondence | sklearn paired shuffle |
| Dense(1, softmax) | Always outputs 1.0 | Dense(2, softmax) |
| Double reshape 150->128 | Runtime crash | Single consistent 128x128 |
| Temp file for ELA | Not thread-safe, slow | In-memory BytesIO |
| binary_crossentropy with one-hot | Wrong loss function | categorical_crossentropy |
| Early stopping commented out | No regularization | Enabled with patience=5 |
| val_acc (deprecated key) | Warning/silent failure | val_accuracy |
| Only .jpg loaded | Missed 40%+ of dataset | All formats supported |

---

## 2. ELA Implementation

### Algorithm
```python
def compute_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    ela_image = ImageChops.difference(original, resaved)
    extrema = ela_image.getextrema()
    max_diff = max(val[1] for val in extrema)
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image
```

### Key Design Decisions
- **In-memory JPEG**: Uses `BytesIO()` instead of temp files (fixes thread-safety issue in reference code)
- **Quality 90**: Matches paper specification (higher quality = subtler differences = harder but more discriminative)
- **Brightness scaling**: Normalizes to full [0, 255] range for maximum contrast
- **3-channel output**: ELA map retains RGB channels (each channel shows compression error independently)

---

## 3. Model Architecture

### Sequential CNN (from Paper Table III)

```python
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', padding='valid', input_shape=(128, 128, 3)),
    Conv2D(32, (5, 5), activation='relu', padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
```

### Parameter Breakdown
| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D(32, 5x5) | (124, 124, 32) | 2,432 |
| Conv2D(32, 5x5) | (120, 120, 32) | 25,632 |
| MaxPooling2D | (60, 60, 32) | 0 |
| Dropout(0.25) | (60, 60, 32) | 0 |
| Flatten | (115,200) | 0 |
| Dense(256) | (256) | 29,491,456 |
| Dropout(0.5) | (256) | 0 |
| Dense(2) | (2) | 514 |
| **Total** | | **~29.5M** |

The parameter count is dominated by the Flatten->Dense(256) connection (115,200 x 256 = 29.5M). This is a known characteristic of this architecture.

---

## 4. Training Configuration

```python
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
```

### Data Split Strategy
- **70% train / 15% validation / 15% test** (stratified by label)
- Validation set used for early stopping + monitoring
- Test set used only for final evaluation (never seen during training)
- Random seed = 42 for reproducibility

---

## 5. Evaluation Metrics

### Computed on Test Set
| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Correct positive rate |
| Recall | TP/(TP+FN) | Detection rate |
| F1 Score | 2*(P*R)/(P+R) | Harmonic mean of P and R |
| ROC-AUC | Area under ROC curve | Threshold-independent discrimination |

### Visualizations
- Training/validation loss curves
- Training/validation accuracy curves
- Precision and recall curves over epochs
- Confusion matrix heatmap
- ROC curve with AUC
- Sample correct and incorrect predictions with confidence scores

---

## 6. Reproducibility

| Setting | Value |
|---------|-------|
| Python random seed | 42 |
| NumPy random seed | 42 |
| TensorFlow random seed | 42 |
| Train/test split seed | 42 |
| Dataset order | sorted(os.listdir()) |
| Shuffle | sklearn shuffle with seed=42 |

---

## 7. Pretrained Encoder-Decoder Implementation (Track 2: vR.P.x)

### Framework: PyTorch + Segmentation Models PyTorch (SMP)

The pretrained track uses PyTorch + SMP instead of TensorFlow/Keras because:
- SMP has native ResNet-34 support (Keras does not)
- Built-in freeze/unfreeze, segmentation losses (Dice, BCE, Focal)
- v6.5 used this exact stack and achieved the project's best localization result

### Model Architecture: UNet + ResNet-34

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',   # Pretrained on ImageNet
    in_channels=3,                # RGB input (384×384×3)
    classes=1,                    # Binary mask output
    activation='sigmoid'          # Pixel-wise probability [0, 1]
)
```

### Architecture Diagram

```
Input: 384×384×3 (RGB)
│
├── ENCODER (ResNet-34, ImageNet pretrained, FROZEN)
│   ├─ conv1:    7×7, 64, stride 2    → 192×192×64
│   ├─ maxpool:  3×3, stride 2        → 96×96×64
│   ├─ layer1:   3×[3×3, 64]          → 96×96×64      ──── skip ────┐
│   ├─ layer2:   4×[3×3, 128]         → 48×48×128     ──── skip ──┐ │
│   ├─ layer3:   6×[3×3, 256]         → 24×24×256     ── skip ──┐ │ │
│   └─ layer4:   3×[3×3, 512]         → 12×12×512     ─ skip ─┐ │ │ │
│                                                               │ │ │ │
├── DECODER (UNet-style, TRAINABLE)                             │ │ │ │
│   ├─ up1: 12→24,  concat with layer3 skip  ─────────────────┘ │ │ │
│   ├─ up2: 24→48,  concat with layer2 skip  ───────────────────┘ │ │
│   ├─ up3: 48→96,  concat with layer1 skip  ─────────────────────┘ │
│   ├─ up4: 96→192, concat with conv1 skip   ───────────────────────┘
│   └─ final: 192→384, 1×1 conv → sigmoid
│
└── Output: 384×384×1 (pixel-level probability mask)
```

### Parameter Breakdown

| Component | Parameters | Trainable? |
|-----------|-----------|------------|
| Encoder (ResNet-34) | 21,284,672 | **No (frozen)** |
| Decoder (UNet blocks) | ~450,000 | **Yes** |
| Segmentation head (1×1 conv) | ~50,000 | **Yes** |
| **Total trainable** | **~500,000** | — |

Compared to ETASR CNN's 29,520,034 fully trainable parameters — **59× fewer trainable params**.

### Freezing and Unfreezing

```python
# Phase 1: Freeze encoder (vR.P.0)
for param in model.encoder.parameters():
    param.requires_grad = False

# Phase 2: Gradual unfreeze last 2 blocks (vR.P.2)
def unfreeze_encoder(model, num_blocks=2):
    children = list(model.encoder.children())
    for child in children[-num_blocks:]:
        for param in child.parameters():
            param.requires_grad = True

# Differential learning rate after unfreezing
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},   # 10× lower for encoder
    {'params': model.decoder.parameters(), 'lr': 1e-3},
    {'params': model.segmentation_head.parameters(), 'lr': 1e-3},
])
```

### Training Configuration

```python
# Loss: combination of BCE and Dice (v6.5 config)
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss

bce_loss = SoftBCEWithLogitsLoss()
dice_loss = DiceLoss(mode='binary')
criterion = lambda pred, target: bce_loss(pred, target) + dice_loss(pred, target)

# Optimizer (decoder only — encoder is frozen)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-5
)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# Early stopping: patience=7 (more than ETASR's 5 — pretrained converges slower)
```

### Data Pipeline (PyTorch)

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ImageNet normalization (for RGB input with pretrained encoder)
IMAGE_SIZE = 384
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CASIADataset(Dataset):
    def __init__(self, image_paths, masks, transform=None):
        self.image_paths = image_paths
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = self.masks[idx]  # Binary mask (H×W)
        if self.transform:
            image = self.transform(image)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
```

### Classification Variant (Pretrained Encoder + Classification Head)

For direct comparison with ETASR before localization:

```python
import torch.nn as nn

class ForensicClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling (fixes W10)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.encoder(x)[-1]  # Last stage: 12×12×512
        pooled = self.pool(features)     # 1×1×512
        return self.classifier(pooled)   # [batch, 2]
```

This gives ~650K trainable parameters (vs ETASR's 29.5M) with ImageNet features as a frozen backbone.

### ELA Input Adaptation (vR.P.3)

When using ELA as input to the pretrained encoder, the BatchNorm domain shift must be addressed:

```python
# ELA images have different statistics than ImageNet
# Option 1: Compute ELA-specific normalization
ela_mean = compute_channel_mean(train_ela_images)  # e.g., [0.08, 0.07, 0.09]
ela_std = compute_channel_std(train_ela_images)     # e.g., [0.10, 0.09, 0.11]
ela_transform = transforms.Normalize(mean=ela_mean, std=ela_std)

# Option 2: Unfreeze BatchNorm layers only (keep conv weights frozen)
for module in model.encoder.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.requires_grad_(True)
        module.track_running_stats = True
```

### Key Differences: ETASR vs Pretrained Implementation

| Aspect | ETASR (vR.1.x) | Pretrained (vR.P.x) |
|--------|----------------|---------------------|
| Framework | TensorFlow/Keras | PyTorch + SMP |
| Model type | Sequential CNN | UNet encoder-decoder |
| Input | ELA 128×128 | RGB 384×384 |
| Params trained | 29.5M (all) | ~500K (decoder only) |
| Output | [P(Au), P(Tp)] | 384×384 pixel mask |
| Training API | `model.fit()` | Custom training loop |
| Loss | categorical_crossentropy | BCEDiceLoss |
| Normalization | [0, 1] scaling | ImageNet mean/std |
