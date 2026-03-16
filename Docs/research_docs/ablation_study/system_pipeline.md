# System Pipeline Documentation

| Field | Value |
|-------|-------|
| **Notebook** | vR.0 Image Detection and Localisation.ipynb |
| **Paper** | ETASR_9593 (Gorle & Guttavelli, 2025) |
| **Dataset** | CASIA v2.0 Image Tampering Detection |

---

## 1. End-to-End Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
│  Raw Image  │───>│  ELA (Q=90) │───>│ Resize 128²  │───>│ Normalize │───>│   CNN    │
│  (any fmt)  │    │ JPEG resave │    │   + flatten   │    │  [0, 1]   │    │ Softmax  │
└─────────────┘    └─────────────┘    └──────────────┘    └───────────┘    └──────────┘
                                                                              │
                                                                              v
                                                                     ┌──────────────┐
                                                                     │  Authentic /  │
                                                                     │  Tampered     │
                                                                     └──────────────┘
```

---

## 2. Stage Details

### Stage 1: Data Collection

- **Source**: CASIA v2.0 dataset on Kaggle
  - `Au/` directory: ~7,491 authentic images
  - `Tp/` directory: ~5,123 tampered images
- **Formats supported**: JPG, JPEG, PNG, TIF, TIFF, BMP
- **Labels**: Authentic = 0, Tampered = 1
- **Path discovery**: Auto-detects Kaggle (`/kaggle/input/`) or Colab (`/content/drive/`) paths

### Stage 2: ELA Preprocessing

```python
def compute_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    ela_image = ImageChops.difference(original, resaved)
    # Scale brightness to [0, 255]
    max_diff = max(val[1] for val in ela_image.getextrema())
    scale = 255.0 / max(max_diff, 1)
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image
```

**Why ELA works**: JPEG compression applies DCT quantization. When an edited region has a different compression history, re-saving reveals this as brighter error levels in the difference map. Authentic regions show uniform, low error; tampered regions show non-uniform, higher error.

### Stage 3: Image Preparation

- Resize ELA map to 128x128 pixels (bilinear interpolation)
- Convert to NumPy float32 array
- Normalize pixel values to [0, 1] range

### Stage 4: Data Splitting

- **Strategy**: Stratified 70/15/15 split (train/val/test)
- **Random seed**: 42 (reproducible)
- **One-hot encoding**: Labels converted to 2-class one-hot for categorical cross-entropy

### Stage 5: CNN Model

```
Conv2D(32, 5x5, ReLU) -> Conv2D(32, 5x5, ReLU) -> MaxPool(2x2) -> Dropout(0.25)
  -> Flatten -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(2, Softmax)
```

- ~29.5M parameters (dominated by Flatten-to-Dense connection)
- No pretrained weights needed
- Fits comfortably on T4 GPU

### Stage 6: Training

- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Categorical cross-entropy
- **Batch size**: 32
- **Max epochs**: 50
- **Early stopping**: monitor=val_accuracy, patience=5, restore_best_weights=True
- **Metrics tracked**: Accuracy, Precision, Recall

### Stage 7: Evaluation

Computed on held-out test set (never seen during training):

- **Classification metrics**: Accuracy, Precision, Recall, F1 Score
- **ROC-AUC**: Area under ROC curve
- **Confusion matrix**: TP, TN, FP, FN breakdown
- **Classification report**: Per-class precision/recall/F1
- **Visual**: Training curves, ROC curve, sample predictions

---

## 3. Data Flow Diagram

```
CASIA v2.0 Dataset
├── Au/ (7,491 authentic)
└── Tp/ (5,123 tampered)
        │
        v
    collect_image_paths()          # Scan directories, assign labels
        │
        v
    compute_ela() + prepare_image() # ELA -> resize -> normalize
        │
        v
    shuffle + train_test_split()    # Stratified 70/15/15
        │
        ├── X_train, Y_train       # Training set
        ├── X_val, Y_val           # Validation set (for early stopping)
        └── X_test, Y_test         # Test set (for final evaluation)
                │
                v
            model.fit()             # Train CNN
                │
                v
            model.predict(X_test)   # Evaluate on held-out test
                │
                v
            Metrics + Visualizations
```

---

## 4. Runtime Estimates

| Stage | Estimated Time (T4 GPU) |
|-------|------------------------|
| ELA preprocessing (~12K images) | ~5-10 minutes |
| Data splitting + encoding | < 1 second |
| Model training (50 epochs max) | ~5-15 minutes |
| Evaluation + visualization | < 1 minute |
| **Total** | **~15-25 minutes** |

---

## 5. Dependencies

```
tensorflow >= 2.10
numpy
matplotlib
seaborn
Pillow (PIL)
scikit-learn
tqdm
```

All packages are pre-installed on Kaggle and Google Colab.

---

## 6. Pretrained Encoder-Decoder Pipeline (Track 2: vR.P.x)

### 6.1 End-to-End Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Raw Image  │───>│ Resize 384²  │───>│  ImageNet    │───>│  ResNet-34   │───>│  UNet Decoder   │
│  (any fmt)  │    │  + ToTensor  │    │ Normalize    │    │  (Frozen)    │    │  (Trainable)    │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └────────┬────────┘
                                                                                        │
                                                                                        v
                                                                               ┌──────────────────┐
                                                                               │  384×384 Binary   │
                                                                               │  Pixel Mask       │
                                                                               │  (Tampered/Auth)  │
                                                                               └──────────────────┘
```

### 6.2 Comparison: ETASR vs Pretrained Pipeline

```
TRACK 1 (ETASR — Classification Only):
  Raw Image → ELA (Q=90) → Resize 128² → [0,1] normalize → CNN → [P(Au), P(Tp)]

TRACK 2 (Pretrained — Localization):
  Raw Image → Resize 384² → ImageNet normalize → ResNet-34 encoder → UNet decoder → 384×384 pixel mask
```

### 6.3 Stage Details

#### Stage 1: Data Collection (Same as Track 1)

- CASIA v2.0 dataset: ~7,491 Au + ~5,123 Tp images
- Same train/val/test split (70/15/15, seed=42, stratified)

#### Stage 2: Image Preparation (Different from Track 1)

| Aspect | Track 1 (ETASR) | Track 2 (Pretrained) |
|--------|-----------------|---------------------|
| Preprocessing | ELA (JPEG Q=90, brightness scale) | None (raw RGB) |
| Resize | 128×128 | 384×384 |
| Normalization | [0, 1] scaling | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |
| Channels | 3 (ELA RGB) | 3 (natural RGB) |
| Data format | NumPy arrays | PyTorch DataLoader batches |

#### Stage 3: Model (Pretrained Encoder-Decoder)

```
Encoder (ResNet-34, FROZEN):
  conv1(64) → layer1(64) → layer2(128) → layer3(256) → layer4(512)
     ↓            ↓            ↓            ↓            ↓
  [skip]      [skip]      [skip]      [skip]      [bottleneck]
                                                       ↓
Decoder (UNet, TRAINABLE):
  up4(256)+skip → up3(128)+skip → up2(64)+skip → up1(32)+skip → 1×1 conv → sigmoid
                                                                      ↓
                                                                  384×384×1 mask
```

#### Stage 4: Training

| Parameter | Track 1 (ETASR) | Track 2 (Pretrained) |
|-----------|-----------------|---------------------|
| Framework | TensorFlow/Keras | PyTorch + SMP |
| Optimizer | Adam (lr=1e-4) | Adam (decoder lr=1e-3, encoder lr=1e-5 when unfrozen) |
| Loss | categorical_crossentropy | BCEDiceLoss (pixel-level) |
| Batch size | 32 | 16 |
| Max epochs | 50 | 25 |
| Early stopping | val_accuracy, patience=5 | val_loss, patience=7 |
| Trainable params | 29.5M | ~500K |

#### Stage 5: Evaluation

| Metric Category | Track 1 (ETASR) | Track 2 (Pretrained) |
|-----------------|-----------------|---------------------|
| Classification | ✅ Accuracy, Per-class P/R/F1, Macro F1, AUC | ✅ Same (derived from mask) |
| Localization | ❌ Not possible | ✅ Pixel-F1, IoU, Pixel-AUC, Dice |
| Visualization | Confusion matrix, ROC curve | + Original/GT/Predicted/Overlay |

#### Stage 6: Assignment Deliverables

| Requirement | Track 1 (ETASR) | Track 2 (Pretrained) |
|-------------|-----------------|---------------------|
| Tampered region masks | ❌ | ✅ |
| Original / GT / Predicted / Overlay | ❌ | ✅ |
| Model weights | .keras file | .pth file |
| Classification metrics | ✅ | ✅ |
| T4 GPU compatible | ✅ | ✅ |

### 6.4 Data Flow Diagram (Pretrained Track)

```
CASIA v2.0 Dataset
├── Au/ (7,491 authentic)
└── Tp/ (5,123 tampered)
        │
        v
    collect_image_paths()                  # Scan directories, assign labels
        │
        v
    RGB loading + ImageNet normalization   # No ELA — raw RGB at 384×384
        │
        v
    train_test_split() (70/15/15)          # Stratified, seed=42
        │
        ├── train_loader (batch=16)
        ├── val_loader (batch=16)
        └── test_loader (batch=16)
                │
                v
            model = smp.Unet('resnet34')   # Pretrained encoder + UNet decoder
                │
                v
            Training loop (25 epochs)       # BCEDiceLoss, ReduceLROnPlateau
                │
                v
            model.predict(test_loader)      # Pixel-level masks
                │
                v
            Metrics + Visualizations
            ├── Pixel F1, IoU, AUC
            ├── Confusion matrix (per-pixel)
            ├── Original/GT/Predicted/Overlay grid
            └── Training curves
```

### 6.5 Runtime Estimates (Pretrained Track)

| Stage | Estimated Time (T4 GPU) |
|-------|------------------------|
| SMP installation (`pip install`) | ~30 seconds |
| Image loading + normalization (12K images at 384²) | ~3-5 minutes |
| Data splitting + DataLoader setup | < 5 seconds |
| Model build + freeze | < 2 seconds |
| Training (25 epochs, batch=16) | ~15-20 minutes |
| Evaluation + visualization | ~2 minutes |
| **Total** | **~25-30 minutes** |

### 6.6 Dependencies (Pretrained Track)

```
torch >= 1.10
torchvision
segmentation-models-pytorch >= 0.3.0
numpy
matplotlib
seaborn
Pillow (PIL)
scikit-learn
tqdm
```

`torch` and `torchvision` are pre-installed on Kaggle. `segmentation-models-pytorch` requires `pip install` in the first notebook cell.
