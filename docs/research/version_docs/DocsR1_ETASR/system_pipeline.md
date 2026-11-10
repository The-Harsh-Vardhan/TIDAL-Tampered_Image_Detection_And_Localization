# System Pipeline — ELA + CNN Tampering Detection

---

## 1. End-to-End Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    COMPLETE SYSTEM PIPELINE                          │
│                                                                      │
│  ┌─────────┐    ┌─────────┐    ┌──────┐    ┌──────┐    ┌─────────┐ │
│  │ Dataset  │───▶│  ELA    │───▶│Resize│───▶│Norm. │───▶│  CNN    │ │
│  │ Loading  │    │ Preproc │    │128² │    │[0,1] │    │ Model   │ │
│  └─────────┘    └─────────┘    └──────┘    └──────┘    └────┬────┘ │
│       │                                                      │      │
│       │              ┌───────────────────────────────────┐   │      │
│       └──────────────│        Label Encoding             │───┘      │
│                      │  Authentic=0, Tampered=1          │          │
│                      │  One-hot: [1,0] / [0,1]           │          │
│                      └───────────────────────────────────┘          │
│                                      │                              │
│                                      ▼                              │
│                      ┌───────────────────────────────────┐          │
│                      │     Train / Validation Split      │          │
│                      │          80% / 20%                │          │
│                      └───────────────────────────────────┘          │
│                                      │                              │
│                                      ▼                              │
│                      ┌───────────────────────────────────┐          │
│                      │         Model Training            │          │
│                      │  Adam, lr=0.0001, early stopping  │          │
│                      └───────────────────────────────────┘          │
│                                      │                              │
│                                      ▼                              │
│                      ┌───────────────────────────────────┐          │
│                      │         Evaluation                │          │
│                      │  Accuracy, Precision, Recall, F1  │          │
│                      │  Confusion Matrix, Training Curves│          │
│                      └───────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage Details

### Stage 1: Dataset Loading

```
CASIA2/
├── Au/    (Authentic images — ~7,491 images)
└── Tp/    (Tampered images  — ~5,123 images)

Supported formats: .jpg, .jpeg, .png, .tif, .bmp
```

**Process:**
1. Walk through Au/ directory → label as 0 (Authentic)
2. Walk through Tp/ directory → label as 1 (Tampered)
3. Collect file paths and labels
4. Report dataset statistics (class distribution)

**Improvement over reference:** Load ALL image formats, not just .jpg

---

### Stage 2: ELA Preprocessing

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│ Original   │────▶│ Re-save as │────▶│ Compute    │────▶│ Scale to   │
│ Image (RGB)│     │ JPEG Q=90  │     │ |Orig - Re││     │ [0, 255]   │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
```

**Algorithm:**
```python
def compute_ela(image_path, quality=90):
    image = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    ela = ImageChops.difference(image, resaved)
    max_diff = max(v[1] for v in ela.getextrema())
    scale = 255.0 / max(max_diff, 1)
    return ImageEnhance.Brightness(ela).enhance(scale)
```

**Key improvement:** Use `BytesIO` in-memory buffer instead of writing temp files to disk.

---

### Stage 3: Resize

```
ELA Image (variable size) ──▶ Resize to 128 × 128 × 3
```

Using PIL's `resize()` with bilinear interpolation (default).

---

### Stage 4: Normalization

```
Pixel values [0, 255] ──▶ Divide by 255.0 ──▶ [0.0, 1.0]
```

Applied as NumPy array operation: `np.array(ela_image) / 255.0`

---

### Stage 5: Label Encoding

```
Authentic → 0 → [1, 0]  (one-hot)
Tampered  → 1 → [0, 1]  (one-hot)
```

Using `tensorflow.keras.utils.to_categorical(labels, num_classes=2)`

---

### Stage 6: Data Splitting

```
Full Dataset
    │
    ├── 80% Training Set
    │
    └── 20% Validation Set
```

Using `sklearn.model_selection.train_test_split` with `random_state=42` for reproducibility.

---

### Stage 7: CNN Model

```
Layer (type)                 Output Shape          Param #
═══════════════════════════════════════════════════════════
Conv2D-1 (32, 5×5, ReLU)    (None, 124, 124, 32)  2,432
Conv2D-2 (32, 5×5, ReLU)    (None, 120, 120, 32)  25,632
MaxPooling2D (2×2)           (None, 60, 60, 32)    0
Dropout (0.25)               (None, 60, 60, 32)    0
Flatten                      (None, 115200)        0
Dense (256, ReLU)            (None, 256)           29,491,456
Dropout (0.5)                (None, 256)           0
Dense (2, Softmax)           (None, 2)             514
═══════════════════════════════════════════════════════════
Total params: ~29,520,034
Trainable params: ~29,520,034
```

---

### Stage 8: Training

```
Optimizer:      Adam (lr=0.0001)
Loss:           categorical_crossentropy
Batch Size:     32
Max Epochs:     50
Early Stopping: monitor='val_accuracy', patience=5
Callbacks:      EarlyStopping, optional ModelCheckpoint
```

---

### Stage 9: Evaluation

```
Test Predictions
      │
      ├── Accuracy    = (TP + TN) / Total
      ├── Precision   = TP / (TP + FP)
      ├── Recall      = TP / (TP + FN)
      ├── F1 Score    = 2 × (P × R) / (P + R)
      │
      ├── Confusion Matrix (2×2 heatmap)
      │
      ├── Training Loss Curve
      └── Training Accuracy Curve
```

---

## 3. Data Flow Diagram

```
                    ┌─────────────────┐
                    │   Raw Image     │
                    │  (JPEG/PNG)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Convert to    │
                    │      RGB        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Re-save as     │
                    │  JPEG (Q=90)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Pixel-wise     │
                    │  Difference     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Brightness     │
                    │  Enhancement    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Resize to      │
                    │  128 × 128      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Normalize      │
                    │  ÷ 255.0        │
                    └────────┬────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
      ┌────────▼────────┐        ┌────────▼────────┐
      │  Training Set   │        │  Validation Set  │
      │    (80%)        │        │    (20%)         │
      └────────┬────────┘        └────────┬────────┘
               │                          │
               └──────────┬───────────────┘
                          │
                 ┌────────▼────────┐
                 │    CNN Model    │
                 │  (Sequential)  │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  Predictions    │
                 │  [P(Au), P(Tp)] │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  Evaluation     │
                 │  Metrics +      │
                 │  Visualizations │
                 └─────────────────┘
```

---

## 4. Implementation Notes

### Memory Management

- Use `tf.data.Dataset` pipeline or generator-based loading for large datasets
- Process ELA images in batches rather than loading all into RAM
- Delete large temporary arrays after splitting

### Reproducibility

- Set random seeds: `np.random.seed(42)`, `tf.random.set_seed(42)`
- Use deterministic splitting: `random_state=42`
- Document all hyperparameters

### File Handling

- Use `io.BytesIO` for ELA temp file to avoid disk I/O race conditions
- Handle corrupt/unreadable images gracefully with try/except
- Support multiple image formats via PIL
