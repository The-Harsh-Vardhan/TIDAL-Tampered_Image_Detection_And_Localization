# Architecture Overview — ETASR_9593

## Enhanced Image Tampering Detection using Error Level Analysis and a CNN

---

## 1. Paper Summary

This paper presents a binary image tampering detection system that combines **Error Level Analysis (ELA)** as a forensic preprocessing step with a **compact Convolutional Neural Network (CNN)** classifier. The key insight is that ELA preprocessing makes compression inconsistencies visible, allowing even a small CNN to achieve competitive accuracy against larger pretrained models on the CASIA v2.0 dataset.

**Core Claim:** The input representation (ELA) matters as much as network depth — a lightweight CNN with forensic preprocessing can rival deeper architectures.

---

## 2. Preprocessing Pipeline

### 2.1 Input Images

- **Source:** CASIA v2.0 Image Tampering Detection Dataset
- **Classes:** Authentic (Au/) and Tampered (Tp/)
- **Format:** Mixed JPEG/PNG images of varying dimensions

### 2.2 Error Level Analysis (ELA)

ELA is a passive image forensics technique that detects regions with different JPEG compression histories.

**Algorithm:**

```
1. Load original image I and convert to RGB
2. Re-save I as JPEG at quality level Q (Q = 90)
3. Load the re-saved image I'
4. Compute pixel-wise difference: ELA(x,y) = |I(x,y) - I'(x,y)|
5. Find maximum difference across all channels: max_diff
6. Compute scale factor: scale = 255.0 / max_diff
7. Enhance brightness of difference image by scale factor
8. Return the ELA image
```

**Intuition:** Authentic regions (compressed uniformly) show low, uniform error. Tampered/spliced regions (compressed differently) show higher, non-uniform error — appearing brighter in the ELA map.

### 2.3 Image Resizing

- **Target size:** 128 × 128 × 3 pixels
- All ELA images are resized to this uniform dimension before feeding into the CNN

### 2.4 Normalization

- Pixel values scaled to [0, 1] range by dividing by 255.0
- Applied after ELA conversion and resizing

---

## 3. CNN Architecture

The paper describes a compact Sequential CNN with the following layer structure:

```
┌─────────────────────────────────────────────┐
│              INPUT LAYER                     │
│           128 × 128 × 3 (ELA image)         │
├─────────────────────────────────────────────┤
│         CONV2D LAYER 1                       │
│   32 filters, 5×5 kernel, ReLU activation    │
│   padding: valid                             │
│   Output: 124 × 124 × 32                    │
├─────────────────────────────────────────────┤
│         CONV2D LAYER 2                       │
│   32 filters, 5×5 kernel, ReLU activation    │
│   padding: valid                             │
│   Output: 120 × 120 × 32                    │
├─────────────────────────────────────────────┤
│         MAX POOLING                          │
│   Pool size: 2×2                             │
│   Output: 60 × 60 × 32                      │
├─────────────────────────────────────────────┤
│         DROPOUT                              │
│   Rate: 0.25                                 │
├─────────────────────────────────────────────┤
│         FLATTEN                              │
│   Output: 115,200                            │
├─────────────────────────────────────────────┤
│         DENSE LAYER                          │
│   256 units, ReLU activation                 │
├─────────────────────────────────────────────┤
│         DROPOUT                              │
│   Rate: 0.5                                  │
├─────────────────────────────────────────────┤
│         OUTPUT LAYER                         │
│   2 units, Softmax activation                │
│   (Authentic vs Tampered)                    │
└─────────────────────────────────────────────┘
```

### Architecture Diagram (ASCII)

```
Input Image (any size)
        │
        ▼
   ┌─────────┐
   │  ELA    │  Re-save at Q=90, compute difference, scale
   │ Preproc │
   └────┬────┘
        │
        ▼
   Resize to 128×128×3
        │
        ▼
   Normalize [0, 1]
        │
        ▼
┌───────────────┐
│  Conv2D 32    │  5×5, ReLU, valid padding
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Conv2D 32    │  5×5, ReLU, valid padding
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  MaxPool 2×2  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Dropout 0.25  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Flatten     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Dense 256     │  ReLU
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Dropout 0.5   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Dense 2       │  Softmax → [P(authentic), P(tampered)]
└───────────────┘
```

---

## 4. Training Pipeline

| Parameter        | Value                          |
|------------------|--------------------------------|
| Optimizer        | Adam                           |
| Learning Rate    | 0.0001                         |
| Loss Function    | Categorical Cross-Entropy      |
| Batch Size       | 32                             |
| Epochs           | Up to 50 (with early stopping) |
| Train/Val Split  | 80% / 20%                      |
| Early Stopping   | Monitor val_accuracy, patience 5|

### Training Configuration Notes

- Labels are one-hot encoded (2 classes): `[1, 0]` = Authentic, `[0, 1]` = Tampered
- Since the output uses softmax with 2 units, categorical cross-entropy is the correct loss
- Early stopping prevents overfitting by halting training when validation accuracy plateaus

---

## 5. Evaluation Metrics

The paper evaluates the model using:

| Metric           | Description                                       |
|------------------|---------------------------------------------------|
| **Accuracy**     | Overall correct predictions / total predictions    |
| **Precision**    | TP / (TP + FP) — reliability of positive class     |
| **Recall**       | TP / (TP + FN) — completeness of positive class    |
| **F1 Score**     | Harmonic mean of Precision and Recall              |
| **Confusion Matrix** | 2×2 matrix showing TP, TN, FP, FN             |

### Training Visualizations

- **Loss curves:** Training loss vs. Validation loss over epochs
- **Accuracy curves:** Training accuracy vs. Validation accuracy over epochs

---

## 6. Key Design Decisions

### Why ELA?

ELA exposes JPEG compression inconsistencies that are invisible in raw RGB. Spliced or edited regions were often compressed at different quality levels or have been through a different compression pipeline, making them distinguishable in the ELA domain.

### Why a Small CNN?

The paper argues that with good preprocessing (ELA), a compact CNN is sufficient. This is significant because:

1. **Compute efficient** — trainable on consumer hardware or free cloud GPUs
2. **Fast inference** — suitable for practical deployment
3. **Less prone to overfitting** on small-to-medium datasets like CASIA v2.0

### Why 128×128 Input?

A balance between preserving enough spatial detail for forensic analysis and keeping the model computationally tractable.

---

## 7. Reported Results

The paper reports competitive accuracy on CASIA v2.0, demonstrating that the ELA+CNN combination can approach the performance of larger pretrained models (VGG, ResNet) while using significantly fewer parameters.

---

## 8. Limitations

1. **Classification only** — no pixel-level localization of tampered regions
2. **JPEG dependency** — ELA is most effective on JPEG images; may be less reliable on PNG or uncompressed formats
3. **Single dataset** — evaluation limited to CASIA v2.0
4. **No augmentation** — the paper does not describe data augmentation strategies
