# Implementation Strategy: Pretrained Models for Tampering Detection

---

## 1. Phased Rollout Plan

### Phase 1: Re-establish v6.5 Baseline (vR.P.0)

**Goal:** Reproduce v6.5's Tam-F1 = 0.41 result with the clean vR evaluation pipeline.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Encoder | ResNet-34 | Proven in v6.5 |
| Encoder weights | ImageNet | Standard transfer learning |
| Encoder frozen | **Yes (fully frozen)** | Protect pretrained features |
| Decoder | UNet (SMP default) | Skip connections from all 4 encoder stages |
| Input | RGB (3 channels) | Clean transfer, no domain mismatch |
| Input size | 384×384 | v6.5 used this; 3× more pixels than ETASR's 128×128 |
| Loss | BCEDiceLoss | Combination of pixel BCE and Dice (v6.5 config) |
| Optimizer | Adam | Decoder only, lr=1e-3 |
| Batch size | 16 | Fits T4 with 384×384 |
| Epochs | 25 | v6.5 setting |
| Augmentation | None (first) | Establish clean baseline |
| Evaluation | Pixel F1, IoU, AUC, Accuracy | Full segmentation metrics |
| Ground truth | CASIA v2.0 edge masks (if available) or ELA-thresholded pseudo-masks | Need to verify GT availability |

**Critical prerequisite:** Verify CASIA v2.0 provides pixel-level ground truth masks. If not, you'll need to generate pseudo-masks (ELA thresholding) or use a different dataset split with actual masks.

### Phase 2: Gradual Unfreeze (vR.P.1)

| Change | Details |
|--------|---------|
| Unfreeze last 2 encoder blocks (layer3, layer4) | Allows encoder to adapt to forensic domain |
| Differential learning rate | Encoder: 1e-5, Decoder: 1e-3 |
| All else frozen | Same as Phase 1 |

### Phase 3: Test Alternative Inputs (vR.P.2 — vR.P.3)

| Version | Input | Channels | Notes |
|---------|-------|----------|-------|
| vR.P.2 | ELA only | 3 | Test if ELA maps transfer better/worse than RGB |
| vR.P.3 | RGB + ELA | 4 | Concatenate as 4-channel input; modify first conv |

### Phase 4: Test Alternative Encoders (vR.P.4 — vR.P.5)

| Version | Encoder | Notes |
|---------|---------|-------|
| vR.P.4 | ResNet-50 | Test deeper features |
| vR.P.5 | EfficientNet-B0 | Test modern architecture |

---

## 2. Framework Decision

### Option A: PyTorch + SMP (Recommended)

```python
# pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation='sigmoid'
)

# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Differential LR (for later unfreezing)
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.decoder.parameters(), 'lr': 1e-3},
    {'params': model.segmentation_head.parameters(), 'lr': 1e-3},
])
```

| Pros | Cons |
|------|------|
| SMP has 10+ encoders pre-integrated | Different framework from ETASR notebooks (Keras) |
| Native ResNet-34 support | Kaggle needs `pip install` |
| Built-in loss functions (Dice, Focal, Lovasz) | Learning curve if unfamiliar with PyTorch |
| Well-documented freeze/unfreeze | |
| v6.5 used this exact approach | |

### Option B: TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Encoder
encoder = ResNet50(weights='imagenet', include_top=False, input_shape=(384, 384, 3))
encoder.trainable = False

# Build U-Net decoder manually or use keras-segmentation library
# Skip connections require extracting intermediate layers
```

| Pros | Cons |
|------|------|
| Consistent with ETASR notebooks | No native ResNet-34 (must use ResNet-50+) |
| Kaggle pre-installed | Manual U-Net decoder construction |
| Familiar API | Skip connection wiring is error-prone |
| | No built-in segmentation loss library |

### Verdict

**Use PyTorch + SMP.** It's what v6.5 used, it has native ResNet-34 support, and it provides built-in losses, metrics, and encoder freeze/unfreeze with one line of code. The minimal friction of switching framework is worth the vastly simpler implementation.

---

## 3. Ground Truth Masks

### The Localization Challenge

CASIA v2.0 **does not provide pixel-level ground truth masks** for all images. Some versions include edge-based masks, but coverage is inconsistent.

### Solution Options

| Option | Pro | Con |
|--------|-----|-----|
| **A: Use CASIA mask subset** | Real ground truth | Limited coverage |
| **B: ELA-thresholded pseudo-masks** | Available for all images | Noisy, circular logic if using ELA input |
| **C: Splicing boundary masks** | Can generate from filenames | Only works for splicing, not copy-move |
| **D: Classification only with pretrained encoder** | Simple, no masks needed | Doesn't satisfy localization requirement |

**Recommended:** Start with **Option D** (classification with pretrained encoder as feature extractor) to establish a performance baseline, then investigate mask availability for localization. The classification comparison against ETASR is the immediate priority.

### Classification Architecture with Pretrained Encoder

```python
import segmentation_models_pytorch as smp
import torch.nn as nn

# Use encoder only (no decoder) for classification
encoder = smp.encoders.get_encoder('resnet34', in_channels=3, weights='imagenet')
encoder.eval()  # Freeze

class ForensicClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.encoder(x)[-1]  # Last stage features
        pooled = self.pool(features)
        return self.classifier(pooled)
```

This gives you a pretrained ResNet-34 classifier with:
- ~650K trainable parameters (vs ETASR's 29.5M)
- GlobalAveragePooling instead of Flatten (addresses W10)
- ImageNet features as frozen backbone

---

## 4. Training Configuration

### Frozen Encoder (Phase 1)

```python
# Loss
criterion = nn.CrossEntropyLoss()  # For classification
# OR
criterion = smp.losses.DiceLoss(mode='binary') + smp.losses.SoftBCEWithLogitsLoss()  # For segmentation

# Optimizer (decoder only)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-5
)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# Early stopping
patience = 7  # More patience than ETASR's 5 — pretrained models converge slower
```

### Gradual Unfreeze (Phase 2)

```python
# After N epochs, unfreeze last encoder blocks
def unfreeze_encoder(model, num_blocks=2):
    """Unfreeze last num_blocks of the encoder."""
    children = list(model.encoder.children())
    for child in children[-num_blocks:]:
        for param in child.parameters():
            param.requires_grad = True

# Differential LR after unfreezing
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},   # 10× lower for encoder
    {'params': model.decoder.parameters(), 'lr': 1e-3},
])
```

---

## 5. Evaluation Protocol

### Classification Metrics (Same as ETASR)

| Metric | Method |
|--------|--------|
| Accuracy | `sklearn.metrics.accuracy_score` |
| Per-class Precision/Recall/F1 | `sklearn.metrics.classification_report` |
| Macro F1 | `sklearn.metrics.f1_score(average='macro')` |
| ROC-AUC | `sklearn.metrics.roc_auc_score` |
| Confusion Matrix | `sklearn.metrics.confusion_matrix` |

### Segmentation Metrics (New for Localization)

| Metric | Method |
|--------|--------|
| Pixel-level F1 | `(2 * TP) / (2 * TP + FP + FN)` over all pixels |
| IoU (Intersection over Union) | `TP / (TP + FP + FN)` |
| Pixel AUC | ROC-AUC on per-pixel probabilities |
| Dice Score | Same as F1 but commonly used in segmentation |

### Comparison Table Format

| Version | Encoder | Input | Frozen? | Test Acc | Macro F1 | AUC | Tam-F1 | Pixel-F1 | IoU |
|---------|---------|-------|---------|----------|----------|-----|--------|----------|-----|
| vR.1.1 | ETASR CNN | ELA | N/A | 88.38% | 0.8805 | 0.9601 | 0.8606 | N/A | N/A |
| vR.P.0 | ResNet-34 | RGB | Yes | — | — | — | — | — | — |

---

## 6. Kaggle Notebook Template

### Cell Structure for vR.P.0

```
Cell 0:  [Markdown] Title, paper reference, change log
Cell 1:  [Markdown] Architecture comparison (ETASR vs pretrained)
Cell 2:  [Code]     Imports, config, GPU check, version info
Cell 3:  [Code]     Install SMP (pip install segmentation-models-pytorch)
Cell 4:  [Markdown] Dataset section header
Cell 5:  [Code]     Dataset discovery, path collection
Cell 6:  [Markdown] Preprocessing section
Cell 7:  [Code]     Image loading + normalization (RGB, resize to 384×384)
Cell 8:  [Code]     ELA visualization (for comparison, not as input)
Cell 9:  [Markdown] Data splitting section
Cell 10: [Code]     70/15/15 stratified split + DataLoader setup
Cell 11: [Markdown] Model architecture section
Cell 12: [Code]     Build model (SMP UNet + ResNet34, freeze encoder)
Cell 13: [Code]     Print model summary, count trainable params
Cell 14: [Markdown] Training section
Cell 15: [Code]     Training loop (with validation, early stopping)
Cell 16: [Markdown] Evaluation section
Cell 17: [Code]     Test set evaluation (per-class, macro, AUC)
Cell 18: [Code]     Confusion matrix
Cell 19: [Code]     ROC curve
Cell 20: [Code]     Training curves
Cell 21: [Code]     Sample predictions
Cell 22: [Markdown] Ablation comparison section
Cell 23: [Code]     Tracking table (ETASR vs pretrained)
Cell 24: [Markdown] Discussion
Cell 25: [Code]     Save model
```

---

## 7. Migration Checklist

Before starting the pretrained track:

- [ ] vR.1.3 (class weights) has been run and results recorded
- [ ] Decision on framework (PyTorch + SMP recommended)
- [ ] Confirm CASIA v2.0 is available on Kaggle as a dataset
- [ ] Verify T4 GPU can handle 384×384 × batch 16 × ResNet-34
- [ ] Decide on classification-first or segmentation-first approach
- [ ] Create `vR.P.0` notebook with the structure above
- [ ] Document the branching decision in the master ablation plan
