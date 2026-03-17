# Docs11: Current System Overview

This document is part of Docs11, a technical review and improvement plan for the tampered image detection project. It provides a comprehensive snapshot of vK.10.5 — the current best notebook — as the baseline for all proposed improvements.

Docs11 synthesizes findings from 5+ audit cycles (Audit v7.1, Audit vK.3 run 01, v08_kaggle_run_01, Audit10-Improvements), 21 research paper analyses, and run results from v8, v9, vK.3, and vK.7.5.

---

## 1. Architecture Specification

The model is a custom `UNetWithClassifier` with a shared encoder and two output heads.

| Component | Specification |
|---|---|
| Model class | `UNetWithClassifier` (custom PyTorch, not SMP) |
| Encoder | DoubleConv blocks: 3 → 64 → 128 → 256 → 512 → 1024 |
| Downsampling | MaxPool2d(2) at each encoder stage |
| Decoder | TransposedConv2d upsampling + skip concatenation + DoubleConv |
| Decoder path | 1024 → 512 → 256 → 128 → 64 |
| Segmentation head | Conv2d(64, 1, kernel_size=1) |
| Classification head | AdaptiveAvgPool2d(1,1) → Flatten → Linear(1024,512) → ReLU → Dropout(0.5) → Linear(512,2) |
| Total parameters | ~31M |
| Pretrained weights | **None** (trains from scratch) |
| Input | 256×256, 3 channels (RGB only) |
| Multi-GPU | nn.DataParallel (2×T4 on Kaggle) |

### DoubleConv Block
Each encoder and decoder stage uses two sequential Conv2d(3×3, padding=1, bias=False) → BatchNorm2d → ReLU(inplace=True) layers.

### Skip Connections
Each decoder `Up` stage concatenates upsampled decoder features with the corresponding encoder features after explicit padding to handle shape mismatches.

---

## 2. Training Configuration

All hyperparameters are centralized in a CONFIG dictionary.

| Parameter | Value | Notes |
|---|---|---|
| image_size | 256 | Square resize, no aspect ratio preservation |
| batch_size | 8 (default) | Auto-adjusted: 16 (≥15GB), 24 (≥20GB), 32 (≥28GB) |
| learning_rate | 1e-4 | Single LR for entire model |
| weight_decay | 1e-4 | Applied to all parameters |
| optimizer | Adam | Standard Adam, not AdamW |
| max_epochs | 50 | Validated by vK.3 (converged in 50 epochs) |
| scheduler | CosineAnnealingLR | T_max=50 (single cosine decay) |
| patience | 10 | Early stopping on tampered-only Dice |
| seg_threshold | 0.5 | **Fixed** — no threshold sweep |
| use_amp | True | autocast + GradScaler |
| max_grad_norm | 5.0 | Gradient clipping |

### Loss Function

```
total_loss = α × FocalLoss(cls_logits, labels) + β × (w_bce × BCE(seg_logits, masks) + w_dice × DiceLoss(seg_logits, masks))
```

| Weight | Value |
|---|---|
| α (classification) | 1.5 |
| β (segmentation) | 1.0 |
| focal_gamma | 2.0 |
| seg_bce_weight | 0.5 |
| seg_dice_weight | 0.5 |

---

## 3. Data Pipeline

### Dataset
- **Source:** CASIA v2.0 from Kaggle (`harshv777/casia2-0-upgraded-dataset`)
- **Content:** Authentic images + tampered images + ground truth masks
- **Split:** 70% train / 15% validation / 15% test, stratified by label

### Augmentation (Training)

| Transform | Parameters |
|---|---|
| A.Resize | 256×256 |
| A.HorizontalFlip | p=0.5 |
| A.RandomBrightnessContrast | limit=0.3, p=0.3 |
| A.GaussNoise | var_limit=(10, 50), p=0.25 |
| A.ImageCompression | quality_lower=50, quality_upper=90, p=0.25 |
| A.Affine | translate, scale, rotate, p=0.5 |
| A.Normalize | ImageNet mean/std |
| ToTensorV2 | — |

### Augmentation (Val/Test)
- A.Resize(256×256) + A.Normalize(ImageNet) + ToTensorV2

### Not Implemented
- No ELA preprocessing
- No SRM noise maps
- No frequency-domain features
- No chrominance (YCbCr) channels
- No CLAHE contrast enhancement
- No aspect-ratio-preserving resize

---

## 4. Engineering Features

vK.10.5 has the strongest engineering foundation across all notebook versions:

| Feature | Status |
|---|---|
| Centralized CONFIG dictionary | Implemented |
| Multi-GPU DataParallel (2×T4) | Implemented |
| Three-file checkpoint system (last, best, periodic) | Implemented |
| Checkpoint resume with full history | Implemented |
| Early stopping on tampered-only Dice | Implemented |
| AMP (autocast + GradScaler) | Implemented |
| VRAM-based batch size auto-adjustment | Implemented |
| Metadata caching (skip redundant dataset scanning) | Implemented |
| Seeded reproducibility (Python, NumPy, PyTorch CPU/CUDA) | Implemented |
| Seeded DataLoader workers | Implemented |
| Persistent workers in DataLoaders | Implemented |
| W&B integration with graceful offline fallback | Implemented |
| Kaggle/Colab environment auto-detection | Implemented |
| `get_base_model()` unwrapper for DataParallel | Implemented |

---

## 5. Evaluation Capabilities

### Implemented in vK.10.5

| Capability | Details |
|---|---|
| Dice coefficient | All-sample and tampered-only |
| IoU | All-sample and tampered-only |
| F1 (pixel-level) | All-sample and tampered-only |
| Image-level accuracy | Classification head accuracy |
| ROC-AUC | Image-level (code exists, needs running) |
| Training curves | Loss, accuracy, dice, LR per epoch |
| 4-panel visualization | Original, GT mask, predicted mask, overlay |
| Inference function | Single-image prediction with visualization |
| Training history CSV | Exported every epoch for crash resilience |

### Missing from vK.10.5

| Capability | Impact | Available in v8? |
|---|---|---|
| Threshold sweep/optimization | HIGH — free metric improvement | Yes |
| Robustness testing (JPEG, noise, blur, resize) | HIGH — bonus B1 | Yes (8 conditions) |
| Grad-CAM explainability | HIGH — evaluation rigor | Yes |
| Forgery-type breakdown (splice vs copy-move) | MEDIUM — bonus B2 insight | Yes |
| Mask-size stratified evaluation | MEDIUM — reveals tiny-mask weakness | Yes |
| Shortcut learning validation | MEDIUM — scientific rigor | Yes |
| Failure case analysis (worst N predictions) | MEDIUM — self-awareness | Yes |
| Confusion matrix | LOW — standard deliverable | No |
| ROC curve / PR curve plots | LOW — standard visual | No |
| Data leakage verification | MEDIUM — credibility | Yes |
| Artifact inventory | LOW — professionalism | Yes |
| Pixel-level AUC-ROC | MEDIUM — threshold-independent | No |

---

## 6. Cross-Notebook Metrics Comparison

Results from the best run of each notebook (source: Audit10-Improvements/04_Head_to_Head_Comparison.md):

| Metric | v8 (ResNet34 SMP) | vK.3 (Custom UNet) | vK.7.5 (Custom UNet) | vK.10.5 |
|---|---|---|---|---|
| Image Accuracy | 0.719 | **0.899** | 0.553 | Not run |
| AUC-ROC | **0.817** | N/A | N/A | Not run |
| Tampered F1 | 0.295 | N/A | N/A | Not run |
| Dice (all) | N/A | 0.576 | 0.594* | Not run |
| IoU (all) | 0.493 | 0.553 | 0.594* | Not run |
| Epochs trained | 27 (early stopped) | 50 | 2 (!) | 0 |
| Encoder | ResNet34 (pretrained) | Custom (scratch) | Custom (scratch) | Custom (scratch) |

\* vK.7.5 metrics are degenerate (Dice=IoU=F1=0.5935 is mathematically impossible unless predicting all-zeros).

### Audit Scores

| Notebook | Score | Primary Strength | Primary Weakness |
|---|---|---|---|
| v8 Run | 82/100 | Best evaluation methodology | Mediocre localization (tampered F1=0.29) |
| vK.10.5 (code) | 75/100 | Best engineering foundation | Never run, no advanced evaluation |
| vK.3 Run | 65/100 | Best classification (Acc=0.899) | No engineering refinements |
| vK.7.5 Run | 30/100 | Best documentation | Untrained (2 epochs) |

---

## 7. Key Insight

No single notebook achieves what a competitive submission requires. The ideal notebook combines:

1. **vK.10.5's engineering** — CONFIG dict, checkpoint resume, AMP, early stopping, DataParallel, tampered-only metrics
2. **v8's evaluation methodology** — robustness testing, Grad-CAM, threshold sweep, forgery breakdown, mask-size stratification, shortcut checks, failure analysis
3. **v8's pretrained encoder** — ImageNet-pretrained ResNet34 for stronger features
4. **vK.3's proven convergence** — 50 epochs of actual training with real metrics
5. **Research-backed preprocessing** — ELA as 4th channel, edge supervision loss

The remaining Docs11 files detail exactly how to build this combined notebook.
