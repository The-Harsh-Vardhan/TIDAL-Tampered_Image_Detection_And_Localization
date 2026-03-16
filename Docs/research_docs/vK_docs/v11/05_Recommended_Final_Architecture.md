# Docs11: Recommended Final Architecture

A concrete specification for the recommended final model, loss, training, and evaluation configuration. This serves as the implementation blueprint.

---

## 1. Model Architecture

### 1.1 Overview

```
SMP UNet (ResNet34 encoder, ImageNet pretrained)
+ Custom classification head on bottleneck
+ 4-channel input (RGB + ELA)
+ Edge supervision loss
```

### 1.2 Model Class

```python
import segmentation_models_pytorch as smp

class TamperDetector(nn.Module):
    """
    Dual-head model for tampered image detection and localization.

    Segmentation branch: SMP UNet with pretrained ResNet34 encoder.
    Classification branch: FC head on encoder bottleneck features.
    """
    def __init__(self, config):
        super().__init__()
        self.segmentor = smp.Unet(
            encoder_name=config.get('encoder_name', 'resnet34'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            in_channels=config.get('in_channels', 4),  # RGB + ELA
            classes=config.get('n_classes', 1),
        )

        # Classification head on bottleneck
        encoder_out = self.segmentor.encoder.out_channels[-1]  # 512 for ResNet34
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.get('dropout', 0.5)),
            nn.Linear(256, config.get('n_labels', 2)),
        )

    def forward(self, x):
        features = self.segmentor.encoder(x)
        cls_logits = self.classifier(features[-1])
        decoder_output = self.segmentor.decoder(*features)
        seg_logits = self.segmentor.segmentation_head(decoder_output)
        return cls_logits, seg_logits
```

### 1.3 Architecture Diagram

```
              Input (4 × 256 × 256)
              [RGB + ELA channel]
                      │
            ┌─────────▼─────────┐
            │  ResNet34 Encoder  │ (ImageNet pretrained)
            │                   │
            │  Stage 0: 64ch    │ → skip₀
            │  Stage 1: 64ch    │ → skip₁
            │  Stage 2: 128ch   │ → skip₂
            │  Stage 3: 256ch   │ → skip₃
            │  Stage 4: 512ch   │ → bottleneck
            └──┬────────────┬───┘
               │            │
    ┌──────────┘            └─────────────┐
    │ DECODER                   CLASSIFIER │
    │                                      │
    │ Up(512→256) + skip₃      AdaptiveAvgPool(1×1)
    │ Up(256→128) + skip₂           Flatten
    │ Up(128→64) + skip₁       Linear(512→256)
    │ Up(64→32) + skip₀         ReLU + Dropout
    │                          Linear(256→2)
    │ Conv(32→1, 1×1)               │
    │       │                        ▼
    ▼       ▼              cls_logits (B × 2)
seg_logits (B × 1 × 256 × 256)
```

### 1.4 Parameter Budget

| Component | Parameters | Memory (FP16) |
|---|---|---|
| ResNet34 encoder | 21.3M | ~42 MB |
| SMP U-Net decoder | ~3M | ~6 MB |
| Classification head | ~130K | ~0.3 MB |
| **Total model** | **~24.5M** | **~49 MB** |
| Feature maps (256×256, batch=16) | — | ~800 MB |
| Optimizer state | — | ~200 MB |
| **Total estimated VRAM** | | **~1.1 GB** |

This fits easily on a single T4 (15.6 GB VRAM). On 2×T4 with DataParallel, batch size can be 24-32.

---

## 2. Loss Function

### 2.1 Formulation

```
total_loss = α × FocalLoss(cls_logits, labels)
           + β × (w_bce × BCEWithLogitsLoss(seg_logits, masks)
                 + w_dice × DiceLoss(seg_logits, masks))
           + γ × EdgeLoss(seg_logits, masks)
```

### 2.2 Loss Weights

| Weight | Value | Rationale |
|---|---|---|
| α (classification) | 1.5 | Preserved from vK.10.5 — slightly favors classification for stronger learning signal |
| β (segmentation) | 1.0 | Preserved from vK.10.5 |
| w_bce | 0.5 | Equal balance between per-pixel and region-level supervision |
| w_dice | 0.5 | Directly optimizes the Dice metric |
| γ (edge) | 0.3 | New — boundary supervision inspired by EMT-Net/ME-Net |
| focal_gamma | 2.0 | Standard focal loss downweighting for easy negatives |

### 2.3 Edge Loss Implementation

```python
def edge_loss(pred_logits, gt_masks, threshold=0.5):
    """BCE loss between predicted and ground truth mask edges."""
    pred_prob = torch.sigmoid(pred_logits)
    # Compute Sobel edges
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32,
                           device=pred_prob.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2, 3)
    # GT edges
    gt_edge_x = F.conv2d(gt_masks, sobel_x, padding=1)
    gt_edge_y = F.conv2d(gt_masks, sobel_y, padding=1)
    gt_edges = (gt_edge_x.abs() + gt_edge_y.abs()).clamp(0, 1)
    # Pred edges
    pred_edge_x = F.conv2d(pred_prob, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred_prob, sobel_y, padding=1)
    pred_edges = (pred_edge_x.abs() + pred_edge_y.abs()).clamp(0, 1)
    return F.binary_cross_entropy(pred_edges, gt_edges)
```

---

## 3. Training Configuration

### 3.1 CONFIG Changes from vK.10.5

| Parameter | vK.10.5 | Recommended | Rationale |
|---|---|---|---|
| encoder_name | N/A (custom) | `'resnet34'` | Pretrained backbone |
| encoder_weights | N/A | `'imagenet'` | Transfer learning |
| in_channels | 3 | 4 | RGB + ELA |
| optimizer | Adam | AdamW | Better weight decay handling |
| learning_rate | 1e-4 (global) | encoder=1e-4, decoder=1e-3 | Differential LR |
| edge_loss_weight | N/A | 0.3 | Edge supervision |
| accumulation_steps | N/A | 4 | Gradient accumulation |
| encoder_freeze_epochs | N/A | 2 | Protect pretrained BN stats |
| n_channels | 3 | 4 | Updated for ELA |

### 3.2 Preserved Parameters

| Parameter | Value | Rationale |
|---|---|---|
| image_size | 256 | T4 memory compatible |
| max_epochs | 50 | Proven sufficient (vK.3 converged in 50) |
| scheduler | CosineAnnealing(T_max=50) | Proven effective |
| patience | 10 | Tampered-only Dice early stopping |
| use_amp | True | Faster training on T4 |
| max_grad_norm | 5.0 | Gradient clipping |
| alpha | 1.5 | Classification weight |
| beta | 1.0 | Segmentation weight |
| focal_gamma | 2.0 | Focal loss parameter |
| seg_bce_weight | 0.5 | BCE component weight |
| seg_dice_weight | 0.5 | Dice component weight |
| dropout | 0.5 | Classification head dropout |
| seed | 42 | Reproducibility |

### 3.3 Encoder Freeze Strategy

For the first 2 epochs, freeze all encoder parameters (prevent gradient updates) while the decoder and classification head warm up. This protects pretrained BatchNorm statistics from being destroyed by random decoder gradients.

```python
def freeze_encoder(model):
    base = get_base_model(model)
    for param in base.segmentor.encoder.parameters():
        param.requires_grad = False

def unfreeze_encoder(model):
    base = get_base_model(model)
    for param in base.segmentor.encoder.parameters():
        param.requires_grad = True
```

### 3.4 Differential Learning Rate

```python
base_model = get_base_model(model)
optimizer = torch.optim.AdamW([
    {'params': base_model.segmentor.encoder.parameters(), 'lr': 1e-4},
    {'params': base_model.segmentor.decoder.parameters(), 'lr': 1e-3},
    {'params': base_model.segmentor.segmentation_head.parameters(), 'lr': 1e-3},
    {'params': base_model.classifier.parameters(), 'lr': 1e-3},
], weight_decay=1e-4)
```

---

## 4. Data Pipeline

### 4.1 ELA Computation

```python
def compute_ela(image_bgr, quality=90):
    """Compute Error Level Analysis map.
    Re-saves image as JPEG at given quality, then computes
    absolute difference to reveal compression inconsistencies.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image_bgr, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image_bgr, decoded)
    return cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
```

### 4.2 Dataset Integration

In the dataset's `__getitem__`:
1. Load RGB image
2. Compute ELA map
3. Apply Albumentations transforms to image + ELA + mask together
4. Stack ELA as 4th channel after normalization

ELA normalization: scale to [0, 1] range by dividing by 255, then optionally apply dataset-specific mean/std (compute from training set in a setup cell).

### 4.3 Augmentation Pipeline (Preserved + Extended)

| Transform | Parameters | Status |
|---|---|---|
| A.Resize | 256×256 | Preserved |
| A.HorizontalFlip | p=0.5 | Preserved |
| A.VerticalFlip | p=0.3 | **New** |
| A.RandomBrightnessContrast | limit=0.3, p=0.3 | Preserved |
| A.GaussNoise | var_limit=(10,50), p=0.25 | Preserved |
| A.ImageCompression | QF=50-90, p=0.25 | Preserved |
| A.Affine | translate/scale/rotate, p=0.5 | Preserved |
| A.Normalize | ImageNet mean/std (RGB channels) | Preserved |
| ToTensorV2 | — | Preserved |

---

## 5. Evaluation Suite

The full 12-point evaluation stack to apply to the final trained model:

| # | Evaluation | Implementation |
|---|---|---|
| 1 | Threshold sweep | 0.05-0.80 in 0.05 steps on validation; select best |
| 2 | Test metrics | Dice, IoU, F1, Accuracy, AUC-ROC (all-sample + tampered-only) |
| 3 | Confusion matrix | sklearn + seaborn heatmap |
| 4 | ROC curve + PR curve | sklearn + matplotlib |
| 5 | Forgery-type breakdown | Parse CASIA filenames → splicing vs copy-move metrics |
| 6 | Mask-size stratification | Bucket by region %: <2%, 2-5%, 5-15%, >15% |
| 7 | Robustness testing | 8 degradation conditions with delta from clean |
| 8 | Grad-CAM | Hook encoder layer4, overlay heatmaps |
| 9 | Shortcut learning | Mask randomization + boundary sensitivity |
| 10 | Failure case analysis | 10 worst predictions with metadata |
| 11 | Data leakage verification | Path overlap + pHash check |
| 12 | Artifact inventory | List all saved files with sizes |

---

## 6. Visualization Requirements

### 6.1 Training Plots
- Loss curves (train + val)
- Accuracy curves (train + val)
- Dice curves (tampered-only, train + val)
- Learning rate schedule
- All in a 2×2 subplot grid

### 6.2 Prediction Visualization (4-Panel)
For each sample: Original → Ground Truth Mask → Predicted Mask → Overlay
Include 3 correct authentic, 3 correct tampered, 3 incorrect predictions.

### 6.3 Grad-CAM Visualization
3 authentic + 3 tampered: Original → Grad-CAM heatmap overlay → Predicted mask

### 6.4 ELA Visualization
Show samples: RGB → ELA map → Predicted mask → Overlay. Demonstrates the model sees forensic signal.

### 6.5 Robustness Bar Chart
Grouped bar chart: 8 degradation conditions × tampered-only F1, with clean baseline reference line.

### 6.6 Evaluation Plots
- ROC curve with AUC annotation
- PR curve with AP annotation
- F1 vs threshold curve with optimal threshold marked
- Confusion matrix heatmap

---

## 7. Architecture Justification

### 7.1 Why SMP UNet + ResNet34?

| Factor | Justification |
|---|---|
| **Proven baseline** | v8 achieved AUC=0.817 with this exact encoder |
| **Parameter efficiency** | 24.5M params vs 31M in custom UNet — smaller and stronger |
| **Transfer learning** | ImageNet features provide rich low-level edge/texture detectors that forensic detection builds on |
| **SMP ecosystem** | Drop-in encoder swaps (EfficientNet, ResNeXt) for future experiments |
| **T4 compatible** | ~1.1 GB VRAM — runs comfortably on single T4 with batch_size=16 |

### 7.2 Why Not Larger Encoders?

| Encoder | Params | T4 Feasibility | Rationale for Exclusion |
|---|---|---|---|
| ResNet34 | 21.8M | Excellent | **Selected** |
| ResNet50 | 25.6M | Good | Marginal improvement, 4M more params |
| EfficientNet-B0 | 5.3M | Excellent | Could be tested in ablation (P2) |
| EfficientNet-B4 | 19.3M | Good | Similar to ResNet34, less proven for forensics |
| ConvNeXt-T | 28.6M | Marginal | Newer, less tested |

### 7.3 Why ELA Over SRM/Frequency?

| Preprocessing | Complexity | Evidence | Decision |
|---|---|---|---|
| ELA | Low (JPEG re-save + absdiff) | P1/P7: 96.21% accuracy | **Phase 3 (I2)** |
| SRM noise maps | Medium (filter bank design) | P13: critical for AUC=0.987 | Phase 5 ablation (I14) |
| DCT frequency | Medium (block DCT computation) | P4, P6: used but not standalone | Future work |
| YCbCr chrominance | Low (color space conversion) | P16: 96.52% with Cb/Cr alone | Future work |

ELA is the simplest, best-evidenced, and most assignment-relevant forensic preprocessing. If it proves insufficient, SRM noise maps are the escalation path.
