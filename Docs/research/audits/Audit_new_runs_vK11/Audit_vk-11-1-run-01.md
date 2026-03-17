# Technical Audit: vK.11.1 (Run 01) -- UNEXECUTED

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vk-11-1-tampered-image-detection-and-localization-run-01.ipynb` |
| **Platform** | Kaggle, 2x Tesla T4 GPUs (31.3 GB VRAM) |
| **Cells** | 102 total (49 code, 53 markdown) |
| **Executed** | 18 of 49 code cells (cells 1-18 only, up to model instantiation) |
| **Status** | **CODE ONLY -- No training, evaluation, or visualization executed** |

---

## 1. Notebook Overview

vK.11.1 is the **base notebook** of the vK.11.x synthesis series. It represents the culmination of all architectural recommendations from prior audits: combining the pretrained ResNet34 encoder (from v6.5), per-sample Dice loss (from v8), comprehensive evaluation suite (from vK.10.6), and three new components -- ELA 4th input channel, Sobel-based edge loss, and an FC classification head.

**Execution stopped after cell 38** (model instantiation). Everything from W&B setup (cell 40) through conclusion (cell 101) has `execution_count=None`. This means:
- The model was defined and shape-checked but never trained
- All loss functions, training loops, and evaluation code exist only as untested source
- No metrics, no visualizations, no results of any kind

### CONFIG Snapshot

```python
CONFIG = {
    'img_size': 256,
    'batch_size': 8,         # auto-scaled to 32 for 2xT4
    'max_epochs': 100,       # NOTE: reduced to 50 in vK.11.4/11.5
    'patience': 20,          # NOTE: reduced to 10 in vK.11.4/11.5
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'accumulation_steps': 4, # effective batch = 128
    'encoder_freeze_epochs': 2,
    'max_grad_norm': 5.0,
    'seed': 42,
}
```

### Sections Present

1-18: Environment, Configuration, Data Pipeline, Model Architecture, Training Loop, Evaluation (11 subsections including threshold sweep, pixel-AUC, confusion matrix, forgery-type breakdown, mask-size stratification, shortcut detection, failure analysis, Grad-CAM, robustness testing, ELA visualization), Inference Demo, Model Card, Conclusion.

**Sections NOT present** (added in later versions): Executive Summary, Results Dashboard, Reproducibility Verification.

---

## 2. Dataset Pipeline Review

### Dataset Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CASIA v2.0 Upgraded (Kaggle) |
| Total Images | 12,614 (7,491 authentic + 5,123 tampered) |
| Split Ratios | 70 / 15 / 15 (stratified, seed=42) |
| Train | 8,829 (5,243 auth + 3,586 tamp, 40.6% tampered) |
| Validation | 1,892 (1,124 auth + 768 tamp, 40.6% tampered) |
| Test | 1,893 (1,124 auth + 769 tamp, 40.6% tampered) |
| Image Size | 256x256 |
| Input Channels | 4 (RGB + ELA grayscale) |

### Data Leakage Check

The data leakage verification cell (cell 23) **was executed and PASSED** -- zero overlap between train/val/test splits. This is a significant improvement over vK.1-vK.7.1 which had Block 1 data leakage (training on the test set).

### ELA Processing

```python
def compute_ela(image_bgr, quality=90):
    _, encoded = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image_bgr, decoded)
    return cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
```

ELA is computed on the BGR image BEFORE color conversion to RGB -- correct order. The ELA channel is stacked into 3 channels for Albumentations compatibility, then only channel 0 is extracted and concatenated with RGB to form the 4-channel input tensor.

**Potential issue**: CASIA images stored as PNG/TIFF would produce minimal JPEG re-compression artifacts, potentially making the ELA channel uninformative. This was **not validated** since training never ran.

### Augmentations (Training Only)

| Transform | Parameters |
|-----------|------------|
| Resize | 256x256 |
| HorizontalFlip | p=0.5 |
| VerticalFlip | p=0.3 |
| RandomBrightnessContrast | p=0.3 |
| GaussNoise | p=0.25 |
| ImageCompression | quality_range=(50,90), p=0.25 |
| Affine | translate=2%, scale=(0.9,1.1), rotate=(-10,10), p=0.5 |
| Normalize | ImageNet mean/std |
| ToTensorV2 | -- |

Sound augmentation strategy. Includes JPEG compression augmentation (addresses v6.5's JPEG robustness gap) and geometric augmentations with border reflection.

### Verdict

**Dataset pipeline is well-designed.** Fixes the data leakage from vK.1-vK.7.1, adds ELA channel, includes comprehensive augmentations. However, none of this was tested in execution.

---

## 3. Model Architecture Review

### TamperDetector Architecture

```
Input (B, 4, 256, 256)  ──  RGB (3ch) + ELA (1ch)
        │
        ▼
┌─────────────────────────────────────┐
│  SMP UNet Encoder (ResNet34)        │
│  Pretrained on ImageNet             │
│  Modified: first conv accepts 4ch   │
│  Output: multi-scale features       │
└──────────┬──────────┬───────────────┘
           │          │
     ┌─────▼─────┐   └──────────┐
     │ UNet       │              │
     │ Decoder    │     GAP → FC │
     │ (SMP)      │     512→256  │
     │            │     ReLU     │
     │ Output:    │     Drop(0.5)│
     │ (B,1,H,W)  │     256→2   │
     └─────┬──────┘     │       │
           │            │       │
    Seg Logits    Cls Logits    │
    (B,1,256,256) (B,2)        │
```

**Parameters**: ~24,571,347 (all trainable)

**Classification Head**:
```python
self.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 2),  # authentic vs tampered
)
```

### Architecture Assessment

**Strengths:**
- Pretrained ResNet34 encoder -- proven in v6.5 to be critical for CASIA scale
- SMP's UNet decoder -- well-tested implementation
- Dual-head design addresses both detection and localization
- 4-channel input (RGB+ELA) provides additional forensic signal

**Concerns:**
- **No attention mechanisms** -- no channel attention, no spatial attention, no ASPP. The decoder relies entirely on skip connections, which may not capture subtle tampering artifacts
- **Classification head shares encoder** -- the FC head's gradient signal (weighted 1.5x in the loss) may dominate encoder updates at the expense of segmentation-relevant features
- **4-channel adaptation**: How SMP handles the 4th input channel matters. If the pretrained first conv weights are copied for channel 4 (averaging RGB weights), this may not be optimal for ELA which has fundamentally different statistics

### Verdict

The architecture is **conceptually sound** and represents the correct synthesis of prior recommendations. However, the dual-head design with heavy classification loss weighting (1.5x) is a theoretical risk for segmentation performance.

---

## 4. Training Pipeline Review

**NOTE**: All training code is **unexecuted**. This is a code review only.

### Loss Functions

```python
# Classification: FocalLoss with balanced class weights
cls_weights = tensor([0.842, 1.231])  # inverse frequency weighting
focal_loss = FocalLoss(alpha=cls_weights, gamma=2.0)

# Segmentation: 0.5*BCE + 0.5*Dice (per-sample)
seg_loss = 0.5 * F.binary_cross_entropy_with_logits(pred, target) + 0.5 * dice_loss(pred, target)

# Edge: Sobel-based boundary supervision
edge_loss = F.binary_cross_entropy(pred_edge, gt_edge)

# Total loss
loss = 1.5 * cls_loss + 1.0 * seg_loss + 0.3 * edge_loss
```

### BUG: Edge Loss AMP Incompatibility

```python
# vK.11.1 code (BUGGY):
def edge_loss(pred_mask, gt_mask):
    pred_edge = sobel_edges(torch.sigmoid(pred_mask))
    gt_edge = sobel_edges(gt_mask)
    return F.binary_cross_entropy(pred_edge, gt_edge)  # <-- NO .float() cast
```

`F.binary_cross_entropy` does **not support float16 inputs**. Under AMP's `autocast`, the Sobel output tensors may be float16, causing either a runtime crash or silently incorrect gradients. This bug was **fixed in vK.11.4/11.5** by adding:

```python
with torch.amp.autocast('cuda', enabled=False):
    return F.binary_cross_entropy(pred_edge.float(), gt_edge.float())
```

**This bug alone may have been the reason vK.11.1 was never successfully executed past the setup cells.**

### Training Configuration

| Feature | Implementation | Assessment |
|---------|---------------|------------|
| Optimizer | AdamW (differential LR) | Good -- enc=1e-4, dec=1e-3 |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) | Good -- improvement over CosineAnnealing |
| Gradient Accumulation | 4 steps (effective batch=128) | Good |
| AMP | torch.amp.autocast + GradScaler | Good -- but edge loss bug breaks it |
| Encoder Freeze | First 2 epochs | Good for pretrained encoder |
| Early Stopping | patience=20, monitoring val tampered Dice | Good -- 20 is reasonable |
| Gradient Clipping | max_norm=5.0 | Good |
| Checkpoints | best_model.pt + last_checkpoint.pt + periodic (every 10 epochs) | Good |
| DataParallel | 2x T4 GPUs | Adequate |

### Verdict

Training pipeline is **well-engineered in design** but **fatally broken by the edge loss AMP bug**. All other components (optimizer, scheduler, accumulation, freezing, checkpointing) are sound. The CONFIG values (`max_epochs=100`, `patience=20`) are the most generous in the vK.11.x series -- ironically, they were reduced in the versions that actually ran.

---

## 5. Evaluation Metrics Review

All evaluation code is defined but **unexecuted**. The evaluation suite includes:

| Feature | Present | Executed |
|---------|---------|----------|
| Dice coefficient (per-sample, tampered-only) | Yes | No |
| IoU coefficient | Yes | No |
| F1 score (pixel-level) | Yes | No |
| Classification AUC-ROC | Yes | No |
| Threshold sweep (0.05-0.80, 50 points) | Yes | No |
| Pixel-level AUC-ROC | Yes | No |
| Confusion matrix + ROC/PR curves | Yes | No |
| Per-forgery-type evaluation | Yes | No |
| Mask-size stratified evaluation | Yes | No |
| Shortcut learning detection | Yes | No |
| Failure case analysis | Yes | No |
| Grad-CAM heatmaps | Yes | No |
| Robustness testing (5 conditions) | Yes | No |
| ELA visualization | Yes | No |

This is the comprehensive 12+ feature evaluation suite inherited from vK.10.6. The metric computation functions use standard formulations:
- Dice: `2*TP / (2*TP + FP + FN)` with per-sample averaging
- IoU: `TP / (TP + FP + FN)` with per-sample averaging
- Tampered-only metrics correctly filter by label=1

### Verdict

**Evaluation design is excellent** -- the most comprehensive in the project. But without execution, no assessment of correctness or results is possible.

---

## 6. Visualization Quality

All visualization cells are defined but produce **no output**:

| Visualization | Status |
|--------------|--------|
| Data samples with masks | Defined, not rendered |
| Training curves | Defined, not rendered |
| Sample predictions | Defined, not rendered |
| Confusion matrix | Defined, not rendered |
| ROC / PR curves | Defined, not rendered |
| Failure cases | Defined, not rendered |
| Grad-CAM heatmaps | Defined, not rendered |
| Robustness bar chart | Defined, not rendered |
| ELA visualization | Defined, not rendered |

### Verdict

Cannot assess visualization quality. The code defines comprehensive visualization functions, but none have been rendered.

---

## 7. Assignment Alignment Check

| Requirement | Status | Notes |
|-------------|--------|-------|
| **1. Dataset Selection** | Code present | CASIA v2.0, pipeline defined |
| **1. Data Pipeline** | Code present | Cleaning, preprocessing, split, mask alignment |
| **1. Augmentation** | Code present | 7 augmentation transforms defined |
| **2. Architecture** | Code present | TamperDetector dual-head UNet, documented |
| **2. Resource Constraints** | Code present | Designed for 2xT4, AMP enabled |
| **3. Performance Metrics** | Code present | 12+ evaluation features defined |
| **3. Visual Results** | Code present | Original/GT/Predicted/Overlay defined |
| **4. Single Notebook** | Yes | All in one .ipynb |
| **4. Documentation** | Partial | Architecture described, training strategy documented |
| **4. Trained Weights** | **NO** | Model never trained |
| **Bonus: Robustness** | Code present | JPEG, noise, blur, resize tests defined |
| **Bonus: Subtle Tampering** | Code present | Copy-move vs splicing breakdown defined |

### Verdict

**Structurally compliant, functionally non-compliant.** The notebook contains all required components as code, but the assignment expects executed results, trained weights, and visual outputs -- none of which exist.

---

## 8. Engineering Quality

### Strengths

| Aspect | Assessment |
|--------|-----------|
| CONFIG centralization | Excellent -- single dict controls all parameters |
| Reproducibility | Seeds set for torch, numpy, random, CUDA, cuDNN |
| Code organization | 18 numbered sections with clear markdown headers |
| Checkpoint system | 3-file strategy (best/last/periodic) |
| Data leakage prevention | Explicit verification with cross-set overlap check |
| Model Card | Present with architecture, training, and evaluation details |
| Loss design | Multi-component with documented weights |

### Weaknesses

| Aspect | Issue |
|--------|-------|
| **Edge loss AMP bug** | `F.binary_cross_entropy` with float16 inputs -- fatal |
| Section numbering | Inconsistent (jumps from 2.2 to 4.4, 2.1.x appears under section 4) |
| No Executive Summary | Added in vK.11.4 |
| No Reproducibility Verification | Added in vK.11.4 |
| No Results Dashboard | Added in vK.11.5 |
| `train_dice` always 0.0 | `history['train_dice'].append(0.0)` with comment "computed at eval time if needed" -- never computed |

### Verdict

**Engineering quality is high for code design but the AMP bug is a showstopper.** The CONFIG system, checkpoint strategy, and evaluation suite design are the best in the project. The AMP bug likely prevented this notebook from completing training.

---

## 9. Roast Section

**"The Blueprint That Never Built"**

vK.11.1 is the project's most ambitious notebook. It synthesizes every recommendation from 10+ prior audits into a single architecture: pretrained encoder (v6.5's gift), per-sample Dice (v8's one good idea), comprehensive evaluation (vK.10.6's evaluation suite), ELA channel (Docs v9 recommendation), edge loss (novel addition), and a classification head (dual-task learning). On paper, this is a 0.50+ Tam-F1 architecture.

In practice, it has the same fundamental problem as vK.1, vK.2, and vK.3: **nobody ran it.**

Eighteen cells executed. Eighteen. Out of forty-nine. The model was instantiated, shape-checked against a dummy input, and then... nothing. The training loop that took weeks to design sits idle. The 12-feature evaluation suite that would have been the most rigorous in project history never processed a single prediction. The robustness testing that addresses every prior audit's #1 complaint is pure dead code.

And the reason? An `F.binary_cross_entropy` call without a `.float()` cast in the edge loss function. Under AMP, this is a guaranteed crash on the first backward pass. One line of code killed the entire synthesis.

The CONFIG is the only version in the vK.11.x series with `max_epochs=100` and `patience=20` -- the generous training budget that every prior audit recommended. When the bug was fixed in vK.11.4, these values were inexplicably halved to 50/10. So the one version with enough training budget has a fatal bug, and the versions without the bug have insufficient training budget.

vK.11.1 is the project's best architectural design document -- presented as a notebook that was supposed to run.

**Score: 3/10** (code quality is high, but an unexecuted notebook is not an experiment)

---

## Summary

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Dataset Pipeline | **B+** | Well-designed, ELA integration, leakage-free |
| Model Architecture | **B+** | Correct synthesis, dual-head, pretrained encoder |
| Training Pipeline | **D** | Sound design, fatal AMP bug in edge loss |
| Evaluation Metrics | **N/A** | Unexecuted |
| Visualization | **N/A** | Unexecuted |
| Assignment Alignment | **F** | Code present, nothing executed |
| Engineering Quality | **B** | Strong CONFIG/checkpoints, section numbering issues |
| **Overall** | **3/10** | The blueprint that never built |
