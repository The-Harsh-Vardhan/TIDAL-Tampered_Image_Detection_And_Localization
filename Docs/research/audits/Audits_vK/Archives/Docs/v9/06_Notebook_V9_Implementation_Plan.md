# 06 — Notebook V9 Implementation Plan

## Purpose

Detailed, ordered blueprint for Notebook v9. Every change is traceable to a Docs9 decision. This document serves as the implementation specification — the notebook developer follows this plan, not a vague description.

---

## v9 Baseline

v9 builds on the v8 notebook. All v8 features are retained:

- CONFIG-driven pipeline with feature flags
- BCE + Dice loss with pos_weight and per-sample Dice
- AdamW with differential LR (encoder=1e-4, decoder=1e-3)
- ReduceLROnPlateau scheduler (patience=3, factor=0.5)
- Expanded augmentation (ColorJitter, ImageCompression, GaussNoise, GaussianBlur)
- AMP (autocast + GradScaler) with gradient accumulation
- Gradient clipping and gradient norm logging
- Tampered-only metrics as primary evaluation
- Expanded threshold sweep (0.05–0.80)
- Mask-size stratification
- W&B experiment tracking

**No v8 feature is removed or regressed.**

---

## Implementation Phases

### Phase 0: Pre-Training Data Validation

#### 0.1 pHash Near-Duplicate Check

**What:** Before any training begins, compute perceptual hashes for all images and check for near-duplicates across train/val/test splits.

**Code sketch:**
```python
import imagehash
from PIL import Image
from collections import defaultdict

def compute_phash(image_path, hash_size=8):
    img = Image.open(image_path)
    return str(imagehash.phash(img, hash_size=hash_size))

# Compute hashes for all images
hashes = {}
for split_name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
    for pair in pairs:
        h = compute_phash(pair['image_path'])
        if h not in hashes:
            hashes[h] = []
        hashes[h].append((split_name, pair['image_path']))

# Find cross-split duplicates
cross_split_dupes = {h: locs for h, locs in hashes.items()
                     if len(set(loc[0] for loc in locs)) > 1}

print(f"Total unique hashes: {len(hashes)}")
print(f"Cross-split near-duplicates: {len(cross_split_dupes)}")
for h, locs in list(cross_split_dupes.items())[:5]:
    print(f"  Hash {h}: {[(s, os.path.basename(p)) for s, p in locs]}")
```

**Dependency:** `imagehash` library. Install with `pip install imagehash`.

**Expected output:** Either "0 cross-split duplicates" (good) or a list of duplicates to address. If duplicates are found, group them into the same split.

**Report format:** "Content-level near-duplicate check via pHash: [N] cross-split duplicates found. [Action taken if any]."

---

### Phase 1: Data Pipeline Changes

#### 1.1 ELA Computation

**What:** Add Error Level Analysis as a 4th input channel. Compute ELA for every image in the dataset.

**Code sketch:**
```python
def compute_ela(image_bgr, quality=90):
    """Compute Error Level Analysis map.

    Re-encodes image as JPEG at specified quality, then computes
    absolute difference. Highlights regions with different compression
    histories (e.g., pasted regions).
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded = cv2.imencode('.jpg', image_bgr, encode_param)
    if not result:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image_bgr, decoded)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    return ela_gray
```

**Integration in TamperingDataset.__getitem__:**
```python
def __getitem__(self, idx):
    # Read image (BGR)
    image_bgr = cv2.imread(self.pairs[idx]['image_path'])

    # Compute ELA before color conversion
    if self.config.get('use_ela', False):
        ela_map = compute_ela(image_bgr, quality=self.config.get('ela_quality', 90))
    else:
        ela_map = None

    # Convert to RGB
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Read mask
    mask = self._load_mask(idx)

    # Apply augmentation (to image + mask; ELA is recomputed or augmented separately)
    if self.transform:
        if ela_map is not None:
            # Augment image and ELA together using additional_targets
            transformed = self.transform(image=image, mask=mask, ela=ela_map)
            image = transformed['image']        # Already normalized + tensor
            mask = transformed['mask']
            ela = transformed['ela']            # Tensor [H, W]
            # Concatenate: [4, H, W]
            image = torch.cat([image, ela.unsqueeze(0).float() / 255.0], dim=0)
        else:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

    return image, mask, label
```

**Augmentation integration:** Use Albumentations' `additional_targets` to apply the same spatial transforms to the ELA map:
```python
train_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'ela': 'image'})
```

**Note on normalization:** ELA channel is normalized to [0, 1] by dividing by 255. RGB channels use ImageNet normalization. The model's first conv layer will learn to scale these appropriately.

**CONFIG changes:**
```python
CONFIG = {
    ...
    'use_ela': True,
    'ela_quality': 90,
    'in_channels': 4,  # 3 RGB + 1 ELA
    ...
}
```

---

#### 1.2 DataLoader Optimization

**What:** Add persistent_workers, pin_memory, and prefetch_factor.

```python
loader_kwargs = {
    'batch_size': CONFIG['batch_size'],
    'num_workers': CONFIG['num_workers'],
    'pin_memory': True,
    'persistent_workers': CONFIG['num_workers'] > 0,
    'prefetch_factor': 2 if CONFIG['num_workers'] > 0 else None,
}

train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
```

---

### Phase 2: Model Architecture Changes

#### 2.1 Dual-Task Model

**What:** Add a learned classification head to the U-Net, producing both segmentation logits and classification logits.

**Code sketch:**
```python
class DualTaskModel(nn.Module):
    """U-Net with dual-task output: segmentation + classification."""

    def __init__(self, config):
        super().__init__()
        self.segmentation_model = smp.Unet(
            encoder_name=config['encoder_name'],
            encoder_weights='imagenet' if config['in_channels'] == 3 else None,
            in_channels=config['in_channels'],
            classes=1,
            activation=None,
        )

        # If using 4 channels with ImageNet pretrained weights:
        if config['in_channels'] == 4 and config.get('pretrained', True):
            # Load 3-channel pretrained, copy weights, add random 4th channel
            pretrained_model = smp.Unet(
                encoder_name=config['encoder_name'],
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
            )
            # Copy encoder weights
            self._copy_encoder_with_extra_channel(pretrained_model)
            del pretrained_model

        # Classification head: Global Average Pool → FC → 1
        encoder_out_channels = self.segmentation_model.encoder.out_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(encoder_out_channels, 1),
        )

    def _copy_encoder_with_extra_channel(self, pretrained_model):
        """Copy pretrained 3-channel weights, initialize 4th channel from mean."""
        pretrained_dict = pretrained_model.encoder.state_dict()
        current_dict = self.segmentation_model.encoder.state_dict()

        for key in pretrained_dict:
            if key in current_dict:
                if pretrained_dict[key].shape == current_dict[key].shape:
                    current_dict[key] = pretrained_dict[key]
                elif len(pretrained_dict[key].shape) == 4 and pretrained_dict[key].shape[1] == 3:
                    # First conv layer: [out, 3, k, k] → [out, 4, k, k]
                    w = pretrained_dict[key]
                    extra = w.mean(dim=1, keepdim=True)  # Average RGB channels for 4th
                    current_dict[key] = torch.cat([w, extra], dim=1)

        self.segmentation_model.encoder.load_state_dict(current_dict)

    def forward(self, x):
        # Get encoder features
        features = self.segmentation_model.encoder(x)

        # Segmentation path
        seg_logits = self.segmentation_model.decoder(*features)
        seg_logits = self.segmentation_model.segmentation_head(seg_logits)

        # Classification path (from deepest encoder feature)
        cls_logits = self.cls_head(features[-1])

        return seg_logits, cls_logits
```

**CONFIG changes:**
```python
CONFIG = {
    ...
    'use_dual_task': True,
    'cls_loss_weight': 0.5,  # λ for classification loss
    ...
}
```

---

#### 2.2 Model for DeepLabV3+ Comparison

**What:** Same dual-task pattern but with DeepLabV3+ backend. Only used for comparison experiment.

```python
# In DualTaskModel.__init__, switch based on config:
if config.get('architecture', 'unet') == 'deeplabv3plus':
    self.segmentation_model = smp.DeepLabV3Plus(
        encoder_name=config['encoder_name'],
        encoder_weights='imagenet' if config['in_channels'] == 3 else None,
        in_channels=config['in_channels'],
        classes=1,
        activation=None,
    )
```

---

### Phase 3: Loss Function Changes

#### 3.1 Multi-Task Loss

**What:** Combine segmentation loss with classification loss.

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, seg_loss, cls_loss_weight=0.5, pos_weight_cls=None):
        super().__init__()
        self.seg_loss = seg_loss  # Existing BCEDiceLoss
        self.cls_loss_weight = cls_loss_weight
        self.cls_bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_cls  # Ratio of authentic to tampered images
        )

    def forward(self, seg_pred, cls_pred, seg_target, cls_target):
        loss_seg = self.seg_loss(seg_pred, seg_target)
        loss_cls = self.cls_bce(cls_pred.squeeze(), cls_target.float())
        total = loss_seg + self.cls_loss_weight * loss_cls
        return total, loss_seg, loss_cls
```

**cls_target computation:** For each image in the batch, the label is 1 if the ground truth mask has any tampered pixels, 0 otherwise.

```python
cls_target = (masks.view(masks.size(0), -1).sum(dim=1) > 0).float()
```

---

#### 3.2 Auxiliary Edge Loss

**What:** Add a boundary-focused loss to improve localization quality at tampered edges.

```python
def compute_edge_mask(mask, kernel_size=3):
    """Extract boundary pixels from binary mask using morphological operations."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [B, 1, H, W]
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    padding = kernel_size // 2
    dilated = F.conv2d(mask.float(), kernel, padding=padding)
    dilated = (dilated > 0).float()
    eroded = F.conv2d(mask.float(), kernel, padding=padding)
    eroded = (eroded >= kernel_size * kernel_size).float()
    edge = dilated - eroded
    return edge.clamp(0, 1)

# In loss computation:
if CONFIG.get('use_edge_loss', False):
    edge_mask = compute_edge_mask(target)
    # Weight the BCE loss higher on edge pixels
    edge_weight = 1.0 + CONFIG.get('edge_loss_weight', 2.0) * edge_mask
    edge_bce = F.binary_cross_entropy_with_logits(
        pred, target, weight=edge_weight, reduction='mean'
    )
    # Replace standard BCE component or add as auxiliary
    total_loss = total_loss + CONFIG.get('edge_loss_lambda', 0.3) * edge_bce
```

**CONFIG changes:**
```python
CONFIG = {
    ...
    'use_edge_loss': True,
    'edge_loss_weight': 2.0,   # Extra weight on boundary pixels
    'edge_loss_lambda': 0.3,   # Weight of edge loss in total
    ...
}
```

---

### Phase 4: Training Loop Changes

#### 4.1 Updated Training Step

**What:** The training loop must handle dual-task output.

```python
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, config, device):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (images, masks, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            seg_logits, cls_logits = model(images)
            total_loss, seg_loss, cls_loss = criterion(
                seg_logits, cls_logits, masks, labels
            )
            total_loss = total_loss / config['accumulation_steps']

        scaler.scale(total_loss).backward()

        if (batch_idx + 1) % config['accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config['max_grad_norm']
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if config.get('use_wandb'):
                wandb.log({'grad_norm': grad_norm.item()})

        running_loss += total_loss.item() * config['accumulation_steps']
        running_seg_loss += seg_loss.item()
        running_cls_loss += cls_loss.item()

    n = len(train_loader)
    return running_loss / n, running_seg_loss / n, running_cls_loss / n
```

#### 4.2 Per-Forgery-Type Loss Tracking

**What:** Log separate loss values for copy-move vs splicing batches during training.

```python
# In train_one_epoch, after computing loss:
if config.get('track_forgery_type_loss') and config.get('use_wandb'):
    # Requires forgery_type labels in the batch
    for ftype in ['splicing', 'copymove']:
        type_mask = (forgery_types == ftype)
        if type_mask.any():
            type_loss = criterion.seg_loss(
                seg_logits[type_mask], masks[type_mask]
            )
            wandb.log({f'train_loss_{ftype}': type_loss.item()})
```

**Note:** This requires passing forgery_type labels through the DataLoader. Modify the dataset to return `(image, mask, label, forgery_type)`.

---

### Phase 5: Evaluation Improvements

#### 5.1 Boundary F1 Metric

```python
def boundary_f1(pred_mask, gt_mask, tolerance=2):
    """Compute Boundary F1 score with pixel tolerance."""
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import binary_dilation

    pred_boundary = find_boundaries(pred_mask, mode='inner')
    gt_boundary = find_boundaries(gt_mask, mode='inner')

    # Dilate boundaries by tolerance
    struct = np.ones((2*tolerance+1, 2*tolerance+1))
    pred_dilated = binary_dilation(pred_boundary, structure=struct)
    gt_dilated = binary_dilation(gt_boundary, structure=struct)

    # Precision: fraction of predicted boundary within tolerance of GT boundary
    if pred_boundary.sum() == 0:
        precision = 1.0 if gt_boundary.sum() == 0 else 0.0
    else:
        precision = (pred_boundary & gt_dilated).sum() / pred_boundary.sum()

    # Recall: fraction of GT boundary within tolerance of predicted boundary
    if gt_boundary.sum() == 0:
        recall = 1.0 if pred_boundary.sum() == 0 else 0.0
    else:
        recall = (gt_boundary & pred_dilated).sum() / gt_boundary.sum()

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

#### 5.2 Precision-Recall Curves

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def plot_pr_curves(all_preds, all_targets, all_cls_scores, all_cls_labels, save_path):
    """Plot pixel-level and image-level PR curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pixel-level PR curve (subsample for efficiency)
    pixel_preds = torch.sigmoid(torch.cat(all_preds)).cpu().numpy().flatten()
    pixel_targets = torch.cat(all_targets).cpu().numpy().flatten()
    # Subsample to avoid memory issues
    n = len(pixel_preds)
    if n > 1_000_000:
        idx = np.random.choice(n, 1_000_000, replace=False)
        pixel_preds = pixel_preds[idx]
        pixel_targets = pixel_targets[idx]
    prec, rec, _ = precision_recall_curve(pixel_targets, pixel_preds)
    axes[0].plot(rec, prec)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Pixel-Level PR Curve')

    # Image-level PR curve
    cls_scores = np.array(all_cls_scores)
    cls_labels = np.array(all_cls_labels)
    prec_img, rec_img, _ = precision_recall_curve(cls_labels, cls_scores)
    axes[1].plot(rec_img, prec_img)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Image-Level PR Curve')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

#### 5.3 Mask Randomization Test

```python
def mask_randomization_test(model, test_dataset, device, config, n_iterations=5):
    """Falsification test: shuffle masks and re-evaluate.

    If the model learned genuine localization, F1 should drop
    dramatically with shuffled masks. If it exploits shortcuts,
    F1 stays similar.
    """
    # Baseline: normal evaluation
    baseline_f1 = evaluate_tampered_only(model, test_dataset, device, config)

    # Shuffled: randomize mask assignments
    shuffled_f1s = []
    for i in range(n_iterations):
        shuffled_dataset = copy.deepcopy(test_dataset)
        mask_paths = [p['mask_path'] for p in shuffled_dataset.pairs]
        random.shuffle(mask_paths)
        for j, pair in enumerate(shuffled_dataset.pairs):
            pair['mask_path'] = mask_paths[j]

        shuffled_f1 = evaluate_tampered_only(model, shuffled_dataset, device, config)
        shuffled_f1s.append(shuffled_f1)

    mean_shuffled = np.mean(shuffled_f1s)
    print(f"Baseline tampered-only F1: {baseline_f1:.4f}")
    print(f"Shuffled tampered-only F1: {mean_shuffled:.4f} ± {np.std(shuffled_f1s):.4f}")
    print(f"Drop: {baseline_f1 - mean_shuffled:.4f}")

    if baseline_f1 - mean_shuffled > 0.15:
        print("✅ Model has learned genuine localization (>0.15 drop with shuffled masks)")
    else:
        print("⚠️ WARNING: Small drop suggests possible shortcut reliance")

    return baseline_f1, mean_shuffled
```

#### 5.4 Image-Level Evaluation (Learned Head)

```python
def evaluate_image_level(model, test_loader, device):
    """Evaluate image-level detection using learned classification head."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            _, cls_logits = model(images)
            scores = torch.sigmoid(cls_logits).squeeze().cpu().numpy()
            all_scores.extend(scores if scores.ndim > 0 else [scores.item()])
            all_labels.extend(labels.numpy())

    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    auc = roc_auc_score(all_labels, all_scores)

    # Find optimal threshold for image-level
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (np.array(all_scores) > t).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    preds = (np.array(all_scores) > best_thresh).astype(int)
    acc = accuracy_score(all_labels, preds)

    print(f"Image-Level Detection (Learned Head):")
    print(f"  AUC-ROC:    {auc:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  F1:         {best_f1:.4f}")
    print(f"  Threshold:  {best_thresh:.2f}")

    return auc, acc, best_f1, best_thresh
```

#### 5.5 ELA Visualization

```python
def visualize_ela(image_path, quality=90):
    """Visualize ELA map alongside the original image."""
    img = cv2.imread(image_path)
    ela = compute_ela(img, quality=quality)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(ela, cmap='hot')
    axes[1].set_title(f'ELA (QF={quality})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
```

---

### Phase 6: Experiments

#### 6.1 Primary Experiment: v9 Full Pipeline (3 seeds)

**Config:** All v9 improvements enabled (ELA, dual-task, edge loss, full augmentation).

**Seeds:** {42, 123, 789}

**Metrics to report:**
- Tampered-only Pixel-F1 (mean ± std)
- Copy-move F1 (mean ± std)
- Splicing F1 (mean ± std)
- Boundary F1 (mean ± std)
- Image-level AUC (learned head) (mean ± std)
- Optimal pixel threshold
- Robustness Δ (JPEG QF50)

#### 6.2 Architecture Comparison: DeepLabV3+

**Config:** Same as 6.1 but with `architecture='deeplabv3plus'`.

**Seeds:** {42} (1 seed — comparison only)

**Purpose:** Determine whether DeepLabV3+ provides meaningful improvement over U-Net.

#### 6.3 Augmentation Ablation

**Config A:** v9 full augmentation
**Config B:** v9 without photometric augmentation (geometric only)

**Seeds:** {42} (1 seed each)

**Purpose:** Confirm that photometric augmentation improves robustness.

---

### Phase 7: Documentation & Delivery

#### 7.1 Notebook Markdown Updates

Update all markdown cells to reflect v9 changes:

- **Design Rationale cell:** Add dual-task architecture justification, ELA channel rationale, edge loss rationale.
- **Dataset description:** Correct CASIA framing ("selected baseline" not "expected dataset").
- **Use-case statement:** "This is a forensic localization research baseline intended for analyst assistance."
- **Architecture cell:** Describe DualTaskModel with learned classification head.
- **Loss cell:** Document multi-task loss with edge-aware component.
- **Evaluation cell:** Describe Boundary F1, PR curves, mask randomization test.
- **Results cell:** Lead with tampered-only metrics. Report copy-move honestly.

#### 7.2 Colab Verification

**Mandatory pre-submission gate:**
1. Open Colab notebook in Google Colab
2. Verify all cells execute without error on a T4 runtime
3. Training completes (at least 5 epochs as a smoke test)
4. All evaluation cells produce output
5. All visualizations render

---

## CONFIG Template for v9

```python
CONFIG = {
    # Experiment
    'experiment_name': 'v9-run',
    'seed': 42,

    # Data
    'image_size': 384,
    'batch_size': 4,
    'num_workers': 2,
    'use_ela': True,
    'ela_quality': 90,

    # Model
    'architecture': 'unet',           # 'unet' or 'deeplabv3plus'
    'encoder_name': 'resnet34',
    'in_channels': 4,                 # 3 (RGB) or 4 (RGB+ELA)
    'pretrained': True,
    'use_dual_task': True,
    'cls_loss_weight': 0.5,

    # Training
    'max_epochs': 50,
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'accumulation_steps': 4,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 10,

    # Scheduler
    'scheduler': 'reduce_on_plateau',
    'scheduler_patience': 3,
    'scheduler_factor': 0.5,
    'min_lr': 1e-6,

    # Loss
    'use_edge_loss': True,
    'edge_loss_weight': 2.0,
    'edge_loss_lambda': 0.3,

    # Augmentation flags
    'aug_color_jitter': True,
    'aug_compression': True,
    'aug_gauss_noise': True,
    'aug_gauss_blur': True,
    'aug_hflip': True,
    'aug_vflip': True,
    'aug_rotate90': True,

    # Evaluation
    'threshold_min': 0.05,
    'threshold_max': 0.80,
    'threshold_step': 0.05,

    # Experiment tracking
    'use_wandb': True,
    'wandb_project': 'tampered-image-detection',

    # Paths (set per environment)
    'data_dir': None,
    'output_dir': None,
}
```

---

## Pre-Flight Checklist

- [ ] `imagehash` library installed
- [ ] pHash duplicate check completed
- [ ] ELA computation verified on sample images
- [ ] DualTaskModel instantiates and produces correct output shapes
- [ ] MultiTaskLoss computes without error
- [ ] Edge loss computes without error
- [ ] Training loop handles (seg_logits, cls_logits) output
- [ ] Validation loop handles dual output
- [ ] Boundary F1 metric verified on sample predictions
- [ ] PR curve plotting verified
- [ ] Image-level evaluation uses learned head (not max(prob_map))
- [ ] W&B run name identifies this as v9
- [ ] CONFIG['in_channels'] = 4 when use_ela=True
- [ ] Augmentation additional_targets includes 'ela'

## Post-Run Validation Checklist

- [ ] Tampered-only F1 > 0.55 (was 0.41 in Run01, expected 0.50-0.60 in v8)
- [ ] Image-level AUC > 0.88 with learned head (was 0.87 with heuristic)
- [ ] Copy-move F1 > 0.38 (was 0.31)
- [ ] Optimal threshold in 0.30-0.55 range
- [ ] Robustness Δ (JPEG QF50) < 0.06
- [ ] Mask randomization test shows >0.15 drop (proves real learning)
- [ ] pHash check completed and documented
- [ ] Multi-seed results reported with mean ± std
- [ ] Colab notebook verified end-to-end
