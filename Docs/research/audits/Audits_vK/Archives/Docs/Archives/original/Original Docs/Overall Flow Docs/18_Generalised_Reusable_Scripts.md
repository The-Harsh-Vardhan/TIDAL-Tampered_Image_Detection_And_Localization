# 18. Generalised Reusable Scripts — Engineering Guide

## 18.1 The Problem

Your current notebook is hardcoded for one specific scenario:
- **One dataset**: CASIA v2.0
- **One architecture**: UNet + EfficientNet-B1
- **One task**: Binary tampering segmentation
- **One environment**: Google Colab

If tomorrow the assignment says "now try DeepLabV3+" or "train on COVERAGE," you'd rewrite half the notebook.

---

## 18.2 The Goal

Write code that's **specific enough to work** but **general enough to reuse** with minimal changes. This isn't about building a framework — it's about clean engineering that separates **what changes** from **what stays the same**.

---

## 18.3 Principle 1: Configuration-Driven Code

### Before (Hardcoded)

```python
# Scattered throughout notebook
model = smp.Unet(
    encoder_name="efficientnet-b1",
    encoder_weights="imagenet",
    in_channels=6,
    classes=1,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
train_loader = DataLoader(dataset, batch_size=4, num_workers=2)
```

### After (Config-Driven)

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Config:
    """Single source of truth for all hyperparameters."""
    # Data
    dataset_name: str = "casia_v2"
    data_dir: str = "./data"
    image_size: int = 512
    
    # Model
    architecture: str = "Unet"          # Unet, DeepLabV3Plus, FPN, etc.
    encoder_name: str = "efficientnet-b1"
    encoder_weights: str = "imagenet"
    in_channels: int = 6
    num_classes: int = 1
    use_srm: bool = True
    
    # Training
    batch_size: int = 4
    accumulation_steps: int = 4
    num_epochs: int = 50
    lr_encoder: float = 1e-4
    lr_decoder: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    
    # Loss
    loss_bce_weight: float = 1.0
    loss_dice_weight: float = 1.0
    loss_edge_weight: float = 0.5
    
    # Environment
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 42
    checkpoint_dir: str = "/content/drive/MyDrive/BigVision/checkpoints"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "bigvision-tampering-detection"
    
    @property
    def effective_batch_size(self):
        return self.batch_size * self.accumulation_steps

# Create config
cfg = Config()

# Everything reads from config
model = create_model(cfg)
optimizer = create_optimizer(model, cfg)
train_loader = create_dataloader(train_dataset, cfg, is_train=True)
```

### Benefits
- Change one value → propagates everywhere
- Configs are serializable (save as JSON with `dataclasses.asdict(cfg)`)
- W&B logs the entire config with `wandb.init(config=dataclasses.asdict(cfg))`
- Different experiments = different configs (not different code)

---

## 18.4 Principle 2: Factory Functions

Create objects through functions that accept config, not inline instantiation.

### Model Factory

```python
def create_model(cfg: Config):
    """
    Create model from config. Swap architectures by changing one string.
    """
    # Choose architecture
    arch_map = {
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus,
        'DeepLabV3Plus': smp.DeepLabV3Plus,
        'FPN': smp.FPN,
        'MAnet': smp.MAnet,
    }
    
    if cfg.architecture not in arch_map:
        raise ValueError(f"Unknown architecture: {cfg.architecture}. "
                        f"Choose from: {list(arch_map.keys())}")
    
    ArchClass = arch_map[cfg.architecture]
    
    if cfg.use_srm:
        # Wrap with SRM preprocessing
        model = TamperingDetector(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            arch_class=ArchClass,
        )
    else:
        # RGB-only model
        model = ArchClass(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=3,
            classes=cfg.num_classes,
            activation=None,
        )
    
    return model.to(cfg.device)
```

### Optimizer Factory

```python
def create_optimizer(model, cfg: Config):
    """Create optimizer with differential learning rates."""
    param_groups = []
    
    if cfg.use_srm and hasattr(model, 'channel_reducer'):
        param_groups.append({
            'params': model.channel_reducer.parameters(),
            'lr': cfg.lr_decoder,
            'name': 'channel_reducer'
        })
    
    seg_model = model.segmentation_model if hasattr(model, 'segmentation_model') else model
    
    param_groups.extend([
        {'params': seg_model.encoder.parameters(), 
         'lr': cfg.lr_encoder, 'name': 'encoder'},
        {'params': seg_model.decoder.parameters(), 
         'lr': cfg.lr_decoder, 'name': 'decoder'},
        {'params': seg_model.segmentation_head.parameters(), 
         'lr': cfg.lr_decoder, 'name': 'seg_head'},
    ])
    
    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
```

### Loss Factory

```python
def create_criterion(cfg: Config):
    """Create loss function from config."""
    return HybridLoss(
        bce_weight=cfg.loss_bce_weight,
        dice_weight=cfg.loss_dice_weight,
        edge_weight=cfg.loss_edge_weight,
    )
```

---

## 18.5 Principle 3: Dataset Abstraction

Make the dataset class work for any image-mask dataset, not just CASIA.

```python
class SegmentationDataset(Dataset):
    """
    Generic binary segmentation dataset.
    Works with any dataset that has image-mask pairs.
    """
    def __init__(self, image_paths, mask_paths, transform=None, 
                 mask_threshold=128):
        """
        Args:
            image_paths: list of paths to images
            mask_paths: list of paths to masks (None for authentic)
            transform: albumentations transform
            mask_threshold: binarisation threshold for non-binary masks
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_threshold = mask_threshold
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        
        # Load mask (zeros if authentic / no mask)
        mask_path = self.mask_paths[idx]
        if mask_path is not None and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > self.mask_threshold).astype(np.float32)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else mask
    
    @property
    def filenames(self):
        return [os.path.basename(p) for p in self.image_paths]
```

### Dataset-Specific Loaders (Adapters)

```python
def load_casia_v2(data_dir):
    """
    CASIA v2.0 specific: discovers images, pairs with masks, returns paths.
    Returns: list of (image_path, mask_path_or_None, label)
    """
    au_dir = os.path.join(data_dir, 'Au')
    tp_dir = os.path.join(data_dir, 'Tp')
    mask_dir = os.path.join(data_dir, 'mask')  # Or 'CASIA 2 Groundtruth'
    
    samples = []
    
    # Authentic images
    for f in os.listdir(au_dir):
        samples.append((os.path.join(au_dir, f), None, 0))
    
    # Tampered images + masks
    for f in os.listdir(tp_dir):
        mask_name = find_matching_mask(f, mask_dir)  # Dataset-specific logic
        mask_path = os.path.join(mask_dir, mask_name) if mask_name else None
        samples.append((os.path.join(tp_dir, f), mask_path, 1))
    
    return samples


def load_coverage(data_dir):
    """
    COVERAGE dataset specific loader.
    """
    image_dir = os.path.join(data_dir, 'image')
    mask_dir = os.path.join(data_dir, 'mask')
    
    samples = []
    for f in sorted(os.listdir(image_dir)):
        mask_name = f.replace('.png', 'forged.png')  # COVERAGE naming
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            samples.append((os.path.join(image_dir, f), mask_path, 1))
    
    return samples
```

Now switching datasets is one line:
```python
# CASIA
samples = load_casia_v2('./data/casia')

# COVERAGE  
samples = load_coverage('./data/coverage')

# Same dataset class, same training pipeline
dataset = SegmentationDataset(
    image_paths=[s[0] for s in samples],
    mask_paths=[s[1] for s in samples],
    transform=get_transforms(cfg, is_train=True)
)
```

---

## 18.6 Principle 4: Transform Pipelines as Functions

```python
def get_transforms(cfg: Config, is_train: bool):
    """Return appropriate transforms for train or val/test."""
    if is_train:
        return A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(5, 25), p=1.0),
            ], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, 
                                       contrast_limit=0.15, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
```

---

## 18.7 Principle 5: Training Loop as a Function

```python
def train_one_epoch(model, loader, criterion, optimizer, scaler, 
                     device, accumulation_steps, epoch=None):
    """
    Generic training loop for one epoch.
    Works with any model, any loss, any optimizer.
    """
    model.train()
    running_loss = 0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}' if epoch is not None else 'Training')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        with autocast('cuda'):
            logits = model(images)
            loss, loss_dict = criterion(logits, masks)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            num_batches += 1
        
        running_loss += loss_dict['total']
        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    return running_loss / max(num_batches * accumulation_steps, 1)
```

---

## 18.8 Principle 6: Metrics as Pluggable Classes

```python
class MetricTracker:
    """
    Track and compute multiple metrics.
    Add new metrics without changing evaluation code.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.pixel_f1s = []
        self.pixel_ious = []
        self.image_scores = []
        self.image_labels = []
    
    def update(self, pred_probs, gt_masks):
        """
        Args:
            pred_probs: (B, 1, H, W) probabilities
            gt_masks: (B, 1, H, W) binary
        """
        for i in range(pred_probs.shape[0]):
            pred = pred_probs[i].squeeze()
            gt = gt_masks[i].squeeze()
            
            f1, _, _ = compute_pixel_f1_safe(pred, gt, self.threshold)
            iou = compute_pixel_iou(pred, gt, self.threshold)
            
            self.pixel_f1s.append(f1)
            self.pixel_ious.append(iou)
            self.image_scores.append(pred.max().item())
            self.image_labels.append(1 if gt.sum() > 0 else 0)
    
    def compute(self):
        return {
            'pixel_f1': np.mean(self.pixel_f1s),
            'pixel_iou': np.mean(self.pixel_ious),
            'image_auc': roc_auc_score(self.image_labels, self.image_scores)
                         if len(set(self.image_labels)) > 1 else 0.0,
        }
```

---

## 18.9 Putting It All Together

```python
# ========== Configuration ==========
cfg = Config(
    architecture="Unet",
    encoder_name="efficientnet-b1",
    use_srm=True,
    num_epochs=50,
    lr_encoder=1e-4,
    lr_decoder=1e-3,
)

# ========== Data ==========
samples = load_casia_v2(cfg.data_dir)
train_samples, val_samples, test_samples = split_samples(samples, cfg.seed)

train_dataset = SegmentationDataset(
    [s[0] for s in train_samples], [s[1] for s in train_samples],
    transform=get_transforms(cfg, is_train=True)
)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                          shuffle=True, num_workers=cfg.num_workers,
                          pin_memory=True, drop_last=True)

# ========== Model + Training ==========
model = create_model(cfg)
optimizer = create_optimizer(model, cfg)
criterion = create_criterion(cfg)

# ========== Train ==========
# ... (same loop, but reading from cfg everywhere)
```

### Switching to a Different Experiment

```python
# Just change config — zero code changes
cfg = Config(
    architecture="DeepLabV3Plus",   # Different architecture
    encoder_name="resnet34",         # Different encoder
    use_srm=False,                   # No SRM
    lr_encoder=5e-5,                 # Different LR
)
# Everything else works identically
```

---

## 18.10 Should You Refactor the Notebook This Way?

| Factor | Assessment |
|--------|-----------|
| **Assignment expectation** | A clean notebook — not necessarily a reusable library |
| **Time cost** | ~1 hour to refactor |
| **Benefits** | Quick ablations (SRM vs. no-SRM), architecture swaps, cleaner code |
| **Evaluator impression** | Strong — shows software engineering discipline |
| **Risk** | Over-engineering might obscure the main logic |

**Verdict: Use Config + factory functions. Skip the rest if time is tight.** The `Config` dataclass + `create_model()` factory pattern takes 15 minutes to set up and makes your notebook dramatically cleaner. The generic dataset class is worth it too. But don't build a full framework — you're writing a notebook, not a library.

### Minimum Viable Reusability
1. ✅ `Config` dataclass (all hyperparameters in one place)
2. ✅ `create_model(cfg)` factory
3. ✅ `get_transforms(cfg, is_train)` function
4. ⚠️ Generic `SegmentationDataset` (nice to have)
5. ❌ Full plugin system (over-engineering for a notebook)
