# Training Strategy

---

## Loss Function

### BCEDiceLoss

```python
class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return bce_loss + dice_loss
```

**Why BCE + Dice?**
- **BCE** provides per-pixel gradients. Every pixel contributes to the loss regardless of class balance. This gives stable, consistent training signal.
- **Dice** directly optimizes the F1-like overlap between prediction and ground truth. For imbalanced masks (tampered pixels are often < 5% of image area), Dice provides stronger gradients for small foreground regions than BCE alone.
- **Combined:** `bce_loss + dice_loss` (equal weight, no tuning). This is a standard combination supported by the reference notebook (`0.5 * BCE + 0.5 * Dice`) and common in medical/forensic segmentation literature.

**Why `smooth=1.0`?** Prevents division by zero when both prediction and ground truth are all-zero (authentic images). Without smoothing, Dice loss would produce `0/0 = NaN`.

**Why `BCEWithLogitsLoss`?** It fuses sigmoid and BCE into a single numerically stable operation. This is more accurate than `sigmoid(logits)` followed by `BCELoss`.

### Alternative Losses (Not Used)

| Loss | Why Not |
|---|---|
| Focal Loss | Designed for classification imbalance, not spatial overlap. Would require tuning γ parameter. |
| Tversky Loss | Generalizes Dice with α/β for FP/FN weighting. Adds two hyperparameters. Marginal benefit on CASIA scale. |
| Lovász-Softmax | Direct IoU optimization. More complex implementation; BCE+Dice already works well. |

These alternatives are documented for future work but are not needed for the current baseline.

---

## Optimizer

### AdamW with Differential Learning Rates

```python
base_model = model.module if is_parallel else model

optimizer = torch.optim.AdamW([
    {'params': base_model.encoder.parameters(), 'lr': config['encoder_lr']},   # 1e-4
    {'params': base_model.decoder.parameters(), 'lr': config['decoder_lr']},   # 1e-3
], weight_decay=config['weight_decay'])  # 1e-4
```

**Why differential LR?** The encoder is pretrained on ImageNet — its weights are already good general feature extractors. A high learning rate would destroy these features. The decoder is randomly initialized and needs faster updates to learn the segmentation task. Ratio of 10× (1e-3 / 1e-4) is a standard practice for fine-tuning pretrained encoders.

**Why AdamW?** AdamW decouples weight decay from the gradient update, which produces more consistent regularization than Adam with L2 penalty. Weight decay of 1e-4 provides mild regularization without aggressive parameter shrinkage.

**DataParallel note:** When `is_parallel` is True, `model.module` accesses the unwrapped model to separate encoder and decoder parameter groups. Without this, `DataParallel`'s wrapper parameters would be a single flat group.

---

## Training Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Image size | 384 × 384 | `CONFIG['image_size']` |
| Batch size | 4 | `CONFIG['batch_size']` |
| Accumulation steps | 4 | `CONFIG['accumulation_steps']` |
| Effective batch size | 16 | batch_size × accumulation_steps |
| Max epochs | 50 | `CONFIG['max_epochs']` |
| Early stopping patience | 10 | `CONFIG['patience']` |
| Gradient clipping | max_norm = 1.0 | `CONFIG['max_grad_norm']` |
| Encoder LR | 1e-4 | `CONFIG['encoder_lr']` |
| Decoder LR | 1e-3 | `CONFIG['decoder_lr']` |
| Weight decay | 1e-4 | `CONFIG['weight_decay']` |

---

## Automatic Mixed Precision (AMP)

AMP uses float16 for forward/backward passes (faster, less VRAM) and float32 for parameter updates (numerical accuracy). Controlled via CONFIG flag:

```python
scaler = GradScaler('cuda', enabled=CONFIG['use_amp'])
```

When `use_amp=False`, the scaler's `scale()`, `unscale_()`, `step()`, and `update()` methods become no-ops. This means the same training loop code works regardless of the AMP flag — no conditionals needed.

**AMP in inference paths:** Every function that runs model forward passes uses `autocast('cuda', enabled=config['use_amp'])`:
- `train_one_epoch()`
- `validate_model()`
- `find_best_threshold()`
- `evaluate()`
- `collect_predictions()`
- `run_robustness_eval()`

**Why flag-controlled?** Some older GPUs or CPU-only environments do not support AMP. Setting `use_amp=False` ensures the notebook still runs. On a T4, AMP typically gives ~1.5× speedup with negligible accuracy loss.

---

## Training Loop — `train_one_epoch()`

```python
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config):
    """Run one training epoch with gradient accumulation and optional AMP."""
    model.train()
    running_loss = 0.0
    accum_steps = config['accumulation_steps']
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, masks, labels) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda', enabled=config['use_amp']):
            logits = model(images)
            loss = criterion(logits, masks) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps

    # Partial window flush
    if (batch_idx + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / len(train_loader)
```

**Key design decisions:**

- **Gradient accumulation** (`accum_steps=4`): Simulates effective batch size of 16 using actual batch size of 4. This fits within T4 VRAM while providing sufficient gradient averaging.
- **`optimizer.zero_grad(set_to_none=True)`**: Sets gradients to None instead of zeroing them. Saves memory and can be slightly faster.
- **Partial window flush**: If the dataset length is not divisible by `accum_steps`, the final accumulated gradients are still applied. Without this, the last few batches' gradients would be silently lost.
- **Gradient clipping** (`max_norm=1.0`): Prevents gradient explosion, especially helpful with mixed precision where scale factors can amplify gradients.

---

## Validation — `validate_model()`

```python
@torch.no_grad()
def validate_model(model, val_loader, criterion, device, config, threshold=0.5):
    """Run validation with optional AMP and return loss, mean F1, mean IoU."""
    model.eval()
    total_loss = 0.0
    f1_scores, iou_scores = [], []

    for images, masks, labels in val_loader:
        images, masks = images.to(device), masks.to(device)

        with autocast('cuda', enabled=config['use_amp']):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        for i in range(images.size(0)):
            f1_scores.append(compute_pixel_f1(preds[i], masks[i]))
            iou_scores.append(compute_iou(preds[i], masks[i]))

    return total_loss / len(val_loader.dataset), np.mean(f1_scores), np.mean(iou_scores)
```

**Threshold-aware validation:** During training, `validate_model()` uses a fixed threshold (0.5) for early stopping decisions. The final threshold is selected via a dedicated sweep after training completes.

---

## Early Stopping

Early stopping monitors validation Pixel-F1 with patience of 10 epochs:

- If val Pixel-F1 improves → save checkpoint as `best_model.pt`, reset patience counter
- If no improvement for 10 consecutive epochs → stop training
- `last_checkpoint.pt` is saved every epoch for resume capability

**Checkpoint resume:** The notebook checks for `last_checkpoint.pt` at startup and resumes from the exact training state (model weights, optimizer state, scaler state, epoch counter, best F1).

---

## Checkpointing

| Checkpoint | When Saved | Purpose |
|---|---|---|
| `best_model.pt` | When val F1 improves | Best model for evaluation |
| `last_checkpoint.pt` | Every epoch | Resume interrupted training |
| `checkpoint_epoch_N.pt` | Every 10 epochs | Periodic snapshots |

All checkpoints save consistent state: `model.state_dict()` (unwrapped if DataParallel), `optimizer.state_dict()`, `scaler.state_dict()`, `epoch`, `best_f1`.

---

## Data Augmentation

### Training Transforms

```python
train_transform = A.Compose([
    A.Resize(384, 384),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

### Validation / Test Transforms

```python
val_transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

**Included augmentations:** Horizontal flip, vertical flip, 90° rotation. These are spatial transforms that apply identically to both image and mask.

**Excluded augmentations:** Color jitter, elastic deformation, cutout/gridmask, MixUp/CutMix. These can destroy or create forensic artifacts that the model should learn to detect. Adding photometric augmentations risks teaching the model to ignore compression inconsistencies — the very signal it needs to detect tampering.

---

## Config-Driven DataLoaders

```python
_nw = CONFIG['num_workers']
loader_kwargs = dict(
    num_workers=_nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=_nw > 0,
)
```

- **`pin_memory`**: Automatically enabled when GPU is available; speeds up host-to-device transfers.
- **`persistent_workers`**: Keeps worker processes alive between epochs; reduces worker startup overhead.
- **`drop_last=True`** for train loader only: Prevents a small final batch from disrupting gradient accumulation.
- **Seeded workers**: `seed_worker()` + `torch.Generator` ensure deterministic data loading order.

---

## Interview: "Why not use a learning rate scheduler?"

The current v6.5 notebook uses a constant learning rate with early stopping. A scheduler (e.g., CosineAnnealingWarmRestarts) could improve convergence by reducing LR toward the end of training. This is listed as a Phase 2 optimization. For the baseline, the combination of differential LR + early stopping + gradient accumulation provides stable training without scheduler complexity.

## Interview: "Why gradient accumulation instead of larger batch size?"

A batch of 16 images at 384×384 with a ResNet34 U-Net would require ~12 GB VRAM — close to the T4's 15 GB limit, leaving little headroom for activations during backpropagation. Gradient accumulation achieves the same effective batch size (16) while keeping peak VRAM usage at ~4 GB (batch of 4). The trade-off is slightly slower per-epoch training due to more optimizer steps.
