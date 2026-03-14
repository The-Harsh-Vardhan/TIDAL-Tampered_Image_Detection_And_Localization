# Training Strategy

---

## Loss Function

**BCEDiceLoss** — combines binary cross-entropy (handles pixel-level classification) with Dice loss (handles class imbalance). Equal weight; `smooth=1.0`.

```python
class BCEDiceLoss(nn.Module):
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

**Rationale:** In CASIA v2.0, tampered pixels are typically <5% of image area. BCE alone under-penalizes false negatives on the minority class. Dice loss directly optimizes the F1-like overlap, providing stronger gradients for small tampered regions.

---

## Optimizer

**AdamW with differential learning rates:**

```python
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(),          'lr': 1e-4},
    {'params': model.decoder.parameters(),          'lr': 1e-3},
    {'params': model.segmentation_head.parameters(), 'lr': 1e-3},
], weight_decay=1e-4)
```

| Parameter | Value | Rationale |
|---|---|---|
| Encoder LR | 1e-4 | Preserve pretrained features |
| Decoder LR | 1e-3 | Train new layers faster |
| Weight decay | 1e-4 | Regularization |

---

## Training Hyperparameters

| Parameter | MVP Value |
|---|---|
| Batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 |
| Max epochs | 50 |
| Early stopping patience | 10 (on val Pixel-F1) |
| Gradient clipping | `max_norm=1.0` |
| AMP | Enabled (`torch.amp`) |
| Seed | 42 |

---

## Mixed Precision Training

```python
scaler = GradScaler('cuda')

with autocast('cuda'):
    logits = model(images)
    loss = criterion(logits, masks) / ACCUMULATION_STEPS

scaler.scale(loss).backward()
```

---

## Training Loop — Critical Details

1. **Loss scaling:** Divide by `ACCUMULATION_STEPS` before `.backward()`.
2. **Gradient step:** Unscale, clip, step, update scaler every `ACCUMULATION_STEPS` batches.
3. **Partial window flush:** After the training loop's final batch, if `(batch_idx + 1) % ACCUMULATION_STEPS != 0`, flush the remaining accumulated gradients:

```python
# After the training batch loop
if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

4. **Validation:** Run after each epoch. Compare `val_f1` to `best_f1`. Save checkpoint if improved.
5. **Early stopping:** Stop if `epoch - best_epoch >= patience`.
6. **W&B logging (if enabled):** Log `train/loss`, `val/loss`, `val/pixel_f1`, `val/pixel_iou`, and learning rates per epoch inside the `USE_WANDB` guard.

---

## Checkpointing

**Saved state:**

```python
state = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_f1': best_f1,
    'best_epoch': best_epoch,
    'train_loss': avg_train_loss,
    'val_loss': val_loss,
    'val_f1': val_f1,
}
```

**Files:**

| File | When saved | Purpose |
|---|---|---|
| `best_model.pt` | When val F1 improves | Best model for evaluation |
| `last_checkpoint.pt` | Every epoch | Resume after crash |
| `checkpoint_epoch_N.pt` | Every 10 epochs | Periodic backup |

**Resume:**

```python
def load_checkpoint(filepath, model, optimizer, scaler):
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt['epoch'] + 1, ckpt['best_f1'], ckpt['best_epoch']
```

**Storage:** Google Drive (`/content/drive/MyDrive/tamper_detection/checkpoints/`).

---

## Phase 2 Addition: LR Scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

If enabled, the scheduler state must also be saved in checkpoints and restored on resume:

```python
# Save
state['scheduler_state_dict'] = scheduler.state_dict()
# Resume
scheduler.load_state_dict(ckpt['scheduler_state_dict'])
```

**Status:** Phase 2 only. MVP uses fixed learning rates.

---

## Data Pipeline

**MVP transforms (Phase 1) — spatial only:**

```python
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

**Phase 2 additions — photometric augmentations** (require `albumentations>=1.3.1,<2.0`):

```python
A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.2),
A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
```

**Excluded augmentations (by design):**
- `RandomCrop` — can crop out tampered regions
- `ElasticTransform` — destroys CFA/demosaicing patterns
- Heavy blur — destroys noise residuals (forensic signal)
- `CoarseDropout` — creates false mask patterns

---

## DataLoader Configuration

```python
DataLoader(dataset, batch_size=4, shuffle=True,     # train only
           num_workers=2, pin_memory=True, drop_last=True)

DataLoader(dataset, batch_size=4, shuffle=False,     # val/test
           num_workers=2, pin_memory=True, drop_last=False)
```
