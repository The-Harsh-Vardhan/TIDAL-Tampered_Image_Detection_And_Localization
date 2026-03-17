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

**Rationale:** In CASIA, tampered pixels are typically <5% of image area. BCE alone under-penalizes false negatives on the minority class. Dice loss directly optimizes the F1-like overlap, providing stronger gradients for small tampered regions.

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

| Parameter | Value |
|---|---|
| Image size | 384 × 384 |
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
3. **Partial window flush:** After the training loop's final batch, if `(batch_idx + 1) % ACCUMULATION_STEPS != 0`, flush the remaining accumulated gradients.
4. **Validation:** Run after each epoch, collect validation probabilities once, and sweep thresholds.
5. **Checkpoint selection:** Compare the best thresholded `val_f1` to `best_f1`. Save checkpoint and `best_threshold` if improved.
6. **Early stopping:** Stop if `epoch - best_epoch >= patience`.
7. **W&B logging (if enabled):** Log `train/loss`, `val/loss`, `val/pixel_f1`, `val/pixel_iou`, `val/best_threshold`, and learning rates per epoch.

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
    'best_threshold': best_threshold,
    'train_loss': avg_train_loss,
    'val_loss': val_loss,
    'val_f1': val_f1,
    'val_iou': val_iou,
}
```

**Files:**

| File | When Saved | Purpose |
|---|---|---|
| `best_model.pt` | When val F1 improves | Best model for evaluation |
| `last_checkpoint.pt` | Every epoch | Resume after interruption |
| `checkpoint_epoch_N.pt` | Every 10 epochs | Periodic backup |

**Storage:** `/kaggle/working/checkpoints/`

---

## Data Pipeline

**Training transforms (spatial augmentation):**

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

**Validation/test transforms:**

```python
val_transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
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
