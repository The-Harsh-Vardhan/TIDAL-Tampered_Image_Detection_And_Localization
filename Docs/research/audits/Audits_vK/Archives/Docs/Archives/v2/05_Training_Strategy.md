# 05 — Training Strategy

## Purpose

Specify the loss function, optimizer, training loop, and checkpointing strategy.

## Loss Function: BCE + Dice

The hybrid loss handles severe class imbalance (typically <5% of pixels are tampered).

```python
import torch
import torch.nn as nn

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

| Component | Role | Without it |
|---|---|---|
| BCE | Pixel-level distribution matching | Dice alone produces noisy gradients |
| Dice | Region overlap; handles class imbalance | Model predicts mostly zeros |

Edge loss is **not** part of the baseline. It may be considered if boundary quality is poor, but the formulation in earlier docs had issues and should not be copied directly.

## Optimizer: AdamW

```python
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 1e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-3},
    {'params': model.segmentation_head.parameters(), 'lr': 1e-3},
], weight_decay=1e-4)
```

Note: These are direct attributes of `smp.Unet`. Do not use `model.unet.encoder` or similar nested paths.

Differential learning rates: the pretrained encoder uses a smaller rate (1e-4) to preserve ImageNet features, while the decoder uses a larger rate (1e-3).

## Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay |
| Encoder LR | 1e-4 | Preserve pretrained features |
| Decoder LR | 1e-3 | Faster learning for new layers |
| Weight decay | 1e-4 | Regularization |
| Batch size | 4 | Fits T4 VRAM with AMP |
| Gradient accumulation | 4 steps | Effective batch = 16 |
| Input resolution | 512x512 | — |
| Max epochs | 50 | With early stopping |
| Early stopping patience | 10 epochs | On validation Pixel-F1 |
| Checkpoint metric | Best validation Pixel-F1 | — |
| Random seed | 42 | Reproducibility |

## LR Scheduler (Phase 2)

The scheduler is **not** part of the Phase 1 MVP. Add it in Phase 2:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

Step at the end of each epoch. If using a scheduler, include `scheduler_state_dict` in checkpoints.

## Mixed Precision Training (AMP)

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')
```

AMP is part of the baseline. It reduces VRAM usage and activates T4 Tensor Cores.

## Training Loop

```python
ACCUMULATION_STEPS = 4

for epoch in range(start_epoch, max_epochs):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch_loss = 0.0

    for batch_idx, (images, masks, labels) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, masks)
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * ACCUMULATION_STEPS

    # Flush any remaining accumulated gradients
    if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_train_loss = epoch_loss / len(train_loader)

    # Validation
    val_loss, val_f1, val_iou = validate(model, val_loader, criterion, device)

    # Checkpointing
    is_best = val_f1 > best_f1
    if is_best:
        best_f1 = val_f1
        best_epoch = epoch

    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'val_f1': val_f1,
    }, is_best, checkpoint_dir, epoch)

    # Early stopping
    if epoch - best_epoch >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

**Critical fix:** The loop flushes remaining gradients after the last batch if it does not align with `ACCUMULATION_STEPS`. Without this, the final partial accumulation window is lost.

**Critical fix:** Loss must be divided by `ACCUMULATION_STEPS` before `.backward()`. Without this, the effective learning rate scales by the accumulation factor.

## Validation Function

```python
@torch.no_grad()
def validate(model, val_loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    f1_scores = []
    iou_scores = []

    for images, masks, labels in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        for pred, gt in zip(preds, masks):
            f1_scores.append(compute_pixel_f1(pred, gt))
            iou_scores.append(compute_iou(pred, gt))

    return total_loss / len(val_loader), np.mean(f1_scores), np.mean(iou_scores)
```

## Checkpointing

```python
import os

def save_checkpoint(state, is_best, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)

    last_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
    torch.save(state, last_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(state, best_path)

    if (epoch + 1) % 10 == 0:
        periodic_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save(state, periodic_path)
```

Resume:

```python
def load_checkpoint(filepath, model, optimizer, scaler, device):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint['best_f1']
    best_epoch = checkpoint['best_epoch']
    return start_epoch, best_f1, best_epoch
```

Checkpoint state must include `best_epoch` for correct early-stopping resume.

Storage requirements depend on model size and optimizer state — measure in the notebook rather than assuming a specific size.

## Reproducibility

```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Related Documents

- [04_Model_Architecture.md](04_Model_Architecture.md) — Model definition and API
- [06_Evaluation_Methodology.md](06_Evaluation_Methodology.md) — Metrics used during validation
