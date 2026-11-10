# 05 — Training Pipeline

## Purpose

This document specifies the loss function, optimizer, scheduler, training loop, and checkpointing strategy.

## Loss Function

### Hybrid Loss: BCE + Dice

The default loss combines Binary Cross-Entropy and Dice loss to handle the severe class imbalance in tampering masks (typically <5% of pixels are tampered).

$$L_{total} = L_{BCE} + L_{Dice}$$

Both components are weighted equally (1.0 each) by default.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        # BCE on raw logits (numerically stable)
        bce_loss = self.bce(logits, targets)

        # Dice on probabilities
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
| Dice | Region overlap optimization; inherently handles class imbalance | Model predicts mostly zeros (no tampered regions) |

### Optional: Edge Loss (Stage 2)

Edge loss improves boundary sharpness. Add it only if the baseline model produces blurry mask boundaries.

```python
def edge_loss(logits, targets, kernel_size=3):
    """BCE loss on morphological edge of the mask."""
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=targets.device)
    dilated = F.conv2d(targets, kernel, padding=kernel_size // 2).clamp(0, 1)
    eroded = 1.0 - F.conv2d(1.0 - targets, kernel, padding=kernel_size // 2).clamp(0, 1)
    edges = dilated - eroded
    return F.binary_cross_entropy_with_logits(logits * edges, targets * edges)
```

If used, the combined loss becomes: $L = L_{BCE} + L_{Dice} + 0.5 \cdot L_{Edge}$.

## Optimizer and Scheduler

### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW([
    {'params': model.unet.encoder.parameters(), 'lr': 1e-4},
    {'params': model.unet.decoder.parameters(), 'lr': 1e-3},
    {'params': model.unet.segmentation_head.parameters(), 'lr': 1e-3},
], weight_decay=1e-4)
```

Differential learning rates: the pretrained encoder uses a smaller learning rate (1e-4) to preserve ImageNet features, while the decoder and segmentation head use a larger rate (1e-3) since they are trained from scratch.

### Scheduler: CosineAnnealingWarmRestarts

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

This provides smooth learning rate decay with periodic warm restarts. Step the scheduler after each epoch.

## Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay; standard for modern architectures |
| Encoder LR | 1e-4 | Preserve pretrained features |
| Decoder LR | 1e-3 | Faster learning for new layers |
| Weight decay | 1e-4 | Regularization for a small dataset |
| Batch size | 4 | Fits T4 VRAM with AMP |
| Gradient accumulation | 4 steps | Effective batch size = 16 |
| Input resolution | 512×512 | Preserves forensic detail |
| Max epochs | 50 | Convergence typically occurs at 25–35 epochs |
| Early stopping patience | 10 epochs | Stop if validation F1 does not improve |
| Checkpoint metric | Best validation Pixel-F1 | F1 is the primary metric, not loss |
| Random seed | 42 | Reproducibility |

## Mixed Precision Training (AMP)

AMP halves VRAM usage for activations and gradients, and activates T4 Tensor Cores for faster computation.

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')
```

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
            loss = loss / ACCUMULATION_STEPS  # Scale for accumulation

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * ACCUMULATION_STEPS

    scheduler.step()
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
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_f1': best_f1,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'val_f1': val_f1,
    }, is_best, checkpoint_dir, epoch)

    # Early stopping
    if epoch - best_epoch >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

**Critical:** Loss must be divided by `ACCUMULATION_STEPS` before `.backward()`. Without this, the effective learning rate is multiplied by the accumulation factor.

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

    avg_loss = total_loss / len(val_loader)
    avg_f1 = np.mean(f1_scores)
    avg_iou = np.mean(iou_scores)

    return avg_loss, avg_f1, avg_iou
```

## Checkpointing

### Save Function

```python
import os

def save_checkpoint(state, is_best, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Always save last checkpoint
    last_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
    torch.save(state, last_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(state, best_path)
    
    # Periodic backup every 10 epochs
    if (epoch + 1) % 10 == 0:
        periodic_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save(state, periodic_path)
```

### Resume Function

```python
def load_checkpoint(filepath, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint['best_f1']
    return start_epoch, best_f1
```

### Storage Estimate

| File | Size | Frequency |
|---|---|---|
| `last_checkpoint.pt` | ~35 MB | Every epoch (overwritten) |
| `best_model.pt` | ~35 MB | When validation F1 improves |
| `checkpoint_epoch_N.pt` | ~35 MB | Every 10 epochs |
| Total for 50 epochs | ~245 MB | Fits Google Drive free tier |

## Reproducibility

Set all random seeds at the start of the notebook:

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Related Documents

- [04_Model_Architecture.md](04_Model_Architecture.md) — Model definition
- [06_Evaluation_Methodology.md](06_Evaluation_Methodology.md) — Metrics computed during validation
- [09_Engineering_Practices.md](09_Engineering_Practices.md) — Code quality and T4 optimization
