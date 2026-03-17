# 5. Resource Constraints — T4 GPU Optimization Guide

## 5.1 Assignment Requirement

> *"The solution should be runnable on Google Colab with a T4 GPU (15 GB VRAM)."*

This document covers every optimization technique needed to train a 512×512 segmentation model within T4 constraints.

---

## 5.2 T4 GPU Specifications

| Spec | Value | Practical Implication |
|------|-------|-----------------------|
| **VRAM** | 15 GB (15,360 MB) | Maximum memory for model + activations + gradients + optimizer states |
| **CUDA Cores** | 2,560 | Moderate throughput; not a bottleneck for our batch size |
| **Tensor Cores** | 320 | Enable FP16 operations → up to 2× speedup with AMP |
| **Memory Bandwidth** | 320 GB/s | Data loading can become a bottleneck with heavy augmentation |
| **FP16 Throughput** | 65 TFLOPS | Tensor cores only activated with AMP |
| **Colab Session** | Max 12 hours (free), ~24h (Pro) | Must complete training within session limits |

---

## 5.3 VRAM Budget Breakdown

### Memory Consumers (512×512, batch_size=4)

| Component | Estimated Memory | Notes |
|-----------|-----------------|-------|
| **Model Parameters (FP32)** | ~33 MB | ~8.2M params × 4 bytes |
| **Model Parameters (AMP)** | ~49 MB | FP32 master copy + FP16 forward copy |
| **Optimizer States (AdamW)** | ~66 MB | 2 momentum buffers per param (FP32) |
| **Forward Activations (FP16)** | ~2.5 GB | All intermediate feature maps (with AMP) |
| **Gradient Storage (FP16)** | ~1.2 GB | Same shape as activations |
| **Input Batch** | ~25 MB | 4 × 6 × 512 × 512 × 2 bytes (FP16) |
| **Ground Truth + Loss** | ~6 MB | 4 × 1 × 512 × 512 × 4 bytes |
| **PyTorch Overhead** | ~1–2 GB | CUDA context, memory allocator fragmentation |
| **Total** | **~5.8 GB** | Leaves ~9 GB headroom |

### Without AMP (FP32 Throughout)

| Component | Estimated Memory |
|-----------|-----------------|
| **Forward Activations (FP32)** | ~5.0 GB |
| **Gradient Storage (FP32)** | ~2.4 GB |
| **Total** | **~9.5 GB** |

**AMP saves ~3.7 GB** — the difference between comfortable and barely-fitting.

---

## 5.4 Optimization 1: Automatic Mixed Precision (AMP)

### What It Does
AMP runs forward pass and loss computation in FP16 (half precision) while maintaining FP32 master weights for gradient accumulation. This:
- Halves VRAM for activations and gradients
- Activates T4 Tensor Cores → ~1.5–2× throughput
- Maintains FP32 accuracy via loss scaling

### Implementation

```python
from torch.amp import autocast, GradScaler

# Create scaler for loss scaling (prevents FP16 underflow)
scaler = GradScaler('cuda')

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass in FP16
        with autocast('cuda'):
            logits = model(images)
            loss, loss_dict = criterion(logits, masks)
        
        # Scale loss and backward in FP16
        scaler.scale(loss).backward()
        
        # Unscale, clip gradients, step optimizer
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

### Critical Notes
- `autocast` wraps ONLY the forward pass + loss, NOT the backward pass (autograd handles that automatically)
- `set_to_none=True` in `zero_grad()` saves memory by releasing gradient tensors instead of zeroing them
- `clip_grad_norm_` AFTER `scaler.unscale_()` to clip in FP32 space

---

## 5.5 Optimization 2: Gradient Accumulation

### Why We Need It
- **Ideal effective batch size**: 16 (stable loss, better generalization)
- **Max physical batch that fits on T4**: 4 (at 512×512 with our model)
- **Solution**: Accumulate gradients from 4 micro-batches before stepping

### Implementation

```python
ACCUMULATION_STEPS = 4  # Effective batch = 4 × 4 = 16

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast('cuda'):
            logits = model(images)
            loss, loss_dict = criterion(logits, masks)
            loss = loss / ACCUMULATION_STEPS  # Scale loss to average over accumulation
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    
    # Handle final incomplete accumulation at epoch end
    if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

### The Key Detail
`loss = loss / ACCUMULATION_STEPS` — this is essential. Without dividing, you're summing 4 losses instead of averaging, which changes your effective learning rate by 4×.

---

## 5.6 Optimization 3: DataLoader Configuration

### Colab-Specific Settings

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,          # Colab has 2 CPU cores (free tier)
    pin_memory=True,        # Pre-loads data into CUDA-pinned RAM → faster GPU transfer
    drop_last=True,         # Avoid small final batch (messes with batch norm)
    prefetch_factor=2,      # Prefetch 2 batches per worker (default, don't increase)
    persistent_workers=True # Keep workers alive between epochs
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,           # Can be larger: no gradients stored during validation
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False         # Evaluate every sample
)
```

### Why These Specific Values

| Parameter | Value | Reason |
|-----------|-------|--------|
| `num_workers=2` | 2 | Colab free tier provides only 2 CPU cores; more workers cause context-switching overhead |
| `pin_memory=True` | True | Allocates data in page-locked memory → DMA transfer to GPU (non-blocking) |
| `drop_last=True` | True | Final batch may be size 1, causing BatchNorm instability |
| `persistent_workers=True` | True | Avoids respawning workers each epoch (~2-3s saved per epoch) |
| `prefetch_factor=2` | 2 | Default; higher values waste RAM with minimal speed benefit on T4 |

---

## 5.7 Optimization 4: Memory-Efficient Validation

```python
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Validation loop with no gradient computation → significantly less VRAM.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast('cuda'):
            logits = model(images)
            loss, _ = criterion(logits, masks)
        
        total_loss += loss.item()
        
        # Move predictions to CPU immediately to free GPU memory
        probs = torch.sigmoid(logits).cpu()
        all_preds.append(probs)
        all_targets.append(masks.cpu())
    
    # Compute metrics on CPU
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return total_loss / len(val_loader), all_preds, all_targets
```

**Key memory optimization**: `.cpu()` immediately after GPU computation prevents accumulating validation tensors on GPU.

---

## 5.8 Optimization 5: Differential Learning Rates

```python
# Encoder is pre-trained → fine-tune gently
# Decoder is randomly initialized → train aggressively
# SRM is frozen → no optimizer needed

optimizer = torch.optim.AdamW([
    {'params': model.channel_reducer.parameters(),
     'lr': 1e-3},               # Small component, train fast
    {'params': model.segmentation_model.encoder.parameters(),
     'lr': 1e-4},               # Pre-trained, fine-tune slowly
    {'params': model.segmentation_model.decoder.parameters(),
     'lr': 1e-3},               # Random init, train fast
    {'params': model.segmentation_model.segmentation_head.parameters(),
     'lr': 1e-3},               # Random init, train fast
], weight_decay=1e-4)

# Cosine annealing: warm start → gradual decay → near-zero
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,           # Total epochs
    eta_min=1e-6        # Minimum learning rate
)
```

### Why Differential LRs
The encoder has pre-trained weights that already extract meaningful features. Training it with a high learning rate would destroy this knowledge ("catastrophic forgetting"). The decoder starts from scratch and needs to learn fast.

---

## 5.9 Optimization 6: Gradient Checkpointing (Emergency Only)

**Use this only if VRAM runs out with the above optimizations.** It trades compute for memory.

```python
# Enable gradient checkpointing on the encoder
model.segmentation_model.encoder.set_grad_checkpointing(enable=True)
```

**Effect**: Recomputes activations during backward pass instead of storing them. Reduces activation memory by ~50% but increases training time by ~25%.

**Our recommendation**: Start WITHOUT gradient checkpointing. If you get OOM with batch_size=4, enable it rather than reducing batch size below 4.

---

## 5.10 VRAM Monitoring

### In-Notebook Monitoring

```python
def print_gpu_memory():
    """Print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# Call at strategic points:
print_gpu_memory()  # After model creation
# ... after first forward pass
print_gpu_memory()  # After first batch (peak usage)
```

### Emergency OOM Recovery

```python
import gc

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
```

---

## 5.11 Training Time Estimates

| Configuration | Time/Epoch | 50 Epochs | Fits in Colab Session? |
|---------------|-----------|-----------|----------------------|
| FP32, BS=2, no accumulation | ~8 min | ~6.7 hours | Barely (free tier) |
| **AMP, BS=4, accumulation=4** | **~4 min** | **~3.3 hours** | **Yes, comfortable** |
| AMP, BS=4, gradient checkpoint | ~5 min | ~4.2 hours | Yes |

**Recommended setup** (AMP + BS=4 + acc=4) completes training well within a Colab free-tier session.

---

## 5.12 Colab-Specific Tips

### Runtime Selection
```python
# Verify you have a T4:
import torch
print(torch.cuda.get_device_name(0))
# Expected: "Tesla T4"
```

### Prevent Disconnection
- Keep browser tab active (Colab disconnects after ~90 min of inactivity)
- Print progress every batch as console activity

### Save Checkpoints to Drive
```python
from google.colab import drive
drive.mount('/content/drive')

CHECKPOINT_DIR = '/content/drive/MyDrive/BigVision/checkpoints/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Save after each epoch
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_f1': best_f1,
}, f'{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pt')
```

### Resume Training After Disconnect
```python
# Load last checkpoint and continue
checkpoint = torch.load(f'{CHECKPOINT_DIR}/checkpoint_epoch_{last_epoch}.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
best_f1 = checkpoint['best_f1']
```

---

## 5.13 Pre-Training Checklist

Before starting the training run, verify:

- [ ] `torch.cuda.get_device_name(0)` shows "Tesla T4"
- [ ] AMP scaler is created and integrated into training loop
- [ ] Gradient accumulation divides loss by `ACCUMULATION_STEPS`
- [ ] `optimizer.zero_grad(set_to_none=True)` used (not `zero_grad()`)
- [ ] Validation loop uses `@torch.no_grad()` and `autocast`
- [ ] `pin_memory=True` on both DataLoaders
- [ ] Google Drive mounted for checkpoint saving
- [ ] `print_gpu_memory()` placed after first batch to verify VRAM usage
- [ ] Expected VRAM usage: 5–6 GB (leaves comfortable headroom)
