# 15. Model Checkpoints — Strategy & Implementation Guide

## 15.1 Why Checkpointing Matters

Training a deep learning model for 50 epochs on a Colab T4 takes ~3.5 hours. If the session disconnects at epoch 45, you lose **everything** without checkpoints. More importantly, the best model may not be the last one — overfitting causes late-epoch performance degradation.

### Checkpointing Serves Three Purposes

| Purpose | What It Saves You From |
|---------|----------------------|
| **Crash recovery** | Session disconnects, OOM errors, runtime limits |
| **Best model selection** | Overfitting — epoch 35 may be better than epoch 50 |
| **Experiment reproducibility** | Anyone can reload and verify your results |

---

## 15.2 What to Save in a Checkpoint

### Minimal Checkpoint (Resume Training)

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),    # AMP scaler state
    'best_f1': best_f1,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_f1': val_f1,
}
```

### Full Checkpoint (Complete Reproducibility)

```python
checkpoint = {
    # Training state
    'epoch': epoch,
    'global_step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    
    # Metrics history
    'best_f1': best_f1,
    'best_epoch': best_epoch,
    'train_losses': train_losses,         # List of all epoch losses
    'val_losses': val_losses,
    'val_f1s': val_f1s,
    'val_ious': val_ious,
    
    # Config (for reconstruction)
    'config': {
        'encoder_name': 'efficientnet-b1',
        'in_channels': 6,
        'classes': 1,
        'image_size': 512,
        'batch_size': 4,
        'accumulation_steps': 4,
        'lr_encoder': 1e-4,
        'lr_decoder': 1e-3,
        'weight_decay': 1e-4,
        'seed': 42,
    },
    
    # Reproducibility
    'torch_rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state(),
    'numpy_rng_state': np.random.get_state(),
    'python_rng_state': random.getstate(),
}
```

---

## 15.3 Checkpoint Strategy

### Strategy: Save Every Epoch + Keep Best

```python
import os
from google.colab import drive

# Mount Google Drive for persistent storage
drive.mount('/content/drive')
CHECKPOINT_DIR = '/content/drive/MyDrive/BigVision/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(state, is_best, checkpoint_dir, epoch):
    """
    Save checkpoint every epoch. Keep only:
    - Last checkpoint (for resume)
    - Best checkpoint (for final evaluation)
    """
    # Save current epoch checkpoint
    filepath = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
    torch.save(state, filepath)
    
    # If this is the best model, save a separate copy
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(state, best_path)
        print(f"  ★ New best model saved (F1={state['val_f1']:.4f})")
    
    # Also save every N epochs for safety (optional)
    if (epoch + 1) % 10 == 0:
        periodic_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(state, periodic_path)
```

### Why Not Save Every Single Epoch?
Each checkpoint is ~35 MB. 50 epochs = 1.75 GB on Google Drive. The free tier has 15 GB total. Instead:
- **Always keep**: `last_checkpoint.pt` (overwritten each epoch) + `best_model.pt`
- **Periodic backups**: Every 10 epochs (5 files × 35 MB = 175 MB total)

---

## 15.4 Integration Into Training Loop

```python
best_f1 = 0.0
best_epoch = 0
train_losses, val_losses, val_f1s, val_ious = [], [], [], []

for epoch in range(start_epoch, NUM_EPOCHS):
    # === Train ===
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, 
                                  scaler, device, ACCUMULATION_STEPS)
    train_losses.append(train_loss)
    
    # === Validate ===
    val_loss, val_f1, val_iou = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_f1s.append(val_f1)
    val_ious.append(val_iou)
    
    # === LR Step ===
    scheduler.step()
    
    # === Checkpoint ===
    is_best = val_f1 > best_f1
    if is_best:
        best_f1 = val_f1
        best_epoch = epoch
    
    save_checkpoint(
        state={
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'val_f1': val_f1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1s': val_f1s,
            'val_ious': val_ious,
        },
        is_best=is_best,
        checkpoint_dir=CHECKPOINT_DIR,
        epoch=epoch,
    )
    
    # === Print Progress ===
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val F1: {val_f1:.4f} | "
          f"Best F1: {best_f1:.4f} (Epoch {best_epoch+1})")
    
    # === Early Stopping ===
    if epoch - best_epoch >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break
```

---

## 15.5 Resuming Training After Disconnect

```python
def load_checkpoint(checkpoint_dir, model, optimizer, scheduler, scaler, device):
    """
    Resume training from last checkpoint.
    Returns start_epoch and best_f1.
    """
    filepath = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
    
    if not os.path.exists(filepath):
        print("No checkpoint found. Starting from scratch.")
        return 0, 0.0, [], [], [], []
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint['best_f1']
    
    print(f"Resumed from epoch {checkpoint['epoch']+1} (best F1: {best_f1:.4f})")
    
    return (
        start_epoch,
        best_f1,
        checkpoint.get('train_losses', []),
        checkpoint.get('val_losses', []),
        checkpoint.get('val_f1s', []),
        checkpoint.get('val_ious', []),
    )

# Usage — put this BEFORE the training loop
start_epoch, best_f1, train_losses, val_losses, val_f1s, val_ious = \
    load_checkpoint(CHECKPOINT_DIR, model, optimizer, scheduler, scaler, device)
```

### Why Each State Matters

| State | What Happens If Not Restored |
|-------|------------------------------|
| `model_state_dict` | Model resets to random — useless |
| `optimizer_state_dict` | AdamW momentum buffers reset → LR spike, training instability |
| `scheduler_state_dict` | LR jumps back to initial value → overshooting |
| `scaler_state_dict` | AMP loss scale resets → potential gradient underflow/overflow |
| `best_f1` | Best model tracking resets → may overwrite a better checkpoint |
| `metric_histories` | Training curves lose history → incomplete plots |

---

## 15.6 Loading Best Model for Evaluation

```python
def load_best_model(checkpoint_dir, model, device):
    """Load the best model checkpoint for evaluation."""
    filepath = os.path.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['best_epoch']+1} "
          f"(F1={checkpoint['best_f1']:.4f})")
    return model

# Before evaluation
model = load_best_model(CHECKPOINT_DIR, model, device)
model.eval()
```

---

## 15.7 Checkpoint File Summary

| File | Size | Purpose | When Saved |
|------|------|---------|------------|
| `last_checkpoint.pt` | ~35 MB | Resume training after crash | Every epoch (overwritten) |
| `best_model.pt` | ~35 MB | Final evaluation & submission | When val F1 improves |
| `checkpoint_epoch_10.pt` | ~35 MB | Safety backup | Every 10 epochs |
| `checkpoint_epoch_20.pt` | ~35 MB | Safety backup | Every 10 epochs |
| **Total (50 epochs)** | **~245 MB** | — | — |

---

## 15.8 Advanced: Upload Best Checkpoint to HF Hub

After training completes, persist the best model beyond Google Drive:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj=os.path.join(CHECKPOINT_DIR, 'best_model.pt'),
    path_in_repo="model.pt",
    repo_id="your-username/tampering-detector-unet-effb1",
    repo_type="model",
    commit_message=f"Best model: F1={best_f1:.4f} at epoch {best_epoch+1}"
)
```

---

## 15.9 Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Saving model (not state_dict) | `torch.save(model, ...)` pickle-locks to exact class structure | Always save `model.state_dict()` |
| Forgetting scaler state | NaN loss after resume | Include `scaler.state_dict()` |
| Google Drive not mounted | `FileNotFoundError` | `drive.mount()` before training |
| Drive sync delay | Checkpoint appears empty | Add `time.sleep(1)` after save |
| Loading to wrong device | CUDA OOM or CPU slowdown | Use `map_location=device` |
