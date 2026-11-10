# 11. Weights & Biases (W&B) — Experiment Tracking Guide

## 11.1 What Is Weights & Biases?

Weights & Biases (wandb) is an **experiment tracking, model versioning, and collaboration platform** for ML teams. Think of it as "Git for experiments" — every training run is logged with its hyperparameters, metrics, system stats, and artifacts in a central dashboard.

### Core Features

| Feature | What It Does | Why You Care |
|---------|-------------|-------------|
| **Experiment Tracking** | Logs loss, metrics, LR, GPU usage per step/epoch | Compare runs side-by-side; never lose a result |
| **Hyperparameter Logging** | Stores all config in a structured dict | Perfect reproducibility; know exactly what produced each result |
| **System Monitoring** | GPU utilization, VRAM, CPU, disk I/O | Spot bottlenecks (e.g., data loading starving GPU) |
| **Artifact Versioning** | Version model weights, datasets, predictions | Track which model came from which run |
| **Collaboration** | Shared project dashboards, teams | Evaluators/teammates see results instantly via a link |
| **Sweeps** | Automated hyperparameter search | Systematic tuning instead of guessing (see Doc 12) |

---

## 11.2 Why Use W&B in This Project?

### The Problem Without It
When training in Colab, you lose everything when the session ends:
- Print statements scroll off screen
- You forget which learning rate produced the best result
- Comparing 3 runs means opening 3 notebooks and scrolling
- Evaluators see only the final state, not the journey

### What W&B Gives You
1. **Persistent logging** — all metrics survive session disconnects
2. **Live dashboard** — watch training curves update in real-time from a browser tab
3. **Run comparison** — overlay loss/F1 curves from multiple experiments
4. **Shareable links** — send your evaluator a single URL showing all experiments
5. **Free tier** — unlimited personal projects (100 GB storage)

### Is It Worth the Setup Cost?
**Yes.** Setup takes ~5 minutes. The return:
- Debug training issues 10× faster (spot NaN loss, GPU idle, LR too high)
- Never re-run an experiment because you forgot to save the config
- Demonstrate professional ML engineering practices to evaluators

---

## 11.3 Setup

### Installation

```python
!pip install -q wandb
```

### Authentication

```python
import wandb

# Option 1: Interactive login (prompts for API key)
wandb.login()

# Option 2: Programmatic (for Colab — paste key from https://wandb.ai/authorize)
wandb.login(key="YOUR_API_KEY")  # Replace with your key
```

### Project Initialization

```python
# Initialize a new run
run = wandb.init(
    project="bigvision-tampering-detection",
    name="unet-effb1-srm-v1",          # Human-readable run name
    config={
        # Hyperparameters — logged and searchable
        "architecture": "UNet",
        "encoder": "efficientnet-b1",
        "input_channels": 6,
        "image_size": 512,
        "batch_size": 4,
        "effective_batch_size": 16,
        "accumulation_steps": 4,
        "learning_rate_encoder": 1e-4,
        "learning_rate_decoder": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 50,
        "loss": "BCE + Dice + Edge",
        "loss_weights": {"bce": 1.0, "dice": 1.0, "edge": 0.5},
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "dataset": "CASIA_v2.0",
        "srm_preprocessing": True,
        "amp": True,
        "seed": 42,
    },
    tags=["baseline", "srm", "efficientnet-b1"],
)
```

---

## 11.4 Integration Into Training Loop

### Per-Step Logging

```python
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        # ... forward pass, loss computation, backward ...
        
        # Log every accumulation step
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            global_step = epoch * len(train_loader) + batch_idx
            
            wandb.log({
                "train/loss_total": loss_dict['total'],
                "train/loss_bce": loss_dict['bce'],
                "train/loss_dice": loss_dict['dice'],
                "train/loss_edge": loss_dict['edge'],
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/gpu_memory_gb": torch.cuda.memory_allocated() / 1e9,
            }, step=global_step)
```

### Per-Epoch Logging

```python
    # After validation
    val_loss, val_f1, val_iou = validate(model, val_loader, criterion, device)
    
    wandb.log({
        "epoch": epoch,
        "val/loss": val_loss,
        "val/pixel_f1": val_f1,
        "val/pixel_iou": val_iou,
        "val/best_f1": best_f1,
    }, step=global_step)
    
    # Log learning rate schedule
    scheduler.step()
```

### Log Prediction Visualizations

```python
    # Every 5 epochs, log sample predictions as images
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            sample_imgs, sample_masks = next(iter(val_loader))
            sample_imgs = sample_imgs.to(device)
            preds = torch.sigmoid(model(sample_imgs)).cpu()
        
        # Create wandb Image objects
        wandb_images = []
        for i in range(min(4, len(sample_imgs))):
            img = sample_imgs[i].cpu().permute(1, 2, 0).numpy()
            # Denormalize
            img = np.clip(img * np.array([0.229, 0.224, 0.225]) + 
                          np.array([0.485, 0.456, 0.406]), 0, 1)
            
            wandb_images.append(wandb.Image(
                img,
                masks={
                    "ground_truth": {"mask_data": sample_masks[i].squeeze().numpy()},
                    "prediction": {"mask_data": (preds[i].squeeze().numpy() > 0.5).astype(float)},
                },
                caption=f"Epoch {epoch}"
            ))
        
        wandb.log({"val/predictions": wandb_images}, step=global_step)
```

---

## 11.5 Model Artifact Logging

```python
# Save best model as a W&B artifact
if val_f1 > best_f1:
    best_f1 = val_f1
    
    # Save locally first
    torch.save(model.state_dict(), 'best_model.pt')
    
    # Log as artifact
    artifact = wandb.Artifact(
        name="best-model",
        type="model",
        description=f"Best model at epoch {epoch} with F1={val_f1:.4f}",
        metadata={"epoch": epoch, "f1": val_f1, "iou": val_iou}
    )
    artifact.add_file('best_model.pt')
    run.log_artifact(artifact)
```

---

## 11.6 Final Summary Logging

```python
# After training completes
wandb.summary.update({
    "best_val_f1": best_f1,
    "best_epoch": best_epoch,
    "total_training_time_hours": total_time / 3600,
    "test_pixel_f1": test_results['pixel_f1_mean'],
    "test_pixel_iou": test_results['pixel_iou_mean'],
    "test_image_auc": test_results['image_auc_roc'],
    "oracle_f1": test_results['oracle_f1'],
})

# Finish the run
wandb.finish()
```

---

## 11.7 What the Dashboard Shows

After training, your W&B project page will contain:

1. **Run table** — all experiments with key metrics as sortable columns
2. **Loss curves** — train/val loss overlaid, zoomable, smoothable
3. **Metric curves** — F1, IoU, AUC over epochs
4. **System metrics** — GPU utilization %, VRAM usage, CPU load (auto-collected)
5. **Config diff** — compare hyperparameters between any two runs
6. **Prediction images** — visual results logged every 5 epochs
7. **Artifacts** — downloadable model weights linked to the exact run

### Sharing with Evaluators
```
https://wandb.ai/YOUR_USERNAME/bigvision-tampering-detection
```
Include this link in your notebook's conclusion section.

---

## 11.8 Should You Use W&B for This Project?

| Consideration | Answer |
|---------------|--------|
| **Is it required?** | No — the assignment only asks for a Colab notebook |
| **Does it add value?** | Yes — demonstrates professional ML practices |
| **Time to set up?** | ~5 minutes |
| **Risk?** | Near zero — if it breaks, comment out `wandb.log()` lines and train normally |
| **Evaluator impression?** | Strong positive — shows you track experiments systematically |

**Verdict: Use it.** The ROI is excellent. Worst case, it's a few extra lines. Best case, it saves hours of debugging and impresses evaluators.

---

## 11.9 Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Forgetting `wandb.finish()` | Always call it at the end; otherwise the run appears "crashed" |
| Logging too frequently | Log per-batch for loss, per-epoch for metrics. Don't log every sample. |
| API key in notebook | Use `wandb.login()` interactively, or use Colab secrets |
| Large artifacts | Don't log full dataset as artifact — just model weights |
| Offline mode in Colab | Set `os.environ['WANDB_MODE'] = 'online'` to ensure cloud sync |
