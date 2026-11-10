# 09 — Engineering Practices

## Purpose

This document specifies coding standards, T4 GPU constraints, dependency management, and reproducibility practices for the Colab notebook.

## Environment: Google Colab (T4 GPU)

| Resource | Specification |
|---|---|
| GPU | NVIDIA T4 (15 GB VRAM, 320 Tensor Cores) |
| RAM | ~12.7 GB (free tier) |
| CPU cores | 2 |
| Session limit | 12 hours |
| Disk | ~100 GB (ephemeral) |

## Dependencies

Install at the top of the notebook. Pin versions for reproducibility.

```python
!pip install -q segmentation-models-pytorch albumentations kaggle tqdm
```

Core imports:

```python
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
```

## T4 VRAM Budget

With AMP enabled, batch size 4, and 512×512 inputs:

| Component | VRAM |
|---|---|
| Model parameters (FP16) | ~49 MB |
| Optimizer states (AdamW) | ~66 MB |
| Forward activations (FP16) | ~2.5 GB |
| Gradient storage (FP16) | ~1.2 GB |
| Input batch | ~25 MB |
| PyTorch overhead | ~1–2 GB |
| **Total** | **~5.8 GB** |

This leaves comfortable headroom on the 15 GB T4. Without AMP, usage rises to ~9.5 GB.

## Mixed Precision Training (AMP)

AMP is required for efficient T4 usage. It halves VRAM for activations/gradients and activates Tensor Cores.

```python
scaler = GradScaler('cuda')

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
```

## Gradient Accumulation

Effective batch size = `batch_size × ACCUMULATION_STEPS` = 4 × 4 = 16.

**Critical:** Divide loss by `ACCUMULATION_STEPS` before `.backward()`. Without this, the effective learning rate scales by the accumulation factor.

## Memory Management

```python
# Use set_to_none=True for faster zero_grad
optimizer.zero_grad(set_to_none=True)

# No-grad context for validation
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    ...

# Clear cache if needed between training and evaluation
torch.cuda.empty_cache()
```

## DataLoader Optimization

```python
DataLoader(
    batch_size=4,
    num_workers=2,        # Match Colab CPU cores
    pin_memory=True,      # Pre-load into CUDA-pinned RAM
    drop_last=True,       # Training only; avoids small final batch
    persistent_workers=True,
)
```

## Reproducibility

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

Call `set_seed()` at the very start of the notebook, before any data loading or model initialization.

## Notebook Structure

The notebook follows a sequential cell structure:

| Section | Content |
|---|---|
| 1. Setup | Install dependencies, set seed, detect GPU |
| 2. Dataset | Download via Kaggle API, discover pairs, validate alignment |
| 3. Preprocessing | Binarize masks, stratified split |
| 4. Data pipeline | Dataset class, transforms, DataLoaders |
| 5. Model | Architecture definition, model instantiation |
| 6. Training | Loss, optimizer, scheduler, training loop |
| 7. Evaluation | Metrics computation, results table |
| 8. Visualization | Prediction grids, training curves, ROC |
| 9. Bonus (optional) | Robustness testing, SRM ablation |
| 10. Save | Checkpoint to Google Drive, export figures |

Each section starts with a markdown cell explaining what follows.

## Kaggle API Setup

```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'

!kaggle datasets download -d sophatvathana/casia-dataset
!unzip -q casia-dataset.zip -d ./data/
```

Store Kaggle credentials in Colab secrets or environment variables. Do not hardcode credentials in the notebook.

## Google Drive Integration (Checkpoints)

```python
from google.colab import drive
drive.mount('/content/drive')

CHECKPOINT_DIR = '/content/drive/MyDrive/tamper_detection/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

Save checkpoints to Google Drive so they persist beyond the Colab session.

## Progress Tracking

Use `tqdm` for all loops:

```python
from tqdm import tqdm

for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}'):
    ...
```

## Code Style

- Use clear variable names: `train_loader`, `val_f1`, `best_epoch`.
- Group related logic into functions: `train_one_epoch()`, `validate()`, `evaluate()`.
- Avoid global state; pass device, model, and config as function arguments.
- Add brief comments only where logic is non-obvious.
- Use type hints in function signatures for clarity.

## Tools Not Used

The following are intentionally excluded from the core implementation:

| Tool | Reason |
|---|---|
| HuggingFace Hub | Post-project portfolio work |
| Databricks | Enterprise platform; irrelevant to Colab notebook |
| DuckDB / DynamoDB | Database layer unnecessary for 5K images |
| NVIDIA DALI | Complex data loading; standard DataLoader is sufficient |
| `torch.compile` | Can cause issues in Colab and adds debugging complexity |
| channels-last memory format | Marginal T4 benefit; adds code complexity |

## Optional: Weights & Biases

W&B is optional experiment tracking. The notebook must be complete without it.

```python
# Optional: W&B logging
try:
    import wandb
    wandb.init(project="tamper-detection", config={...})
    USE_WANDB = True
except ImportError:
    USE_WANDB = False

# In training loop:
if USE_WANDB:
    wandb.log({'train_loss': avg_loss, 'val_f1': val_f1, 'epoch': epoch})
```

## Related Documents

- [05_Training_Pipeline.md](05_Training_Pipeline.md) — Training loop using these practices
- [11_Final_Submission_Checklist.md](11_Final_Submission_Checklist.md) — Pre-submission checks
