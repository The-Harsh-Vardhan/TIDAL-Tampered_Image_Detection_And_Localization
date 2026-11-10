# 09 — Engineering Practices

## Purpose

Specify coding standards, environment setup, and reproducibility practices for the Colab notebook.

## Environment: Google Colab (T4 GPU)

The project targets the free Colab tier with a T4 GPU. Exact resource limits (VRAM, RAM, CPU cores, session time) vary by runtime and should be verified at notebook start.

## Dependencies

Install at the top of the notebook:

```python
!pip install segmentation-models-pytorch albumentations
```

This installs the two libraries not included in base Colab. Other dependencies (`torch`, `torchvision`, `numpy`, `matplotlib`, `sklearn`, `cv2`, `PIL`, `tqdm`) are pre-installed in Colab.

If a specific version is needed for compatibility, pin it explicitly:

```python
!pip install segmentation-models-pytorch==0.3.3 albumentations==1.3.1
```

Do not claim version pinning without actually pinning versions in the install command.

## Core Imports

```python
import os, random
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

## GPU Verification

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

## Kaggle API Setup

```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'

!kaggle datasets download -d sophatvathana/casia-dataset
!unzip -q casia-dataset.zip -d ./data/
```

Use Colab secrets or environment variables for credentials. Do not hardcode API keys in the notebook.

## Google Drive (Checkpoint Persistence)

```python
from google.colab import drive
drive.mount('/content/drive')

CHECKPOINT_DIR = '/content/drive/MyDrive/tamper_detection/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

## Notebook Structure

| Section | Content |
|---|---|
| 1. Setup | Install, seed, GPU check |
| 2. Dataset | Download, discover pairs, validate alignment |
| 3. Preprocessing | Binarize masks, stratified split, persist manifest |
| 4. Data pipeline | Dataset class, transforms, DataLoaders |
| 5. Model | Architecture definition |
| 6. Training | Loss, optimizer, training loop |
| 7. Evaluation | Metrics, results table |
| 8. Visualization | Prediction grids, training curves |
| 9. Bonus (optional) | Robustness testing |
| 10. Save | Checkpoint to Drive, export figures |

Each section starts with a markdown cell explaining purpose and key decisions.

## Progress Tracking

```python
from tqdm import tqdm

for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}'):
    ...
```

## Code Style

- Use clear variable names.
- Group logic into functions: `train_one_epoch()`, `validate()`, `evaluate()`.
- Avoid global state; pass device, model, and config as function arguments.
- Add comments only where logic is non-obvious.

## Tools Excluded from Core Path

| Tool | Reason |
|---|---|
| HuggingFace Hub | Post-project portfolio work |
| Databricks | Enterprise platform; irrelevant to Colab |
| DuckDB / DynamoDB | Database layer unnecessary for this dataset size |
| NVIDIA DALI | Standard DataLoader is sufficient |
| `torch.compile` | Can cause issues in Colab; adds debugging complexity |
| channels-last memory format | Marginal benefit; adds complexity |

## Optional: Weights and Biases

W&B is optional. The notebook must be complete without it.

```python
try:
    import wandb
    wandb.init(project="tamper-detection", config={...})
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
```

## Related Documents

- [05_Training_Strategy.md](05_Training_Strategy.md) — Training loop
- [12_Final_Submission_Checklist.md](12_Final_Submission_Checklist.md) — Pre-submission checks
