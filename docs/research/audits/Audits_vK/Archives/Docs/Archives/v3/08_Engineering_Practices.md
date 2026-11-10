# Engineering Practices

---

## Environment

| Property | Value |
|---|---|
| Runtime | Google Colab (free tier) |
| Target GPU | T4 (compatible CUDA GPU acceptable) |
| Python | 3.10+ (Colab default) |
| Persistent storage | Google Drive |

---

## Dependency Installation

```python
!pip install -q kaggle segmentation-models-pytorch albumentations>=1.3.1,<2.0
```

**Notes:**
- `kaggle` is required for Kaggle API dataset download. Colab does not always pre-install it.
- `segmentation-models-pytorch` and `albumentations` are the only non-default dependencies.
- `albumentations` is pinned to `>=1.3.1,<2.0` to ensure compatibility with documented parameter names (`quality_lower`, `var_limit`, `blur_limit`).
- Optional: `wandb` for experiment tracking (not installed by default; notebook works without it).

**Pre-installed in Colab (do not install):**
torch, torchvision, numpy, matplotlib, sklearn, cv2, PIL, tqdm, json, os, random, warnings

---

## GPU Verification

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Go to Runtime > Change runtime type > GPU.")
```

The notebook targets T4 but does not reject other CUDA GPUs. If a non-T4 GPU is assigned, training still proceeds.

---

## Kaggle API Setup

```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_key'
```

Or use Colab Secrets (recommended for security):
```python
from google.colab import userdata
os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
```

---

## Google Drive Persistence

```python
from google.colab import drive
drive.mount('/content/drive')
CHECKPOINT_DIR = '/content/drive/MyDrive/tamper_detection/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

Saved artifacts:
- `best_model.pt`, `last_checkpoint.pt`, periodic checkpoints
- `split_manifest.json`
- `results_summary.json`

---

## Reproducibility

```python
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Set at notebook start. Used for data split, weight initialization, and DataLoader shuffling.

---

## Notebook Structure

| # | Section | Purpose |
|---|---|---|
| 1 | Setup & Environment | Install, imports, seed, GPU, config |
| 2 | Dataset Download & Discovery | Kaggle API, dynamic pair discovery |
| 3 | Preprocessing & Data Split | Binarize masks, stratified split, manifest |
| 4 | Dataset Class & DataLoaders | TamperingDataset, transforms, loaders |
| 5 | Model Definition | smp.Unet instantiation, parameter count |
| 6 | Loss & Optimizer | BCEDiceLoss, AdamW, AMP scaler |
| 7 | Training Loop | Train, validate, checkpoint, early stop |
| 8 | Threshold Selection | Validation sweep |
| 9 | Evaluation | Test-set metrics, two-view reporting |
| 10 | Visualization | Training curves, prediction grid |
| 11 | Robustness Testing (Bonus) | Degradation evaluation |
| 12 | Experiment Tracking (Optional) | W&B logging |
| 13 | Save & Export | Results JSON, final checkpoint |

---

## Code Style

- Functions over monolithic cells.
- Clear section headers with markdown cells.
- Print statements for runtime verification (dataset counts, shapes, metrics).
- `tqdm` progress bars for training and evaluation loops.
- No unnecessary abstractions — one function per concern.

---

## Excluded Tools

These tools are explicitly out of scope for this project:

- HuggingFace Hub / Spaces / Gradio
- Databricks, DuckDB, DynamoDB
- NVIDIA DALI
- `torch.compile`, channels-last format
- Multi-GPU / distributed training
- Docker / containerization
