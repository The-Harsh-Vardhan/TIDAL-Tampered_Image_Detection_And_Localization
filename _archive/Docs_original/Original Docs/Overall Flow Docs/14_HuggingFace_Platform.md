# 14. Hugging Face as a Platform — Storage, Visualisation & Sharing Guide

## 14.1 What Is Hugging Face Hub?

Hugging Face Hub is a **Git-based platform** for hosting ML artifacts: models, datasets, and Spaces (interactive apps). Think of it as **GitHub for machine learning** — version-controlled, collaborative, and with built-in viewers for data, models, and demos.

### Key Components

| Component | What It Hosts | Why It Matters |
|-----------|--------------|----------------|
| **Model Hub** | Model weights + config + cards | One-line model loading; automatic inference widget |
| **Dataset Hub** | Datasets in Arrow/Parquet/file format | Standardised access; built-in data viewer |
| **Spaces** | Gradio/Streamlit apps | Live demos with GPU — no server management |
| **Organisations** | Team workspaces | Shared repos, access control, billing |

---

## 14.2 Why Use HF Hub for This Project?

### Storage & Sharing Benefits

| Benefit | Details |
|---------|---------|
| **Free storage** | Unlimited public repos; 50 GB for models, datasets |
| **Git LFS for large files** | Model weights (35 MB) tracked automatically |
| **Dataset viewer** | Auto-generates a table viewer — evaluators can browse data without downloading |
| **Model cards** | Standardised documentation template for your trained model |
| **One-line loading** | `model = AutoModel.from_pretrained("you/model-name")` |
| **Team sharing** | Create an org, add teammates — everyone can access, version, and comment |
| **Versioning** | Full git history for all assets — rollback to any version |

### Concrete Use Cases for This Project

1. **Upload cleaned CASIA v2.0 dataset** → teammates/evaluators browse images in the viewer
2. **Upload trained model weights** → anyone can load with one line
3. **Upload prediction visualisations** → high-res images viewable in the browser
4. **Create a Space** → live Gradio demo for tampering detection (see Doc 19)

---

## 14.3 Setup

### Install & Authenticate

```python
!pip install -q huggingface_hub datasets

from huggingface_hub import HfApi, login

# Login (get token from https://huggingface.co/settings/tokens)
login(token="hf_YOUR_TOKEN")  # Or use interactive: login()

api = HfApi()
```

---

## 14.4 Uploading a Dataset

### Create a Dataset Repository

```python
# Create a new dataset repo
api.create_repo(
    repo_id="your-username/casia-v2-cleaned",
    repo_type="dataset",
    private=False  # Public for sharing
)
```

### Upload Processed Data

```python
from huggingface_hub import upload_folder

# Upload the entire processed dataset folder
upload_folder(
    folder_path="./data/processed",
    repo_id="your-username/casia-v2-cleaned",
    repo_type="dataset",
    commit_message="Upload cleaned CASIA v2.0 with train/val/test splits"
)
```

### Create a Dataset Card

```python
# Create README.md (dataset card) in the repo
DATASET_CARD = """
---
license: cc-by-4.0
task_categories:
  - image-segmentation
tags:
  - image-forensics
  - tampering-detection
  - casia
size_categories:
  - 1K<n<10K
---

# CASIA v2.0 — Cleaned & Split

## Description
Cleaned version of the CASIA v2.0 Image Tampering Detection Dataset.

### Changes from original:
- Removed 17 images with resolution misalignment
- Binarised masks (threshold > 128)
- Stratified train/val/test split (85/7.5/7.5)
- Paired all tampered images with their ground truth masks

## Dataset Structure
```
data/
├── train/
│   ├── images/       (Au + Tp images)
│   └── masks/        (binary masks; zeros for authentic)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Statistics
| Split | Authentic | Tampered | Total |
|-------|-----------|----------|-------|
| Train | 1,445 | 2,783 | 4,228 |
| Val   | 128 | 246 | 374 |
| Test  | 128 | 245 | 373 |

## Usage
```python
from huggingface_hub import snapshot_download
path = snapshot_download("your-username/casia-v2-cleaned", repo_type="dataset")
```
"""

api.upload_file(
    path_or_fileobj=DATASET_CARD.encode(),
    path_in_repo="README.md",
    repo_id="your-username/casia-v2-cleaned",
    repo_type="dataset"
)
```

### What Evaluators See
When they visit `https://huggingface.co/datasets/your-username/casia-v2-cleaned`:
- **Dataset card** rendered as rich markdown
- **File browser** with folder structure
- **Auto-generated data viewer** showing images + masks in a table
- **Download button** or `snapshot_download()` one-liner

---

## 14.5 Uploading a Trained Model

### Create a Model Repository

```python
api.create_repo(
    repo_id="your-username/tampering-detector-unet-effb1",
    repo_type="model",
    private=False
)
```

### Upload Model Weights + Config

```python
import json

# Save model config
config = {
    "architecture": "UNet",
    "encoder_name": "efficientnet-b1",
    "in_channels": 6,
    "classes": 1,
    "image_size": 512,
    "srm_preprocessing": True,
    "threshold": 0.45,  # Oracle threshold from validation
    "metrics": {
        "pixel_f1": 0.65,
        "pixel_iou": 0.48,
        "image_auc": 0.89,
    }
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

# Upload model weights
api.upload_file(
    path_or_fileobj="best_model.pt",
    path_in_repo="model.pt",
    repo_id="your-username/tampering-detector-unet-effb1",
    repo_type="model"
)

# Upload config
api.upload_file(
    path_or_fileobj="config.json",
    path_in_repo="config.json",
    repo_id="your-username/tampering-detector-unet-effb1",
    repo_type="model"
)
```

### Create a Model Card

```python
MODEL_CARD = """
---
license: mit
library_name: segmentation-models-pytorch
tags:
  - image-segmentation
  - forensics
  - tampering-detection
pipeline_tag: image-segmentation
---

# Tampering Detector — U-Net + EfficientNet-B1 + SRM

## Model Description
A forensic image tampering detection and localisation model trained on CASIA v2.0.
Uses SRM (Spatial Rich Model) noise preprocessing + U-Net segmentation architecture.

## Performance
| Metric | Value |
|--------|-------|
| Pixel-F1 | 0.65 |
| Pixel-IoU | 0.48 |
| Image AUC-ROC | 0.89 |

## Usage
```python
from huggingface_hub import hf_hub_download
import torch

# Download
weights_path = hf_hub_download("your-username/tampering-detector-unet-effb1", "model.pt")

# Load
model = TamperingDetector(encoder_name='efficientnet-b1')
model.load_state_dict(torch.load(weights_path, weights_only=True))
model.eval()
```
"""

api.upload_file(
    path_or_fileobj=MODEL_CARD.encode(),
    path_in_repo="README.md",
    repo_id="your-username/tampering-detector-unet-effb1",
    repo_type="model"
)
```

---

## 14.6 Uploading Visualisation Artifacts

Store prediction grids, training curves, and ROC plots:

```python
# Upload figures to the model repo for documentation
for fig_name in ['prediction_grid.png', 'training_curves.png', 'roc_curve.png']:
    api.upload_file(
        path_or_fileobj=fig_name,
        path_in_repo=f"figures/{fig_name}",
        repo_id="your-username/tampering-detector-unet-effb1",
        repo_type="model"
    )
```

Reference them in the model card:
```markdown
## Visual Results
![Predictions](figures/prediction_grid.png)
![Training Curves](figures/training_curves.png)
```

---

## 14.7 Team Collaboration with Organisations

### Create an Organisation
1. Go to `https://huggingface.co/organizations/new`
2. Name: `bigvision-forensics`
3. Add teammates by username

### Shared Repos
```python
# Create repos under org namespace
api.create_repo(
    repo_id="bigvision-forensics/casia-v2-cleaned",
    repo_type="dataset"
)
```

### Access Control
- **Read**: Anyone (public repos)
- **Write**: Org members
- **Admin**: Org owner
- Can make repos private for internal work

---

## 14.8 HF Hub vs. Google Drive

| Feature | HF Hub | Google Drive |
|---------|--------|-------------|
| Git versioning | ✅ Full history | ❌ Limited revision history |
| Dataset viewer | ✅ Auto-generated | ❌ Must download to view |
| Model card | ✅ Rich markdown with metadata | ❌ Just files |
| One-line loading | ✅ `from_pretrained()` / `snapshot_download()` | ❌ Manual download + mount |
| Team collaboration | ✅ Orgs, permissions, comments | ✅ Shared folders |
| Free storage | 50 GB (models/datasets) | 15 GB (total) |
| Colab integration | Requires pip install | Native (`drive.mount()`) |
| ML ecosystem signals | ✅ Tags, tasks, metrics, citations | ❌ Generic file storage |

### Recommendation
- Use **Google Drive** for checkpoints during training (instant mount in Colab)
- Use **HF Hub** for the finished model, cleaned dataset, and public sharing

---

## 14.9 Should You Use HF Hub for This Project?

| Factor | Assessment |
|--------|-----------|
| **Required by assignment?** | No |
| **Time to set up?** | ~15 minutes |
| **Evaluator impression** | Strong — shows ML ecosystem fluency |
| **Practical benefit** | Dataset viewer, model card, one-line loading |
| **Risk** | None — additive, doesn't change core notebook |

**Verdict: Yes, after the core notebook is done.** Upload your cleaned dataset and final model to HF Hub as a portfolio piece. Link it in the notebook's conclusion. It's a ~15 minute investment that demonstrates you think beyond a single notebook.
