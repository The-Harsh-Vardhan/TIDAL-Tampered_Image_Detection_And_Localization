# Engineering Practices

---

## Environment

| Property | Value |
|---|---|
| Primary runtime | Google Colab |
| Local fallback | Supported for non-Colab testing |
| Target GPU | T4 (compatible CUDA GPU acceptable) |
| Python | 3.10+ |
| Artifact storage | Google Drive if enabled, otherwise local artifact directory |

---

## Dependency Installation

```python
!pip install -q kaggle segmentation-models-pytorch "albumentations>=1.3.1,<2.0"
```

Notes:
- `kaggle` is required for Kaggle API dataset download.
- `segmentation-models-pytorch` and `albumentations` are the only required non-default dependencies.
- `wandb` is optional and installed only when `USE_WANDB = True`.
- `albumentations` is pinned to `>=1.3.1,<2.0` so parameters like `quality_lower`, `var_limit`, and `blur_limit` match the documented code.

---

## Artifact Directory

The notebook now supports either persistent Google Drive storage or a local fallback:

```python
USE_GOOGLE_DRIVE = True
LOCAL_CHECKPOINT_DIR = './artifacts/tamper_detection'
DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/tamper_detection/checkpoints'
```

Behavior:
- If `USE_GOOGLE_DRIVE = True` and Colab Drive is available, artifacts are written to Drive.
- If Drive is unavailable, the notebook falls back to the local artifact directory.
- If `USE_GOOGLE_DRIVE = False`, the notebook uses the local artifact directory directly.

Saved artifacts:
- `best_model.pt`
- `last_checkpoint.pt`
- `checkpoint_epoch_N.pt`
- `split_manifest.json`
- `results_summary_v5.json`
- `training_curves.png`
- `f1_vs_threshold.png`
- `prediction_grid.png`
- `gradcam_analysis.png`
- `robustness_chart.png`

---

## Kaggle Dataset Handling

The notebook avoids scanning all of `/content`. Instead it uses a slug-specific cache directory:

```python
DATASET_SLUG = 'sagnikkayalcse52/casia-spicing-detection-localization'
DATASET_CACHE_DIR = os.path.join('/content', DATASET_SLUG.split('/')[-1])
```

This reduces the risk of accidentally binding to a stale or unrelated `Image/` + `Mask/` tree left behind by another run.

---

## Reproducibility

### Seed setup

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

### Split persistence and reuse

- The stratified train/val/test split is persisted to `split_manifest.json`.
- On reruns, the notebook reloads the manifest when it is compatible with the currently discovered dataset.
- Manifest entries are stored as paths relative to `DATASET_ROOT`, which keeps the file stable across runs.

### Deterministic DataLoaders

```python
loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
```

This is applied to train/val/test loaders.

---

## Dataset Safety Checks

Before training, the notebook validates:
- tampered image readability
- mask readability
- image-mask dimension agreement
- authentic image readability
- unknown filename patterns (`unknown_forgery_type`) are excluded rather than silently stratified

These checks reduce late failures inside `__getitem__` and prevent split instability from rare unknown classes.

---

## Notebook Structure

The current notebook is [tamper_detection_v5.ipynb](/c:/D%20Drive/Projects/BigVision%20Assignment/notebooks/tamper_detection_v5.ipynb). It contains **17 sections / 61 cells**. See [12_Complete_Notebook_Structure.md](/c:/D%20Drive/Projects/BigVision%20Assignment/Docs5/12_Complete_Notebook_Structure.md) for the full section map.

High-level layout:

| # | Section | Purpose |
|---|---|---|
| 1 | Setup & Environment | Install, imports, seed, GPU, config, artifact dir, W&B guard |
| 2-5 | Dataset setup | Kaggle download, discovery, validation, split/manifest |
| 6-7 | Data pipeline | Dataset class, transforms, deterministic loaders |
| 8-10 | Core training | Model, loss/optimizer/metrics, threshold-aware training loop |
| 11-12 | Selection & evaluation | Recompute threshold on best checkpoint, test-set evaluation |
| 13-15 | Analysis | Visualizations, Grad-CAM, robustness testing |
| 16-17 | Tracking & export | W&B summary, results/artifact export |

---

## Code Style

- Functions are used for distinct concerns instead of monolithic cells.
- Runtime printouts verify counts, shapes, thresholds, and saved artifacts.
- `tqdm` progress bars are used for training, threshold sweeps, and evaluation.
- Optional integrations (`wandb`, Drive mount) are guarded.

---

## Excluded Tools

These remain out of scope:
- Multi-GPU / distributed training
- `torch.compile`
- DALI
- Docker / containerization
- Heavy explainability frameworks such as SHAP or LIME
