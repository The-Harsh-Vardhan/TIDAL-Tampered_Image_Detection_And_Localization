# Engineering Practices

---

## Environment

| Property | Value |
|---|---|
| Runtime | Kaggle Notebook |
| GPU | T4 (15 GB VRAM) |
| Python | 3.10+ |
| Dataset mount | `/kaggle/input/` (read-only) |
| Output storage | `/kaggle/working/` |

---

## Dependency Installation

```python
!pip install -q segmentation-models-pytorch "albumentations>=1.3.1,<2.0"
```

Notes:
- `segmentation-models-pytorch` and `albumentations` are the only required non-default dependencies.
- `wandb` is optional and installed only when `USE_WANDB = True`.
- `albumentations` is pinned to `>=1.3.1,<2.0` so parameters like `quality_lower`, `var_limit`, and `blur_limit` match the documented code.

---

## Output Directory Structure

```
/kaggle/working/
├── checkpoints/
│   ├── best_model.pt              # Best validation F1
│   ├── last_checkpoint.pt         # Latest epoch (resume)
│   └── checkpoint_epoch_10.pt     # Periodic saves (every 10 epochs)
├── results/
│   ├── split_manifest.json        # Train/val/test split metadata
│   └── results_summary.json       # Full metrics, config, robustness
└── plots/
    ├── training_curves.png
    ├── f1_vs_threshold.png
    ├── prediction_grid.png
    ├── gradcam_analysis.png
    └── robustness_chart.png
```

---

## Dataset Handling

The notebook discovers the dataset root dynamically under `/kaggle/input/` using case-insensitive directory matching. This accommodates variations in directory naming (`IMAGE/` vs `Image/`) and nesting (`New folder/`).

No download step is needed — Kaggle pre-mounts the dataset at `/kaggle/input/`.

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

### Split persistence

- The stratified train/val/test split is persisted to `split_manifest.json`.
- Data leakage verification ensures zero file overlap across splits.

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

Applied to train/val/test loaders via `worker_init_fn` and `generator` parameters.

---

## Dataset Safety Checks

Before training, the notebook validates:
- Tampered image readability (corruption guard)
- Mask readability
- Image-mask dimension agreement
- Authentic image readability
- Data leakage (set-intersection assertions across splits)

These checks reduce late failures inside `__getitem__` and prevent silent data issues.

---

## Notebook Structure

The current notebook is `tamper_detection_v5.1_kaggle.ipynb`. It contains **61 cells across 17 sections**. See `12_Complete_Notebook_Structure.md` for the full section map.

| # | Section | Purpose |
|---|---|---|
| 1 | Setup & Environment | Install deps, imports, seed, GPU check, config, W&B guard |
| 2 | Dataset Loading | Kaggle input discovery, directory structure summary |
| 3 | Dataset Discovery | Corruption guards, dimension checks, pair discovery |
| 4 | Dataset Validation | Counts, class balance, exclusion summary |
| 5 | Preprocessing & Split | Stratified 70/15/15, leakage verification, manifest |
| 6 | Dataset Class | `TamperingDataset` with mask binarization `> 0` |
| 7 | DataLoaders | Transforms, deterministic loader construction, batch sanity check |
| 8 | Model Definition | `smp.Unet(resnet34)` instantiation and shape check |
| 9 | Loss & Optimizer | BCEDiceLoss, AdamW differential LR, AMP scaler |
| 10 | Training Loop | Gradient accumulation, AMP, early stopping, checkpointing |
| 11 | Threshold Selection | Reload best checkpoint, validation sweep, confirm threshold |
| 12 | Evaluation | Mixed/tampered metrics, image-level accuracy/AUC |
| 13 | Visualization | Training curves, threshold plot, prediction grid |
| 14 | Explainable AI | Grad-CAM, diagnostic overlays, failure-case analysis |
| 15 | Robustness Testing | JPEG/noise/blur/resize evaluation and chart |
| 16 | Experiment Tracking | W&B integration summary |
| 17 | Save & Export | Results JSON, artifact inventory |

---

## Code Style

- Functions are used for distinct concerns instead of monolithic cells.
- Runtime printouts verify counts, shapes, thresholds, and saved artifacts.
- `tqdm` progress bars for training, threshold sweeps, and evaluation.
- Optional integrations (`wandb`) are guarded behind `USE_WANDB`.

---

## Excluded Tools

Out of scope for this assignment:
- Multi-GPU / distributed training
- `torch.compile`
- DALI
- Docker / containerization
- Heavy explainability frameworks (SHAP, LIME)
