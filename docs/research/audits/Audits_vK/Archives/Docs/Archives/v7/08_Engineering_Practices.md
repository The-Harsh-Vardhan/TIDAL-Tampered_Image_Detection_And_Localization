# Engineering Practices

---

## Environment

| Property | Kaggle | Colab |
|---|---|---|
| GPU | T4 (15 GB VRAM) | T4 / V100 / A100 (varies) |
| Python | 3.10+ | 3.10+ |
| Dataset access | `/kaggle/input/` (read-only mount) | Google Drive + Kaggle API download |
| Output storage | `/kaggle/working/` | Google Drive |
| Secrets | Kaggle Secrets API | Colab Secrets (`userdata.get`) with getpass fallback |

---

## Dependency Installation

```python
!pip install -q segmentation-models-pytorch "albumentations>=1.3.1,<2.0"
```

- `segmentation-models-pytorch` and `albumentations` are the only required non-default dependencies.
- `wandb` is installed only when `CONFIG['use_wandb']` is True.
- `albumentations` is pinned to `>=1.3.1,<2.0` so parameters like `quality_lower`, `var_limit`, and `blur_limit` match the documented code.
- Colab additionally installs `kaggle>=1.6,<1.7` (pinned to avoid API header bugs) with `opendatasets` as a fallback download method.

---

## Hardware Abstraction

### `setup_device(config)`

Centralizes all hardware detection and GPU optimization into a single function:

```python
def setup_device(config):
    """Detect hardware, enable optimizations, and return the training device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True           # Optimized convolution kernels
        torch.backends.cuda.matmul.allow_tf32 = True    # TF32 for matmul (Ampere+)
        torch.backends.cudnn.allow_tf32 = True          # TF32 for cuDNN (Ampere+)
        # Reports GPU name, VRAM, count, AMP status, multi-GPU status
    return device
```

**Why centralize this?** Without `setup_device()`, GPU detection, benchmark flags, and TF32 settings would be scattered across multiple cells. Centralizing them ensures every run configures hardware consistently.

**TF32:** On Ampere and newer GPUs (A100, A10G), TF32 provides float32-level accuracy with approximately float16 speed for matrix multiplications. On older GPUs (T4 is Turing), the TF32 flags are silently ignored — no conditional needed.

**cuDNN benchmark:** Profiles convolution algorithms on the first batch and caches the fastest for subsequent batches. Effective when input sizes are fixed (all images resized to 384×384).

### `setup_model(config, device)`

Creates the model, optionally wraps in DataParallel, and verifies output shape:

```python
def setup_model(config, device):
    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights=config['encoder_weights'],
        in_channels=config['in_channels'],
        classes=config['classes'],
        activation=None,
    )
    model = model.to(device)

    is_parallel = False
    if torch.cuda.device_count() > 1 and config['use_multi_gpu']:
        model = torch.nn.DataParallel(model)
        is_parallel = True

    # Shape verification via dummy forward pass
    with torch.no_grad():
        dummy = torch.randn(1, 3, config['image_size'], config['image_size']).to(device)
        out = model(dummy)
        assert out.shape == (1, 1, config['image_size'], config['image_size'])

    return model, is_parallel
```

---

## Feature Flags

Three boolean flags in CONFIG control optional features:

| Flag | Default | Effect When False |
|---|---|---|
| `use_amp` | True | AMP disabled; GradScaler becomes no-op; all `autocast` calls disabled |
| `use_multi_gpu` | True | DataParallel not applied even with multiple GPUs |
| `use_wandb` | False | No W&B import, install, login, or logging |

**Why feature flags?** They decouple optional capabilities from the core pipeline. The notebook runs correctly with all three flags set to False — a critical property for environments without W&B accounts, without stable AMP support, or with single GPUs.

---

## Mixed Precision — Implementation Pattern

AMP is controlled everywhere through a single pattern:

```python
scaler = GradScaler('cuda', enabled=CONFIG['use_amp'])
```

When `enabled=False`, the scaler's methods (`scale()`, `unscale_()`, `step()`, `update()`) become no-ops. This eliminates all conditional branching in the training loop.

Every inference path uses the matching autocast:
```python
with autocast('cuda', enabled=config['use_amp']):
    logits = model(images)
```

**Functions using this pattern:**
- `train_one_epoch()` — training forward + backward
- `validate_model()` — validation forward
- `find_best_threshold()` — threshold sweep forward
- `evaluate()` — test evaluation forward
- `collect_predictions()` — visualization data collection
- `run_robustness_eval()` — robustness testing forward

---

## Data Pipeline Optimization

### Config-Driven DataLoaders

```python
_nw = CONFIG['num_workers']
loader_kwargs = dict(
    num_workers=_nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=_nw > 0,
)
```

- **`pin_memory`**: Auto-detected from hardware. Allocates tensors in pinned (page-locked) memory for faster GPU transfers.
- **`persistent_workers`**: Keeps worker processes alive between batches/epochs. Eliminates worker startup cost. Only active when `num_workers > 0`.
- **`drop_last=True`**: Applied to train loader only. Prevents a small final batch from disrupting gradient accumulation math.

### Deterministic Loading

```python
loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
```

Applied to all loaders via `worker_init_fn` and `generator` parameters. This ensures identical batch ordering and augmentation random states across runs.

---

## Output Directory Structure

```
/kaggle/working/
├── checkpoints/
│   ├── best_model.pt
│   ├── last_checkpoint.pt
│   └── checkpoint_epoch_10.pt
├── results/
│   ├── split_manifest.json
│   └── results_summary.json
└── plots/
    ├── training_curves.png
    ├── f1_vs_threshold.png
    ├── prediction_grid.png
    ├── gradcam_analysis.png
    └── robustness_chart.png
```

Artifact directories are created at notebook startup using `os.makedirs(exist_ok=True)`.

---

## Reproducibility

### Seed Setup

```python
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Note:** `cudnn.deterministic = True` is set in `set_seed()` for strict reproducibility, while `setup_device()` sets `cudnn.benchmark = True` for performance. The `setup_device()` call runs after `set_seed()`, so benchmark mode takes precedence during training. This is a deliberate trade-off: approximate reproducibility (same results ±0.1%) with faster training.

### Split Manifest

The stratified split is persisted to `split_manifest.json`. On subsequent runs, the notebook reloads the manifest when compatible, ensuring the same images go to the same splits.

### Checkpoint Resume

The notebook checks for `last_checkpoint.pt` at the start of training and resumes from the exact state (model, optimizer, scaler, epoch, best F1). This prevents wasted work on Kaggle (which has session time limits).

---

## Dataset Safety Checks

Before training, the notebook validates:

| Check | What It Catches |
|---|---|
| Image readability (`cv2.imread`) | Corrupted or truncated downloads |
| Mask readability | Missing or corrupted mask files |
| Image–mask dimension agreement | Mismatched annotation sizes |
| Unknown forgery type exclusion | Unclassifiable images |
| Data leakage assertions | Split contamination |

These checks prevent silent data issues that would corrupt training or evaluation.

---

## Checkpoint Portability

Checkpoints always save the unwrapped model state:

```python
state = model.module.state_dict() if is_parallel else model.state_dict()
```

Loading handles the `module.` prefix in both directions:

```python
state_dict = checkpoint['model_state_dict']
# Strip 'module.' prefix if present but not needed (or vice versa)
```

This means a checkpoint saved on a multi-GPU Colab instance can be loaded on a single-GPU Kaggle instance without modification.

---

## Code Style

- Functions for distinct concerns (`setup_device()`, `setup_model()`, `train_one_epoch()`, `validate_model()`)
- Runtime printouts verify counts, shapes, thresholds, and saved artifacts
- `tqdm` progress bars for training, threshold sweeps, and evaluation
- Optional integrations guarded behind CONFIG flags

---

## Excluded Tools

Out of scope for this assignment:
- `torch.compile` (requires PyTorch 2.0+ with compatible backend)
- NVIDIA DALI (GPU data pipeline — overkill for CASIA scale)
- Docker / containerization
- Heavy explainability frameworks (SHAP, LIME)
- Distributed training (DistributedDataParallel) — DataParallel is sufficient for ≤2 GPUs
