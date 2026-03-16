# Audit 6.5 — Part 7: Engineering Review

## Overall Engineering Quality: Strong (B+)

The v6.5 notebook demonstrates professional ML engineering practices that significantly exceed typical Colab notebook quality. The engineering improvements over v6 are well-documented and genuinely useful.

---

## Configuration System ✅ Excellent

```python
CONFIG = {
    'image_size': 384,
    'batch_size': 4,
    'num_workers': 2,
    'train_ratio': 0.70,
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'max_epochs': 50,
    'patience': 10,
    'use_amp': True,
    'use_multi_gpu': True,
    'use_wandb': True,
    # ...
}
```

**Strengths:**
- Single source of truth for all hyperparameters and feature flags
- Passed to W&B for experiment tracking
- Config-driven DataLoader setup
- Feature flags (`use_amp`, `use_multi_gpu`, `use_wandb`) provide clean on/off control

**Weaknesses:**
- No config validation (e.g., `image_size` must be divisible by 32 for U-Net)
- No config serialization beyond W&B and results summary
- Would benefit from `dataclass` or `Namespace` for type safety, though overkill for a notebook

---

## Multi-GPU Support ✅ Good

```python
def setup_model(config, device):
    # ...
    if torch.cuda.device_count() > 1 and config['use_multi_gpu']:
        model = torch.nn.DataParallel(model)
    # ...
```

**Strengths:**
- Optional via feature flag
- `DataParallel` is appropriate for notebook context (single-process)
- Checkpoint save/load handles `module.` prefix transparently
- Shape verification after wrapping

**Weaknesses:**
- `DataParallel` is known to be slower than `DistributedDataParallel` for multi-GPU training
- With batch_size=4 split across 2 GPUs, each GPU processes only 2 images — this leads to noisy BatchNorm statistics
- The engineering note in the markdown correctly explains the DDP tradeoff

**Observation:** The Kaggle run detected 2 GPUs and used DataParallel. This is fine for the assignment scope but should be noted as a production limitation.

---

## Mixed Precision Training ✅ Good

```python
scaler = GradScaler('cuda', enabled=CONFIG['use_amp'])
# ...
with autocast('cuda', enabled=use_amp):
    logits = model(images)
    loss = criterion(logits, masks) / accum_steps
scaler.scale(loss).backward()
```

**Strengths:**
- Controlled by feature flag
- `GradScaler` gracefully disabled when `use_amp=False`
- AMP used consistently in training, validation, and inference
- Scaler state saved in checkpoints for seamless resumption

**Weaknesses:**
- No AMP-specific warnings or diagnostics (e.g., NaN detection)
- Loss scaling feedback not logged (could catch numerical issues early)

---

## DataLoader Optimization ✅ Good

```python
loader_kwargs = dict(
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)
```

**Strengths:**
- Config-driven worker count
- `pin_memory=True` for GPU training
- `persistent_workers=True` to avoid worker restart overhead
- `drop_last=True` for training (prevents small last batch)
- Seeded worker init function for reproducibility

**Weaknesses:**
- `num_workers=2` is conservative for T4 (could likely use 4)
- No prefetch_factor tuning
- No `pin_memory_device` specification (minor, defaults correctly)

---

## Reproducibility ✅ Good

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Strengths:**
- Full seed chain (Python, NumPy, PyTorch CPU, PyTorch CUDA)
- `cudnn.deterministic = True` for reproducible operations
- Seeded DataLoader workers via generator and `worker_init_fn`
- Split manifest saved in JSON

**Contradiction:**
- `set_seed()` sets `cudnn.benchmark = False`
- `setup_device()` then sets `cudnn.benchmark = True`

These are called sequentially:
```python
set_seed(SEED)         # benchmark = False
device = setup_device(CONFIG)  # benchmark = True (overrides!)
```

**This means `cudnn.benchmark = True` is the effective setting**, which improves performance but sacrifices bit-level reproducibility. While the seed still ensures the same sequence of operations, the cuDNN autotuner may select different algorithms across runs. This is a **minor bug** — the seed function's intent to set `benchmark=False` is silently overridden.

---

## Training Pipeline ✅ Good

**Strengths:**
- Extracted `train_one_epoch()` and `validate_model()` as clean functions with docstrings
- Gradient accumulation correctly implemented (loss divided by `accum_steps`, flush on partial final window)
- Gradient clipping (`max_norm=1.0`) prevents explosion
- Early stopping on validation F1
- W&B logging per epoch

**Weaknesses:**
- **No learning rate scheduler** — this is the single biggest engineering/training omission. All modern training pipelines use at least `ReduceLROnPlateau` or `CosineAnnealingLR`.
- Training history not saved in checkpoints (can't resume with full history)
- No warmup phase for learning rate

---

## Evaluation Pipeline ✅ Excellent

**Strengths:**
- Threshold sweep on validation set (50 points)
- Separate evaluation of mixed-set vs tampered-only vs per-forgery-type
- Image-level metrics (accuracy, AUC-ROC) alongside pixel-level metrics
- Comprehensive visualization (prediction grid, Grad-CAM, diagnostic overlays, failure analysis)
- Robustness testing across 8 degradation conditions

**Weaknesses:**
- Mixed-set F1 is reported prominently but is misleading (see Part 2)
- Precision/recall not reported per forgery type

---

## Code Quality

| Aspect | Rating | Notes |
|---|---|---|
| Documentation | ✅ Excellent | Markdown sections, docstrings, inline comments |
| Function design | ✅ Good | Clear signatures, single responsibility |
| Error handling | ⚠️ Adequate | Assertions for critical checks, try/except for Grad-CAM |
| Modularity | ✅ Good | Reusable functions, clean separation |
| Naming | ✅ Good | Descriptive variable and function names |
| Artifact management | ✅ Excellent | Final inventory check with OK/MISSING status |

---

## Summary

| Category | Grade | Key Issue |
|---|---|---|
| Configuration | A | Minor: no validation |
| Multi-GPU | B+ | DataParallel limitations with small batch |
| AMP | A- | No NaN diagnostics |
| DataLoaders | B+ | Conservative num_workers |
| Reproducibility | B+ | `cudnn.benchmark` contradiction |
| Training pipeline | B | **No LR scheduler** |
| Evaluation | A | Mixed-set metric prominence |
| Code quality | A- | Professional standard |

**Overall: B+** — Strong engineering with one critical omission (LR scheduler) and one bug (cudnn.benchmark override).
