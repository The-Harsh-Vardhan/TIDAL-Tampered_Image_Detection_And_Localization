# 07 — Engineering Quality Check

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Summary

Setting aside the zero-execution problem (covered in 00), v9's code engineering has measurable improvements over v8. The config management is cleaner, the checkpointing is more principled, and the Colab path handling is explicit. But several engineering details are poorly handled, and two of them create reproducibility problems.

---

## CONFIG Management

### v8
Environment-specific paths are hardcoded inline across multiple cells. No single config dict. batch_size, image_size, and threshold appear scattered. A reviewer must hunt for them.

### v9
```python
CONFIG = {
    "batch_size": 4,
    "image_size": 384,
    "ela_quality": 90,
    "use_ela": True,
    "use_dual_task": True,
    "cls_loss_weight": 0.5,
    "use_edge_loss": True,
    "edge_loss_weight": 2.0,
    "edge_loss_lambda": 0.3,
    "run_primary_training": True,
    "run_multi_seed_validation": False,
    "run_architecture_comparison": False,
    "run_augmentation_ablation": False,
    ...
}
```

All hyperparameters in one place with explicit flags for optional experiments. This is substantially better than v8.

**One complaint:** `edge_loss_weight=2.0` and `edge_loss_lambda=0.3` are redundant. One of them should be removed. It is not clear which one is used in training. Code audit shows `edge_loss_lambda` appears in the loss calculation. The `edge_loss_weight` value in CONFIG has no obvious code reference. This creates confusion about which knob actually controls edge loss contribution.

---

## Checkpointing

### v8
Saves on every epoch improvement. No concept of periodic checkpoint. Outputs to `/kaggle/working/`.

### v9
```python
save_every_n_epochs: 5     # periodic safety checkpoints
# also saves best model by val_loss
checkpoint_dir: configured per run
```

Better: two checkpoint policies (periodic + best-model) reduce risk of losing good intermediate states on Colab disconnects. Colab sessions time out at ~12 hours — periodic checkpoints are especially important.

**No complaint on design.** The execution problem means these checkpoints were never actually saved.

---

## Weights & Biases Integration

v8 uses Kaggle Secrets to pull the W&B API key and logs training to W&B.

v9 sets `use_wandb=False` by default. The code to log to W&B is present but gated behind the flag.

**Impact:** No training curves were captured externally. If the notebook were run and then closed without saving, all loss curve information would be lost. Given that Colab outputs are ephemeral, this flag should default `True` or the notebook should persist logs to Drive.

---

## Google Colab Path Handling

```python
DRIVE_BASE = "/content/drive/MyDrive/BigVision Assignment"
os.makedirs(DRIVE_BASE, exist_ok=True)
```

This is the correct pattern for Colab. The drive-mount cell uses:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Path construction is consistent throughout. This is a genuine improvement over v8's Kaggle paths. No complaint.

---

## DataLoader Configuration

```python
DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    num_workers=2,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True,
    shuffle=True
)
```

`num_workers=2` with `persistent_workers=True` and `prefetch_factor=2` is a reasonable choice for Colab T4. It avoids the overhead of re-spawning workers every epoch.

`pin_memory=True` is correct with CUDA.

No complaint on DataLoader config. This is sensible.

---

## pHash Computation — No Caching

```python
def compute_phash_for_df(df):
    hashes = []
    for path in df["image_path"]:
        img = Image.open(path)
        hashes.append(imagehash.phash(img))
    return hashes
```

CASIA v2.0 contains ~12,000 images. pHash computation is not free. On Colab S disk, this loop over 12,000 image opens and hash computations takes roughly 3-5 minutes.

If the notebook is restarted (which happens frequently on Colab), this computation runs again. There is no cache file (e.g. `phash_cache.json`) written to Drive.

For a one-time assignment this is tolerable. But the union-find grouping then runs on top of these hashes, adding overhead to every restart. Correct fix is two lines:

```python
cache_path = DRIVE_BASE + "/phash_cache.json"
if os.path.exists(cache_path):
    hashes = json.load(open(cache_path))
else:
    # compute hashes
    json.dump(hashes, open(cache_path, "w"))
```

---

## Line Count Comparison

| Notebook | Lines | Executed Cells | Code Cells |
|----------|-------|----------------|------------|
| v8 run-01 | 16,976 | 20+ | ~35 |
| v9 Colab | 2,569 | 0 | 14 |

The 6.6× line difference is partly because v8 has large JSON output blocks from executed cells embedded in the `.ipynb`. But even the raw code content of v9 is substantially shorter than v8, because v9's code is better factored into functions (which are shorter to read) rather than procedural notebook cells (which are longer).

The low line count is not a quality problem. It is a warning sign that a reviewer might mistake for a shallow notebook before reading it.

---

## Notebook Generation via Script

`Notebooks/generate_v9.py` was committed to the repository. This script programmatically generates the v9 notebook from Python strings.

**Why this matters:** If a reviewer inspects the `.ipynb` file and notices `execution_count: null` on all cells, they may conclude the notebook is a draft. The generate script makes it clear that the notebook was intentionally constructed — but it was constructed as an unexecuted shell. The script does not trigger execution.

The engineering decision to generate a notebook from Python is unusual and adds complexity. The benefit — single source of truth, ability to diff code as plain Python — is real. But for an assignment submission, the output file must be produced and executed. The intermediate generation script is not useful to the evaluator.

---

## `generate_v9.py` Quality

A brief inspection:
- Uses `nbformat` to create notebook cells programmatically
- Cell source strings are triple-quoted Python
- No validation that the generated notebook actually runs

This is a reasonable approach for managing notebook content. The script should have been followed by `jupyter nbconvert --to notebook --execute`, which would have run the notebook and embedded outputs. That step was not taken.

---

## Engineering Quality Score

| Dimension | v8 | v9 | Verdict |
|-----------|----|----|---------|
| Config management | 4/10 | 8/10 | v9 better |
| Checkpointing | 6/10 | 7/10 | v9 marginally better |
| Path handling | 5/10 | 8/10 | v9 better (Colab vs Kaggle) |
| W&B integration | 7/10 | 4/10 | v8 better (actually logged) |
| DataLoader config | 6/10 | 7/10 | v9 marginally better |
| Reproducibility | 4/10 | 1/10 | v8 much better (it ran) |
| Deliverable completeness | 7/10 | 1/10 | v8 much better |

**Conclusion:** v9 has better code engineering in every category except the one that matters: it does not run and produce outputs. Good config management is table stakes. An executed notebook always beats a well-organised unexecuted one.
