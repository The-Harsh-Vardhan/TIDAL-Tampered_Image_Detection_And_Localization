# 09 — Recovery Plan

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

This document translates the audit findings into a prioritised action list. Items are sequenced so that each step unblocks the next.

---

## Prerequisite: Understand What You Have

Before writing any new code, execute this audit on the current v9 notebook:

1. Count how many cells have `execution_count: null` — the answer is all 14 of them.
2. Open v9 in Colab and run Cell 1 (imports). If it passes without error, continue. If it errors, fix dependencies first.
3. Run Cell 2 (dataset load). If paths fail, fix Colab Drive mounting before anything else.

The most common Colab bootstrap failure is a missing `drive.mount()` cell or incorrect path to the CASIA dataset. Fix this before attempting any of the substantive bugs below.

---

## Step 1 — Fix ELA Computation Order (Critical)

**Why first:** Every subsequent training run will produce garbage forensic channels if ELA is computed on normalised inputs. This must be fixed before any training attempt.

**Action:**

In `CASIADataset.__getitem__`, locate where ELA is computed relative to the normalisation transform. Move ELA computation **before** any torchvision normalisation. The correct sequence is:

```python
# 1. Load image as uint8 numpy array
img_np = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
img_np = cv2.resize(img_np, (IMAGE_SIZE, IMAGE_SIZE))

# 2. Compute ELA on the uint8 numpy array (pre-normalisation)
ela_map = compute_ela(img_np)   # returns [H, W, 1] float32 in [0, 1]

# 3. Apply augmentations to BOTH rgb and ela together
# (see Step 2 below for how to do this correctly)

# 4. Normalise RGB
img_tensor = F.normalize(torch.from_numpy(img_np).float() / 255.0, mean, std)

# 5. Concatenate rgb (3ch) + ela (1ch)
combined = torch.cat([img_tensor, ela_tensor], dim=0)
```

Verify `compute_ela()` internally uses PIL + JPEG save to a buffer + pixel diff, operating on uint8 values. If it does `img / 255.0` anywhere before the JPEG step, that is the bug location.

---

## Step 2 — Fix ELA-Augmentation Spatial Alignment (Critical)

**Why second:** After Step 1, ELA and RGB will be computed on the same spatial image, but augmentations may still misalign them.

**Action:**

Create a 4-channel numpy array before applying Albumentations:

```python
ela_map = compute_ela(img_np)             # [H, W, 1] uint8 or float
rgba_np = np.concatenate([img_np, ela_map], axis=2)   # [H, W, 4]

# Apply augmentations to 4-channel array
augmented = transform(image=rgba_np, mask=mask_np)
rgba_aug = augmented["image"]             # [H, W, 4]
mask_aug = augmented["mask"]

# Split channels after augmentation
img_aug = rgba_aug[:, :, :3]              # [H, W, 3]
ela_aug = rgba_aug[:, :, 3:4]            # [H, W, 1]
```

For Albumentations, geometric transforms (flip, rotate, shift-scale-rotate) apply identically to all channels when the input is a multi-channel `image=` argument. Verify that your normalisation transform applies only to the RGB channels (Albumentations normalisation operates on all channels by default — you may need to normalise manually after splitting).

---

## Step 3 — Add Post-Split Leakage Assertion (Quick Win)

**Why third:** This is a one-line addition and can be done before any training run. It will catch split bugs immediately at dataset construction time.

```python
train_paths_set = set(train_df["image_path"].values)
val_paths_set   = set(val_df["image_path"].values)
test_paths_set  = set(test_df["image_path"].values)

overlap_tv = train_paths_set & val_paths_set
overlap_tt = train_paths_set & test_paths_set
overlap_vt = val_paths_set & test_paths_set

assert len(overlap_tv) == 0, f"Train-val leakage: {len(overlap_tv)} paths"
assert len(overlap_tt) == 0, f"Train-test leakage: {len(overlap_tt)} paths"
assert len(overlap_vt) == 0, f"Val-test leakage: {len(overlap_vt)} paths"
print(f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

---

## Step 4 — Verify 4-Channel Encoder Weight Initialisation

**Why fourth:** If the ELA channel weights are zero-initialised, the model will effectively train as a 3-channel model for the first several epochs and the forensic signal will have weak early gradient flow.

**Action:**

After constructing the SMP model, add an inspection block:

```python
conv1_weight = model.encoder.layer0.conv1.weight   # adjust path to your SMP version
print("Conv1 weight shape:", conv1_weight.shape)   # expect [64, 4, 7, 7]
print("Channel 4 weight std:", conv1_weight[:, 3, :, :].std().item())
print("Channel 1 weight std:", conv1_weight[:, 0, :, :].std().item())
```

If channel 4 std is near zero (< 0.001), the 4th channel was zero-initialised. In that case, copy first-channel weights to the 4th channel:

```python
with torch.no_grad():
    conv1_weight[:, 3:4, :, :] = conv1_weight[:, 0:1, :, :].clone()
```

---

## Step 5 — Reduce Scope for First Successful Run

**Why fifth:** The goal at this point is to produce a notebook with real executed outputs. Reduce the training configuration to what is most likely to succeed on the first Colab run:

```python
CONFIG = {
    ...
    "use_edge_loss": False,       # disable until ELA bugs are confirmed fixed
    "use_dual_task": True,        # keep — simple to validate
    "run_multi_seed_validation": False,
    "run_architecture_comparison": False,
    "run_augmentation_ablation": False,
    "epochs": 10,                 # reduce from 30 for first run
    "batch_size": 16,             # increase from 4; add gradient accumulation if OOM
    "gradient_accumulation_steps": 4,  # restore effective batch ~64
    ...
}
```

The edge loss is the most experimental addition and adds the most complexity. Disable it for the first run. If BCE + Dice + cls_loss converges, re-enable edge loss in a second run.

---

## Step 6 — Execute Notebook End-to-End and Preserve Outputs

**Why this is the most important single step:**

```bash
# On Colab terminal or via nbconvert:
jupyter nbconvert --to notebook --execute \
    v9-tampered-image-detection-localization-colab.ipynb \
    --output v9-tampered-image-detection-localization-colab-executed.ipynb \
    --ExecutePreprocessor.timeout=86400
```

Or manually: open in Colab, Runtime → Run All, wait, then File → Save (Ctrl+S) to save outputs to Drive, then download the executed `.ipynb`.

**Minimum evidence required in the executed notebook before committing:**
- Cell with dataset statistics and example batch (images visible)
- Loss curve per epoch (numeric or chart)
- Val F1 at the best epoch
- At least 1 prediction grid with 4+ examples showing Original | GT | Prediction | Overlay

---

## Step 7 — Fix Boundary F1 Precision Edge Case

**Why seventh:** This is a minor fix but clean it up before final submission.

```python
# In boundary_f1() computation
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # was 1.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
```

---

## Step 8 — Cache pHash Results

**Why eighth:** Reduces restart overhead by 3–5 minutes per Colab restart.

```python
PHASH_CACHE_PATH = os.path.join(DRIVE_BASE, "phash_cache.json")

if os.path.exists(PHASH_CACHE_PATH):
    with open(PHASH_CACHE_PATH, "r") as f:
        hash_dict = json.load(f)       # {image_path: hash_hex_string}
    print(f"Loaded {len(hash_dict)} cached pHash values")
else:
    hash_dict = {}
    for path in all_image_paths:
        h = imagehash.phash(Image.open(path))
        hash_dict[path] = str(h)
    with open(PHASH_CACHE_PATH, "w") as f:
        json.dump(hash_dict, f)
    print(f"Computed and cached {len(hash_dict)} pHash values")
```

---

## Step 9 — Document batch_size Justification

If batch_size=4 remains in the final submission (e.g. due to 384×384 image memory constraints), add a CONFIG comment:

```python
"batch_size": 4,   # T4 16GB OOM at batch>8 with image_size=384 and 4-ch input
                   # effective batch ~16 with gradient_accumulation_steps=4
```

If gradient accumulation is re-added (recommended), document the effective batch size explicitly.

---

## Step 10 — Re-enable Edge Loss with Ablation Note

After Step 5's reduced-scope run converges, run a second experiment with edge loss re-enabled. Add a markdown cell comparing the two:

| Configuration | Val F1 (Seg) | Val Acc (Cls) | Epochs |
|---------------|-------------|---------------|--------|
| BCE+Dice+Cls (v9 baseline) | X.XX | X.XX% | 10 |
| + Edge Loss (λ=0.3) | X.XX | X.XX% | 10 |

This turns off a quality objection ("you added terms without evidence") and directly demonstrates whether edge loss helps.

---

## Recovery Priority Stack

| Priority | Action | Estimated Effort |
|----------|--------|-----------------|
| P0 | Fix ELA compute order (#2) | 20 minutes |
| P0 | Fix ELA-augmentation alignment (#3) | 45 minutes |
| P0 | Execute notebook, preserve outputs (#1, #7) | 12+ hours (training time) |
| P1 | Add leakage assertion | 5 minutes |
| P1 | Verify 4-channel weight init | 15 minutes |
| P1 | Reduce CONFIG scope for first run | 10 minutes |
| P2 | Fix Boundary F1 fill value | 5 minutes |
| P2 | Cache pHash results | 15 minutes |
| P2 | Document batch_size | 2 minutes |
| P3 | Edge loss ablation comparison | 12+ hours (second training run) |

**Minimum viable v9 submission:** P0 items only. Fix ELA bugs, run the notebook, save outputs. That promotion from never-executed to executed-with-real-outputs is the single most impactful change possible. Everything else is polish.

---

## What Good Looks Like

A passing v9 submission has:

1. Notebook opens in Colab with outputs visible
2. Dataset load cell shows `Train: ~8000, Val: ~1500, Test: ~1500` (or similar split)
3. Augmentation sample cell shows a batch of 8 augmented images with their masks — visual sanity check
4. Training loop cell shows per-epoch loss and F1 for 10–30 epochs
5. Evaluation cell shows test-set results as a table: `Pixel Acc | IoU | Dice | Boundary F1 | Cls Acc`
6. Prediction grid cell shows 8–12 examples with Original | GT | Prediction | Overlay quad-panels
7. Best model saved to Drive; path printed

v8 run-01 provides all of (1)–(7). v9 provides none of them in its current state. Close that gap.
