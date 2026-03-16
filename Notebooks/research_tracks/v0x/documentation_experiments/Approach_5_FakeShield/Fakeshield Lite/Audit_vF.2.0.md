# Audit Report: FakeShield-Lite vF.2.0

**Notebook:** `vf-2-0-fakeshield-lite-run-01.ipynb`
**Run Environment:** Kaggle T4 GPU (16 GB VRAM), PyTorch 2.9.0+cu126
**Run Date:** Extracted from Kaggle execution logs
**Auditor:** Claude (Automated Technical Review)

---

## 1. Run Summary

| Metric | Value |
|---|---|
| Total cells | 84 (43 code, 41 markdown) |
| Cells executed | 20 (cells 4–37) |
| Cells with output | 19 |
| **Last executed cell** | **Cell 37** (Feature Extraction Phase 1) |
| First unexecuted cell | Cell 39 |
| **Run outcome** | **CRASHED** — kernel killed at 0% of feature extraction |
| Training completed | NO |
| Evaluation completed | NO |
| Ablation experiments | NONE ran |

**Verdict: The run produced zero scientific results.** The notebook loaded the model and dataset successfully but crashed before any training occurred.

---

## 2. Architecture Review

### 2.1 What Works Well

| Component | Params | Status | Assessment |
|---|---|---|---|
| CLIP ViT-B/16 | 85.8M | Frozen | Correct — matches FakeShield DTE-FDM vision encoder |
| SAM ViT-B (encoder) | 89.7M | Frozen | Correct — matches FakeShield MFLM |
| SAM (prompt encoder) | 6.2K | Frozen | Correct |
| SAM (mask decoder) | 4.1M | Trainable | Correct — matches FakeShield `_train_mask_decoder` |
| Detection Head (MLP) | 197K | Trainable | Correct — simplified DTG |
| Feature Projection (MLP) | 787K | Trainable | Correct — replaces TCM LLM |
| **Total** | **180.5M** | **2.8% trainable** | Good pruning ratio |

The architecture is sound and correctly maps to the original FakeShield components. Per-image SAM encoding and per-image mask decoding correctly handle the batch-size issues discovered in vF.0.3–vF.0.4.

### 2.2 Architecture Concerns

1. **Full CLIP model download (599 MB) for vision-only use**: `CLIPVisionModel.from_pretrained()` downloads the full CLIP model (vision + text + projections), then only uses the vision part. The run log shows 164 "UNEXPECTED" keys for `text_model.*`, `logit_scale`, `visual_projection`, `text_projection`. This wastes bandwidth and potentially RAM.

2. **SAM encoder at 1024×1024**: Input images are 256×256, but SAM requires them resized to 1024×1024. This is correct per SAM's design, but the 4× upscaling is a major computational bottleneck.

3. **No gradient checkpointing on SAM decoder**: The mask decoder is small (4.1M), so this is acceptable, but worth noting for future scaling.

---

## 3. Training Pipeline Review

### 3.1 Pipeline Design (NOT TESTED — never executed)

- **Gradient accumulation**: BATCH_SIZE=4, ACCUM_STEPS=2 → effective batch 8. Correct implementation with `loss / ACCUM_STEPS` scaling.
- **Mixed precision**: `torch.amp.autocast("cuda")` + `GradScaler`. Uses updated API (good).
- **Differential LR**: Detection head + Feature projection at `1e-4`, SAM decoder at `5e-5`. Reasonable.
- **Scheduler**: CosineAnnealingLR over total steps. Correct.
- **set_to_none=True**: Good VRAM optimisation for `optimizer.zero_grad()`.
- **non_blocking=True**: Applied in data transfers. Good.

### 3.2 Training Pipeline Issues

1. **Scheduler steps per batch, not per epoch**: `scheduler.step()` is called after every accumulation step, inside `train_one_epoch`. The `T_max` is set to `NUM_EPOCHS * len(train_loader)`. This means the scheduler steps once per gradient update (every 2 batches), but `len(train_loader)` counts all batches. With ACCUM_STEPS=2, there are `len(train_loader) / 2` actual gradient updates per epoch. The `T_max` is therefore ~2x too large — the cosine schedule only reaches its minimum at epoch 40 instead of 20. **Impact: LR decays too slowly.**

2. **Frozen encoders not in eval mode during feature extraction (Cell 37)**: The `extract_features` function calls `model.eval()`, which is correct. But during normal training (Cell 44), only `clip_encoder.eval()` and `sam.image_encoder.eval()` are set. The `sam.prompt_encoder` is not explicitly set to eval mode. This is minor since it has no dropout/BN, but it's inconsistent.

3. **No gradient clipping**: The training loop has no `torch.nn.utils.clip_grad_norm_()`. For a pipeline with frozen encoders and small trainable heads, this is acceptable but could cause instability with the mask decoder's attention layers.

---

## 4. Feature Caching Pipeline — THE CRASH POINT

### 4.1 Root Cause Analysis

The run crashed at Cell 37 with the progress bar at **0% (0/2523 batches)**. This means it failed on the **very first batch**.

**Root cause: SAM encoder OOM during feature extraction.**

The `extract_features()` function processes images through both CLIP and SAM encoders:
```python
sam_input = model._prepare_for_sam(images)    # Resize to 1024×1024
sam_feat = model.sam.encode_image(sam_input)   # Per-image SAM encoding
```

With **BATCH_SIZE=4**, the function sends 4 images to the SAM encoder (one at a time, per-image loop). At this point:
- Model already occupies ~3.27 GB (from smoke test)
- CLIP encoder processes at 224×224 (small)
- SAM encoder processes at 1024×1024 (huge)
- Even with per-image encoding, the intermediate tensors and attention matrices consume several GB
- Both CLIP and SAM features are held in GPU memory simultaneously within the batch loop

Combined memory pressure exceeded T4's 16 GB, triggering a kernel kill.

### 4.2 Memory Budget Estimate

| Item | Memory |
|---|---|
| Model weights (float32) | 3.27 GB |
| SAM encoder forward pass (1 image at 1024×1024) | ~4-6 GB |
| CLIP encoder forward pass (1 image at 224×224) | ~0.3 GB |
| CLIP output + SAM output tensors accumulated | ~0.5 GB |
| PyTorch overhead + CUDA allocator | ~1-2 GB |
| **Total** | **~9-12 GB** (fits) |

Wait — this should fit. The issue may be more subtle:

**Alternative hypothesis: Kaggle session timeout.** Processing 10,091 images through SAM encoder one-by-one at ~0.5s/image = ~5,000 seconds ≈ 83 minutes just for training features. Plus validation features. The tqdm bar at 0% suggests the first batch hadn't even completed its timing estimate, meaning it may have been killed by Kaggle's session management before the first batch finished.

**Most likely cause: The feature extraction was running with `torch.no_grad()` but CLIP and SAM encoders were still computing gradients for non-frozen parts OR session timeout killed it.** Given the smoke test passed at 3.27 GB, OOM during the first batch is unlikely unless there's a memory leak. The 0% progress suggests the kernel was killed before even one batch completed — this points to **either immediate OOM from a bug or session timeout**.

### 4.3 Design Flaws in Feature Caching

1. **SAM feature size is enormous**: Each sample produces a (256, 64, 64) = 1,048,576 float16 tensor = 2 MB. For 10,091 training samples: **~20 GB of SAM features**. This cannot fit in CPU RAM on Kaggle (13 GB). Feature caching for SAM is fundamentally non-viable at this dataset size on Kaggle.

2. **CLIP features are tiny**: Each sample produces a (768,) float16 tensor = 1.5 KB. For 10,091 samples: **~15 MB**. CLIP feature caching is viable and would be useful.

3. **No disk-based caching**: Features are accumulated in CPU RAM (`all_sam.append(...cpu())`). No memory-mapped file or disk-based storage is used.

4. **Augmentation incompatibility**: Features are extracted with `val_transform` (no augmentation). This means ablation experiments using cached features cannot test the effect of augmentation on training — a fundamental flaw in the experimental design.

---

## 5. Data Pipeline Audit

### 5.1 Dataset

| Metric | Value |
|---|---|
| Total samples | 12,614 |
| Authentic | 7,491 (59.4%) |
| Tampered | 5,123 (40.6%) |
| Train/Val/Test | 10,091 / 1,261 / 1,262 |
| Imbalance ratio | 1.46:1 (moderate) |

The dataset is well-structured. All tampered images have corresponding masks.

### 5.2 Augmentation Issues

**CRITICAL: Albumentations API deprecation warnings:**

```
UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise
UserWarning: Argument(s) 'quality_lower, quality_upper' are not valid for transform ImageCompression
```

This means:
- `GaussNoise(var_limit=(5.0, 30.0))` — the `var_limit` parameter is silently ignored. GaussNoise applies with default parameters, which may not match intended behaviour.
- `ImageCompression(quality_lower=70, quality_upper=100)` — the `quality_lower/quality_upper` parameters are silently ignored. JPEG compression applies with default parameters.

**Impact: The augmentation pipeline is not performing GaussNoise or ImageCompression as intended.** Both transforms run with default parameters, which may be weaker or stronger than designed.

**Fix needed:** Update to the current Albumentations API:
- `GaussNoise(std_range=(0.02, 0.1), p=0.3)` (or `noise_scale_factor`)
- `ImageCompression(quality_range=(70, 100), p=0.3)`

### 5.3 Robustness Testing Augmentations (Cell 73 — never executed)

The robustness testing cell uses similar deprecated API calls in its `apply_distortion()` function. Same fixes needed.

---

## 6. Performance Bottlenecks

### 6.1 Critical Bottleneck: SAM Encoder

The **single biggest bottleneck** is SAM's image encoder running at 1024×1024 per image. This affects:
1. Feature extraction (crashed the run)
2. Normal training (would be slow at ~0.5s/image)
3. Validation and testing

**Estimated training time** (if run hadn't crashed):
- 10,091 train images × 20 epochs = 201,820 images
- At ~0.5s per SAM encoding: **28 hours** just for SAM encoder passes
- This exceeds Kaggle's session limits (12 hours for GPU sessions)

### 6.2 CLIP Weight Download Waste

The run downloads `pytorch_model.bin` (599 MB) which is the full CLIP model. Only the vision part (~340 MB) is used. The rest is discarded with "UNEXPECTED" key warnings.

### 6.3 DataLoader Configuration

`num_workers=2` with `persistent_workers=True` is reasonable for Kaggle. However, during feature extraction, `num_workers=0` is used — this is intentional but slow.

---

## 7. Evaluation Methodology

### 7.1 Metrics (defined but never computed)

- **Detection**: Accuracy, Precision, Recall, F1
- **Localization**: IoU, Dice
- **Localization evaluated only on tampered images**: Correct design — authentic images have no tampering to localize.

### 7.2 Evaluation Concerns

1. **No per-class detection metrics**: Only aggregate metrics are computed. Should separately report FPR and FNR.
2. **IoU/Dice edge case**: For `compute_iou`, when both `pred` and `target` are all-zeros (authentic image), it returns 1.0. This is correct since there's no tampering to detect. But this never triggers because localization is only evaluated on tampered images.
3. **No confidence calibration analysis**: The threshold sensitivity analysis (Cell 67) only varies the mask threshold, not the detection threshold.

---

## 8. Visualization Quality

The notebook contains well-designed visualizations:
- Dataset statistics (bar chart + pie chart)
- Sample images grid
- Augmentation preview
- Training curves (2×2 subplot with best-epoch annotation)
- Prediction visualization grid
- Ablation bar charts
- Threshold sensitivity plots
- Confidence distribution (histogram + box plot)
- Hard example grid

**All visualization cells are well-coded but NONE produced output** (training never ran).

---

## 9. Experimental Design (Ablation Study)

### 9.1 Ablation Framework

Five experiments designed:
1. **Baseline** — full model, no modifications
2. **No Detection Head** — set `w_det=0`
3. **No CLIP Projection** — zero and freeze FeatureProjection
4. **No Augmentation** — identical to Baseline (both use `val_transform`)
5. **Random SAM Decoder** — reinitialize decoder weights

### 9.2 Ablation Design Issues

1. **"No Augmentation" is meaningless**: Since all ablation experiments use cached features extracted with `val_transform`, there's no augmentation in ANY of them. The "No Augmentation" result is literally `ablation_results["Baseline"].copy()`. This ablation tests nothing.

2. **Ablation reloads full model each time**: `run_ablation()` creates a new `FakeShieldLite` which loads CLIP (599 MB download) and SAM (375 MB) from scratch. With 5 experiments, this means downloading CLIP 5 additional times. This is extremely wasteful.

3. **Ablation uses same CombinedLoss for all except "No Detection Head"**: The "No Detection Head" ablation manually creates a different criterion but still trains the detection head parameters. It should freeze the detection head entirely.

4. **No ablation for SAM encoder resolution**: A useful ablation would test SAM at 512×512 vs 1024×1024 to understand the resolution–performance trade-off.

5. **Ablation epochs too few**: 5 epochs with CosineAnnealing may not be enough to see meaningful differences. Early ablation results often show similar performance before components differentiate.

6. **Feature caching blocks augmentation ablation**: Since features are pre-extracted without augmentation, the ablation cannot test augmentation effects. This is a fundamental limitation of the caching approach.

---

## 10. The Roast

### Things That Are Actually Good
- Clean architecture that faithfully maps to FakeShield components
- Per-image SAM encoding and decoding — correct solution to batch-size issues
- Gradient accumulation implementation is correct
- Mixed precision training with updated PyTorch API
- Comprehensive visualization framework
- Good documentation structure

### Things That Need Honest Criticism

1. **The notebook has never produced a single trained model.** Versions vF.0.0 through vF.2.0 represent six iterations of development, yet not one Kaggle run has completed training. The feature caching system introduced in vF.2.0 made things worse by adding a mandatory pre-processing step that crashes before training even begins.

2. **Feature caching SAM outputs is memory-suicide.** SAM features at (256, 64, 64) × 10K samples × 2 bytes (float16) = **20 GB**. Kaggle has 13 GB CPU RAM. This was dead on arrival. A simple back-of-envelope calculation would have prevented this entire class of errors.

3. **The notebook is over-engineered for something that doesn't work.** There are 84 cells including ablation studies, threshold sensitivity, confidence distributions, hard example mining, robustness testing, computational efficiency analysis — none of which can produce results because the baseline training never completes. Building elaborate analysis infrastructure before getting the basics working is engineering in the wrong order.

4. **Albumentations warnings are silently ignored.** The deprecated `var_limit` and `quality_lower/quality_upper` parameters produce warnings that were visible in the run output, but nothing was done about them. This means the augmentation pipeline is not doing what the code suggests.

5. **The "No Augmentation" ablation is a copy-paste of baseline.** `ablation_results["No Augmentation"] = ablation_results["Baseline"].copy()`. This is not an experiment — it's pretending an experiment happened.

6. **Training time estimation was never done.** A quick estimate: 10K images × 20 epochs × 0.5s/SAM encode ≈ 28 hours. Kaggle T4 sessions are 12 hours max. The training loop as designed cannot physically complete on Kaggle.

---

## 11. Fixes Required for vF.3.0

### P0 — Critical (Must Fix)

| # | Issue | Fix |
|---|---|---|
| 1 | Feature extraction crashes (SAM features too large for RAM) | Remove SAM feature caching entirely. Only cache CLIP features (15 MB total). Run SAM encoder during training. |
| 2 | Training cannot complete in Kaggle time limit | Reduce to 10 epochs. Add estimated time-per-epoch print. Add early stopping. |
| 3 | Albumentations deprecated API | Update GaussNoise and ImageCompression to current API |
| 4 | Feature caching design fundamentally flawed | Redesign: cache only CLIP [CLS] tokens (tiny). SAM encoding runs live during training. |

### P1 — Important

| # | Issue | Fix |
|---|---|---|
| 5 | Ablation reloads CLIP 5 times | Share base model weights, only reinitialize trainable components |
| 6 | "No Augmentation" ablation is meaningless | Replace with a real ablation: train with augmented images vs val_transform only (no caching) |
| 7 | Scheduler T_max counts batches, not gradient steps | Set `T_max = NUM_EPOCHS * len(train_loader) // ACCUM_STEPS` |
| 8 | Full CLIP model download waste | Use `CLIPVisionModel.from_pretrained()` — already correct, but suppress the 164 UNEXPECTED key warnings |

### P2 — Nice to Have

| # | Issue | Fix |
|---|---|---|
| 9 | No gradient clipping | Add `clip_grad_norm_(max_norm=1.0)` |
| 10 | No early stopping | Add patience-based early stopping |
| 11 | No training time estimation | Print estimated completion time after epoch 1 |
| 12 | Hard example analysis processes test set one-by-one | Batch processing would be much faster |

---

## 12. Summary

**vF.2.0 is well-architected but non-functional.** The core model design is correct and maps faithfully to FakeShield. However, the feature caching system introduced in vF.2.0 is fundamentally flawed (20 GB SAM features cannot fit in 13 GB RAM), and the training time exceeds Kaggle's session limits. The notebook needs a pragmatic redesign that:

1. Drops SAM feature caching (keeps CLIP-only caching for ablations)
2. Reduces epochs and adds early stopping
3. Fixes the Albumentations API
4. Makes the ablation experiments meaningful
5. Gets training to actually complete on Kaggle T4

The priority must be: **make it run end-to-end before adding more analysis features.**
