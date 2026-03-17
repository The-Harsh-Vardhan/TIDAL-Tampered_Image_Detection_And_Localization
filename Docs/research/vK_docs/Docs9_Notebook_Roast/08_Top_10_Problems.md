# 08 — Top 10 Problems

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

Ranked by severity to a real submission review.

---

## #1 — Notebook Has Zero Executed Cells

**Severity:** Showstopper

**Every single code cell has `"execution_count": null` and `"outputs": []`.**

The v9 notebook is a code document, not a notebook in the sense that matters. A Jupyter notebook is evaluated by its outputs. Loss curves, prediction images, evaluation tables, accuracy numbers — all absent. The notebook was built but not run.

v8 run-01, which this was supposed to supersede, has 20+ executed cells with real outputs preserved. Any reviewer comparing the two will immediately notice that v9 represents a regression in deliverable quality.

**Fix:** `jupyter nbconvert --to notebook --execute v9-tampered-image-detection-localization-colab.ipynb --output v9-executed.ipynb`

---

## #2 — ELA Computed on Float-Normalised Arrays (Silent Correctness Bug)

**Severity:** High — corrupts the forensic signal

**ELA requires JPEG re-compression on uint8 pixel values (0–255).** It works by comparing each pixel to its JPEG-compressed version's value. The compression artefact magnitude is proportional to the absolute pixel value difference in uint8 space.

The v9 implementation loads images with torchvision transforms that include `Normalize(mean, std)`. If ELA is computed after normalisation, the input to the JPEG encoder is a float tensor with values like –2.1 (below 0) and +1.8 (above 1). The JPEG encoder will clip, wrap, or error on these values, depending on the underlying library.

Even if clipping happens silently, the ELA map computed on normalised floats has no meaningful relationship to the actual JPEG compression artefacts in the image. The 4th channel is garbage.

**Fix:** Compute ELA before normalisation, on the uint8 numpy array after resizing.

---

## #3 — ELA-Augmentation Spatial Misalignment

**Severity:** High — corrupts the training signal

During training, Albumentations augmentations are applied to the RGB image to produce a transformed tensor. If ELA is computed from the pre-augmentation image and concatenated to the post-augmentation RGB, spatial alignment is broken. A horizontal flip of the RGB will not be reflected in the ELA channel. The model receives contradictory spatial information at the 4-channel level.

This is a subtle bug because the numeric values in all 4 channels will be plausible-looking (no NaN, no crash), but the 4th channel will point to a different region than channels 1–3. The model cannot correctly learn to use ELA guidance.

**Fix:** Apply Albumentations transforms to the 4-channel concatenated array (or compute ELA from the already-augmented RGB image).

---

## #4 — batch_size 64 → 4 Without Justification (16× Throughput Regression)

**Severity:** High — regression vs v8

v8 used batch_size=64 with gradient_accumulation_steps=4, giving effective batch size ~256. v9 uses batch_size=4 with no gradient accumulation (not present in v9 code). Effective batch size: 4.

This is a 64× drop from the configured batch size and a 16× drop from the effective batch size.

At batch_size=4, BatchNorm statistics are noisy and loss estimates are high-variance. The model may converge more slowly or destabilise mid-training.

No comment in the code explains this decision. At Colab T4 (16GB VRAM), batch_size=64 with image_size=384 may be impractical (potential OOM). But if that is the reason, it should say so, and gradient accumulation should be re-introduced to preserve effective batch size.

**Fix:** Either add gradient accumulation steps (CONFIG["gradient_accumulation_steps"] = 16) or document why low batch size is acceptable.

---

## #5 — 4-Channel ResNet34 Weight Initialisation Strategy Undefined

**Severity:** High — non-reproducible if run

segmentation_models_pytorch (SMP) loads a ResNet34 encoder with pretrained ImageNet weights. ImageNet weights are for 3-channel input. The v9 notebook adds a 4th channel (ELA) and sets `in_channels=4`.

SMP's behaviour in this case is to either (a) zero-initialise the 4th channel weights, (b) average the first 3 channel weights and use that for channel 4, or (c) error. The actual behaviour depends on the SMP version.

The notebook does not specify which strategy is used, does not test for it, and does not document it. If SMP zero-initialises the 4th channel, the ELA input has no gradient signal in early training (the weight gradient through a zero-initialised channel is zero). This would mean the ELA channel contributes nothing for the first several epochs until the weights drift away from zero.

**Fix:** Explicitly initialise the 4-channel encoder:
```python
# Copy first-channel pretrained weights to 4th channel
encoder = model.encoder
with torch.no_grad():
    encoder.conv1.weight[:, 3:4, :, :] = encoder.conv1.weight[:, :1, :, :].clone()
```

---

## #6 — No Post-Split Data Leakage Assertion

**Severity:** Medium — reproducibility and correctness risk

The dataset is split into train/val/test using pHash-based grouping to prevent duplicate images from crossing the split boundary. This is good design. However, there is no assertion after the split confirming that no image path appears in more than one split.

Without this assertion, a bug in the union-find grouping (e.g. threshold set too loose, hashing collision, off-by-one in group assignment) would silently leak data from train into val or test.

**Fix:** Add a one-line assertion:
```python
assert len(set(train_paths) & set(val_paths) & set(test_paths)) == 0, \
    "Data leakage detected across splits"
```

---

## #7 — Zero Visual Outputs

**Severity:** Medium-High — assignment requirement directly not met

The assignment explicitly requires prediction overlays, ground truth comparisons, and visualisation of results. v9 defines sophisticated 5-column visualisation functions (Original | ELA | GT Mask | Prediction | Overlay). None of them were called. No output images exist.

This is distinct from #1 (executed cells) because it is the most visible requirement to a non-technical reviewer. The absence of images is noticed before the absence of training logs.

**Fix:** Run the notebook.

---

## #8 — Loss Function Has 4 Terms Added Simultaneously Without Ablation

**Severity:** Medium

```python
total_loss = (seg_bce + seg_dice) + cls_loss_weight * cls_loss + edge_loss_lambda * edge_loss
```

BCE + Dice + classification cross-entropy + boundary-weighted BCE — all active by default, all without a prior experiment comparing simpler combinations.

When training diverges or produces poor results, there is no baseline to diagnose which term is responsible. If edge_loss makes the model overfit to predicted boundaries in low-texture regions, there is no edge-loss-disabled checkpoint to compare against.

This is not wrong, but it is unverifiable without ablation runs. For a first submission, the simpler v8 loss (BCE + Dice only) was defensible and known to converge. Adding 2 more terms without evidence of improvement is risk accumulation.

---

## #9 — Boundary F1 Returns Precision=1.0 on Failed Prediction Edge Case

**Severity:** Medium

The custom Boundary F1 implementation computes precision as `TP / (TP + FP)`. When the predicted mask is all-zero (model always outputs "authentic"), `TP = 0` and `FP = 0`. The division `0/0` is handled with a fill value of 1.0 in the implementation.

This means precision=1.0 and recall=0.0, giving F1=0.0. The F1 result is correct. But precision=1.0 is logged to the per-image table. If any aggregation clips or logs precision independently, it reports 1.0 for examples where the model produced no prediction at all — a misleading artefact.

**Fix:** Use fill=0.0 for the precision division fallback when `TP + FP = 0`.

---

## #10 — pHash Recomputed on Every Run (No Caching)

**Severity:** Low-Medium

`compute_phash_for_df()` opens and hashes every image in the dataset on every notebook restart. CASIA v2.0 has ~12,000 images. On Colab T4 disk, this is a 3–5 minute overhead on every restart.

Colab sessions time out frequently. Every timeout means re-running this expensive computation before training can begin. The results are deterministic — the hash of an image never changes — so there is no reason to recompute them.

**Fix:** Cache hash results to a JSON file in Google Drive on first run, load from cache on subsequent runs.

---

## Summary Table

| # | Problem | Category | Severity |
|---|---------|----------|----------|
| 1 | Zero executed cells | Deliverable | Showstopper |
| 2 | ELA on normalised floats | Correctness | High |
| 3 | ELA-augmentation misalignment | Correctness | High |
| 4 | batch_size 64→4 undocumented | Performance | High |
| 5 | 4-channel weight init undefined | Reproducibility | High |
| 6 | No post-split leakage assertion | Correctness | Medium |
| 7 | Zero visual outputs | Deliverable | Medium-High |
| 8 | 4 loss terms without ablation | Methodology | Medium |
| 9 | Boundary F1 precision edge case | Implementation | Medium |
| 10 | pHash not cached | Engineering | Low-Medium |
