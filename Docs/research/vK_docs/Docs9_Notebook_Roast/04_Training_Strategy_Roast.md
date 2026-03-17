# 04 — Training Strategy Roast

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Summary

The v9 training strategy is more sophisticated than v8 on paper and completely invisible in practice. The training loop is well-structured code. The training loop has zero executed outputs. This section reviews the design choices and explains which ones are actually risky.

---

## 1. Loss Function Stack

### v8 loss
```
total_loss = bce_loss(logits, mask) + dice_loss(logits, mask)
```
Two terms. Clear semantics. pos_weight computed from training data. Per-sample Dice.

### v9 loss
```
total = seg_total + 0.5 * cls_loss + 0.3 * edge_loss
```
Where `seg_total = bce_loss + dice_loss`.

Four effective terms. Three loss weight hyperparameters (pos_weight, cls_loss_weight, edge_loss_lambda). Two auxiliary objectives.

### Why this is risky without ablation

When training diverges or produces unexpectedly low metrics, you need to know which loss term is responsible. With 4 terms and no executed baseline to compare against, you have no way to diagnose the problem. If the v9 training loss does not decrease, is it:

- cls_loss dominating early training?
- edge_loss fighting the segmentation gradient direction?
- pos_weight interacting badly with the new balanced BCE?
- ELA channel gradient noise from random initialisation?

You do not know. You cannot know without either ablation experiments or at minimum an executed run.

The v9 approach of adding all improvements simultaneously is exactly how research projects lose interpretability. It is also exactly contrary to sound engineering process.

---

## 2. Batch Size: 4 vs 64

This is an elephant in the room that the notebook completely ignores.

| Parameter | v8 (Kaggle 2×T4) | v9 (Colab T4) |
|-----------|------------------|----------------|
| batch_size | 64 | 4 |
| accumulation_steps | 4 | 4 |
| effective batch | 256 | 16 |
| approximate throughput ratio | 1× | 0.0625× |

An effective batch size of 16 will produce significantly noisier gradient estimates than 256. The `pos_weight` BCE term is computed per-batch in v8. At batch size 4 with a class-imbalanced dataset (roughly equal authentic/tampered at image level, but pixel-level imbalance strongly favouring background), the per-batch pos_weight computed at 4 samples will be extremely noisy.

**The scheduler is also now operating on noisier validation signals.** `ReduceLROnPlateau` with patience 3 is fine at large batch sizes where validation metrics are stable. At small batch size + small val loader, metrics will fluctuate more and the scheduler will fire earlier, potentially reducing LR before training has converged.

None of this is acknowledged in the notebook. The batch size changed by 16× and nobody wrote a word about it.

---

## 3. Augmentation Strategy

### v8 augmentation
- Geometric: HFlip, VFlip, RandomRotate90
- Photometric: ColorJitter, ImageCompression (quality 70-90), GaussNoise, GaussianBlur

### v9 augmentation
Identical transforms. Same probabilities. No additions, no removals.

### The ELA augmentation problem

Here is the critical bug: v9 computes the ELA map AFTER all Albumentations transforms have been applied to the image. This means:

1. Image is loaded in BGR.
2. Augmentation runs: image may be horizontally flipped, rotated, jittered.
3. **ELA is computed from the augmented RGB** (after conversion from BGR).

Actually reading the code more carefully:

```python
if self.config["use_ela"]:
    ela_source_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ela_map = compute_ela(ela_source_bgr, self.config["ela_quality"])
```

`image_rgb` here is the **post-augmentation** image already converted by `A.Normalize`. Which means:

- ELA is computed on a JPEG-recompressed version of a normalised float image
- This is nonsensical. `cv2.imencode('.jpg', ...)` applied to a float image normalised with `(x - mean) / std` will produce garbage

This is a silent correctness bug. The ELA signal will be computed on normalised floating-point values that were never valid JPEG coefficients. The ELA values will be meaningless, but they will be concatenated to the tensor without any error, and training will proceed silently learning from garbage ELA features.

---

## 4. Optimizer Configuration

### AdamW with differential LR
```python
[
    {"params": encoder_params, "lr": 1e-4},
    {"params": decoder_params, "lr": 1e-3},
]
weight_decay = 1e-4
```

This is the same as v8. Correct choice, no regression here.

### Gradient clipping
`max_norm = 1.0` — same as v8. Fine.

### Gradient accumulation
`accumulation_steps = 4` — same logic as v8, but effective batch is now 16 instead of 256. The code is correct but the outcome is worse.

---

## 5. Scheduler

`ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)` — same as v8. Correct choice. Will fire based on `best_pixel_f1` validation metric.

### Risk
The validation set is 15% of 12,614 ≈ 1887 samples. At batch size 4, that is 472 validation steps per epoch. Fine. But if some of those batches have unusual composition (the pHash grouping means visually diverse images are in each split), per-epoch F1 may have higher variance than v8's simpler split.

---

## 6. Training Controls

v9 exposes these flags:
```python
"run_primary_training": True,
"run_multi_seed_validation": False,
"run_architecture_comparison": False,
"run_augmentation_ablation": False,
```

### Verdict on these flags

Having toggle flags is a good engineering pattern. The flags being set to False for the three research experiments is also correct prioritisation. The problem is that `run_primary_training = True` implies the notebook will train — and it has never trained. The flag says "yes, train" and the execution history says "never happened."

---

## Training Strategy Rating

| Aspect | v8 | v9 | Verdict |
|--------|----|----|---------|
| Loss design | Simple, validated | Complex, unvalidated | Regression risk |
| Batch size | 64 (appropriate) | 4 (too small, undocumented) | Regression |
| ELA augmentation | N/A | Correctness bug | Problem |
| Optimizer | AdamW diff-LR | AdamW diff-LR | Same |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau | Same |
| Training evidence | Logs present | None | Regression |
| Multi-seed | N/A | Defined, disabled | No improvement yet |

The training strategy regressed in practice (batch size) and introduced a possible silent corruption (ELA computed on normalised floats). The richer loss stack is directionally correct but empirically unvalidated.
