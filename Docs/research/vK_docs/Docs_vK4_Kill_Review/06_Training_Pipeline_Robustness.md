# 06 — Training Pipeline Robustness

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Reproducibility — ✅ Good

- `set_seed(42)` sets random, numpy, torch, and CUDA seeds
- `cudnn.deterministic = True`, `cudnn.benchmark = False`
- `seed_worker()` and seeded `Generator` for DataLoader
- Fixed `random_state=42` for train_test_split

**Caveat:** TF32 is enabled (Cell 5, lines 172-175: `torch.backends.cuda.matmul.allow_tf32 = True`), which **contradicts** `cudnn.deterministic = True`. TF32 introduces non-deterministic rounding. This is a subtle bug.

## Mixed-Precision (AMP) — ✅ Good

- `autocast('cuda')` wraps forward pass
- `GradScaler` handles loss scaling
- Correctly disabled during validation with `@torch.no_grad()`

## Gradient Accumulation — ⚠️ Bug

Cell 24, line 848: `loss = (ALPHA * loss_cls + BETA * loss_seg) / accum_steps`

The division by `accum_steps` is correct for gradient accumulation, **but** on line 859: `running_loss += loss.item() * accum_steps * images.size(0)` — the multiplication by `accum_steps` reverses the division for logging purposes. This is correct for the loss value, but the **gradient magnitudes** seen by `clip_grad_norm_` are accumulated across steps, meaning the effective max_grad_norm is `1.0 * accum_steps = 2.0`, not 1.0. The grad norm is unscaled before clipping (line 853: `scaler.unscale_`), but the accumulated gradients are larger.

## Checkpointing — ✅ Good

Full checkpoint saves model, optimizer, scaler, scheduler, and config. Checkpoint loading correctly uses `weights_only=False` for the dict.

**Gap:** No resume-from-checkpoint capability. If Kaggle disconnects mid-training, all progress is lost.

## Early Stopping — ✅ Good

Patience=10 on val F1. Counter resets on improvement.

## Gradient Clipping — ✅ Adequate

`clip_grad_norm_` at `max_norm=1.0` with `scaler.unscale_` before clipping.

## Memory Management — ⚠️ Weak

`torch.cuda.empty_cache()` at end of each epoch is somewhat helpful but is a band-aid. No `del` statements for intermediates. The `collect_predictions` function (Cell 33) stores ALL test images/masks/predictions in memory — for large test sets, this will OOM.

## DataLoader — ✅ Good

`persistent_workers=True`, `pin_memory=True`, `drop_last=True` on train, seeded generator.

**Gap:** `num_workers=4` is hardcoded in CONFIG, which may be too many for Kaggle (Kaggle kernels sometimes have restricted CPU count). No auto-detection.

---

## Summary Table

| Aspect | Status | Notes |
|---|---|---|
| Reproducibility | ⚠️ | TF32 contradicts deterministic mode |
| AMP | ✅ | Correctly implemented |
| Gradient Accumulation | ⚠️ | Grad norm effectively 2x intended |
| Checkpointing | ✅ | Full state saved, but no resume logic |
| Early Stopping | ✅ | Functional |
| Memory | ⚠️ | `collect_predictions` stores all in RAM |

**Severity: MEDIUM** — no showstoppers, but several subtle bugs.
