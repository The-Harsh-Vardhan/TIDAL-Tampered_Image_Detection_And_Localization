# Technical Audit: vK.7.5

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**Files:**
- `Run vK.7.5/vK.7.5 Image Detection and Localisation.ipynb`
- `Run vK.7.5/vK.7.5_run_output.ipynb`
- `Run vK.7.5/vK.7.5_temp_run.ipynb`
- `Run vK.7.5/wandb/offline-run-*/logs/debug.log`

---

## 1. Architecture

| Attribute | Value |
|---|---|
| Model | `UNetWithClassifier` — same custom U-Net |
| Parameters | ~15.7M (all trainable, **no pretrained weights**) |
| Input | 3-channel RGB, 256×256 |

**Same architecture. No change.**

---

## 2. Training Pipeline

| Parameter | Value |
|---|---|
| Optimizer | `Adam(lr=1e-4)` |
| Loss | Combined FocalLoss + BCE + Dice (α=1.5, β=1.0) |
| Scheduler | `CosineAnnealingLR` |
| Batch Size | 8 |
| Epochs | 50 |
| AMP | No |
| DataParallel | No |

---

## 3. Execution Status: INCOMPLETE

This run **did not complete training.** Evidence from W&B offline logs and output notebooks:

- The notebook appears to have been interrupted or timed out during execution
- `vK.7.5_run_output.ipynb` contains partial output cells
- `vK.7.5_temp_run.ipynb` is a stripped/temp version used for execution
- W&B logs show offline mode (no API key) with partial logging

**No valid final test metrics are available from this run.**

---

## 4. W&B Logs Analysis

Two W&B offline runs were created:
1. `offline-run-20260313_153137-clq8rh59` — first attempt
2. `offline-run-20260313_155613-21toyhcs` — second attempt

Both ran in offline mode and neither completed the full training loop.

---

## 5. What Was This Supposed To Be?

vK.7.5 appears to be an intermediate version between vK.7.1 and the vK.10.x series. Based on the source notebook:
- It retained the dual-block structure
- It added some documentation improvements over vK.7.1
- It was supposed to be a clean execution run

However, the execution failed to complete, making this run **scientifically invalid** — there are no final metrics to evaluate.

---

## 6. Roast

vK.7.5 is the run that wasn't. Three files exist (source, output, temp) and none of them contain a completed training loop. Two W&B offline runs started and neither finished. This is what happens when you try to run 50 epochs of a 15.7M-parameter model without proper checkpointing, resume logic, or platform awareness. The run simply expired on whatever platform it was running on, leaving behind fragments. It counts as zero data points in the experimental timeline.
