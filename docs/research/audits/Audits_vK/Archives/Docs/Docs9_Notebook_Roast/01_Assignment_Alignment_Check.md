# 01 — Assignment Alignment Check

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Required Deliverables vs Current State

The assignment requires one Google Colab notebook, executed, containing:

| Requirement | v8 run-01 | v9 Colab |
|-------------|-----------|----------|
| Dataset explanation | ✅ Present + executed | ✅ Present, **not executed** |
| Model architecture description | ✅ Present + code output | ✅ Present, **not executed** |
| Training strategy | ✅ Present with logs | ✅ Present, **no logs** |
| Hyperparameter choices | ✅ Printed in output | ✅ In CONFIG, no output |
| Evaluation results | ✅ Metric tables present | ❌ Missing — no run |
| Visual prediction outputs | ✅ 4-panel grid present | ❌ Missing — no run |
| Single Google Colab notebook | ⚠️ Kaggle variant — not Colab | ✅ Colab variant — not executed |
| Runnable top to bottom | ✅ Demonstrated | ❌ Not demonstrated |

---

## Detailed Findings

### 1. Dataset Explanation — Partial Pass (v9)

The v9 notebook includes solid dataset narrative: CASIA framing, mask format, pHash grouping rationale, split methodology. If executed, this would be excellent context.

The problem: no executed output cells confirm the dataset was actually loaded, no sample counts are printed, no split stratification table appears. The explanation is present as prose and dead code.

**v8 advantage:** The run-01 notebook printed actual file counts, confirmed pair validation passed, and showed split statistics in terminal output. Reviewers can read the numbers. v9 requires trust.

### 2. Model Architecture — Partial Pass (v9)

v9 correctly documents the architecture: U-Net/ResNet34, dual-task head, optional ELA 4th channel, edge-loss-aware loss stack. The design narrative is better than v8.

The problem: `smp.Unet` is never instantiated in an executed cell. No model summary, no parameter count, no sanity-check forward pass output exists. Any reviewer doing a proper check has to mentally simulate whether the ELA 4-channel pretrained weight adaptation is correct.

**Architecture risk:** ResNet34 pretrained on ImageNet uses 3-channel input. Switching to 4 channels requires either weight surgery on the first conv layer or a custom encoder. The notebook defines `in_channels = 4 if use_ela else 3` but the output confirming how SMP handles this mismatch is absent. This could silently use random initialisation on the modified first layer, invalidating the "ImageNet pretrained" claim.

**v8 advantage:** The model was instantiated and the parameter count was printed (`~24M parameters`). That output is in the notebook. v9 has zero runtime architecture confirmation.

### 3. Training Strategy — Fail (v9)

v9 documents optimizer (AdamW, differential LR), scheduler (ReduceLROnPlateau, patience 3), loss weights, accumulation steps. All legible in CONFIG.

The problem: there are no training logs. No epoch loss progression. No learning-rate curve. No indication that early stopping triggered. No evidence the model converged or diverged or crashed after 2 epochs.

**v8 advantage:** Training logs are embedded in the notebook. Epoch-by-epoch loss and metric values are visible. The threshold scan output is there. None of this exists in v9.

### 4. Hyperparameter Choices — Partial Pass (v9)

The CONFIG dict is well-structured and clearly documented. This is arguably the cleanest part of v9.

The problem: batch_size is 4. v8 used batch_size 64 on Kaggle 2×T4. v9 dropped to 4 for Colab T4 compatibility. With `accumulation_steps = 4`, the effective batch size is 16 — far below v8's effective 256. This is a consequential hyperparameter change that is not explained, not justified, and not tested.

**No documentation explains why 16 is adequate.** The prior run-01 baseline used effective batch 256. No ablation, no sensitivity analysis, just a silent downgrade.

### 5. Evaluation Results — Complete Fail (v9)

The v9 notebook has:
- IoU, Dice, Boundary-F1 functions defined → ✅
- Tampered-only evaluation pipeline → ✅
- Threshold grid search → ✅
- Per-forgery-type breakdown → ✅
- Mask randomization check → ✅
- Robustness suite → ✅
- **Actual numbers from any of the above → ❌ Zero**

There is not one numeric result in this notebook. Not one IoU value. Not one threshold table. Nothing. It is entirely definition without execution.

### 6. Visual Prediction Outputs — Complete Fail (v9)

The assignment explicitly requires visual results. v9 has `make_overlay()`, `visualize_predictions()`, `select_visualization_rows()` all correctly coded.

Zero images were generated. The `"outputs": []` on every cell is definitive. The reviewer cannot see a single predicted mask, a single overlay, a single comparison of ground truth vs prediction.

**This alone disqualifies v9 as a complete submission.** The assignment names this as a required deliverable. It does not exist.

---

## Overall Assignment Alignment Score

| Version | Score | Reason |
|---------|-------|--------|
| v8 run-01 | **7 / 10** | Executed with outputs, Kaggle-only (not Colab), some heuristic detection, good visualisations |
| v9 Colab | **2 / 10** | Colab-compatible code, better architecture design, **zero execution evidence** |

v9 is structurally better designed. It is assignment-worse because it has no demonstration.

---

## What Must Happen Before v9 Counts as a Submission

1. Execute the notebook end-to-end on a Colab T4.
2. Retain all cell outputs in the saved `.ipynb` file.
3. Confirm the ELA 4-channel weight adaptation produces valid output.
4. Confirm training converges to a reasonable tampered-only F1 (target ≥ 0.50).
5. Confirm visualisation cells produce at least 6 example grids.
6. Save the executed notebook to repository.

Until those six things happen, v9 is not a submission. It is a plan.
