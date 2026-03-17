# Technical Audit: vK.12.0b (Run 01)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vk-12-0b-tampered-image-detection-and-localization-run-01.ipynb` |
| **Platform** | Kaggle, 2x Tesla T4 GPUs (31.3 GB VRAM) |
| **Cells** | 151 total (69 code, 82 markdown) |
| **Executed** | 53 of 69 code cells (76.8%) |
| **Crashed** | Yes -- `AttributeError: 'Tensor' object has no attribute 'astype'` at cell 108 |
| **Cells killed by crash** | 16 cells never ran (Sections 13-19) |
| **Training** | 16 epochs (early stopped), best at epoch 6 |
| **Status** | **CRASHED MID-EXECUTION -- MODEL FAILED TO LEARN (SAME AS vK.12.0)** |

---

## 1. Notebook Overview

vK.12.0b is a **re-run of vK.12.0** with a single code change: the removal of 2 lines from a visualization function. This introduced a new crash bug that replaced vK.12.0's `KeyError: 'true_mask'` with `AttributeError: 'Tensor' object has no attribute 'astype'` -- blocking the same downstream sections.

### Relationship to vK.12.0

| Aspect | vK.12.0 | vK.12.0b |
|--------|---------|----------|
| CONFIG | Identical | Identical |
| Architecture | Identical | Identical |
| Loss function | Identical | Identical |
| Training loop | Identical | Identical |
| Evaluation code | Identical | Identical |
| Cell count | 151 | 151 |
| Crash location | Cell 53 (`KeyError: 'true_mask'`) | Cell 108 (`AttributeError: 'Tensor'.astype`) |
| Crash severity | 42 cells blocked | 16 cells blocked |
| **Code difference** | Has `img.numpy()` guard | **Missing `img.numpy()` guard** |

The sole difference: in the "Difference Map and Contour Overlay" visualization (cell 108), vK.12.0 contained:

```python
if hasattr(img, 'numpy'):
    img = img.numpy()
```

vK.12.0b removed these 2 lines, causing `img` to remain a PyTorch Tensor when `.astype(np.uint8)` is called -- a NumPy-only method.

### What Got Further in vK.12.0b

Because the `KeyError: 'true_mask'` bug was apparently fixed (progressing from cell 53 in vK.12.0 to cell 108 in vK.12.0b), vK.12.0b executed **all evaluation sections (11-12)** completely, including precision/recall/pixel-accuracy metrics, all per-type breakdowns, shortcut testing, threshold optimization, and robustness testing -- none of which were blocked.

**However**, the new crash at cell 108 still blocked Sections 13-19 (failure analysis, Grad-CAM, enhanced robustness, inference speed, model card, reproducibility, inference demo).

### CONFIG

Identical to vK.12.0 / vK.11.4 / vK.11.5:

```python
CONFIG = {
    'image_size': 256,
    'batch_size': 8,         # auto-scaled to 32 for 2xT4
    'max_epochs': 50,
    'patience': 10,
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'accumulation_steps': 4, # effective batch = 128
    'encoder_freeze_epochs': 2,
    'max_grad_norm': 5.0,
    'seed': 42,
}
```

Note: Batch auto-scaled to **32** (unlike vK.12.0's 16), giving effective batch=128. This matches vK.11.4/11.5.

---

## 2. Dataset Pipeline Review

**Identical to all synthesis runs:**

| Parameter | Value |
|-----------|-------|
| Dataset | CASIA v2.0 Upgraded (Kaggle) |
| Total Images | 12,614 (7,491 authentic + 5,123 tampered) |
| Split | 70/15/15 stratified (seed=42) |
| Train / Val / Test | 8,829 / 1,892 / 1,893 |
| Image Size | 256x256 |
| Input Channels | 4 (RGB + ELA grayscale at quality=90) |

Data leakage check **PASSED**. Same augmentation pipeline, same DataLoader configuration.

### Dataset Exploration (from vK.12.0 code)

| Metric | Value |
|--------|-------|
| Mean mask coverage | 9.1% |
| Median mask coverage | 3.1% |
| Images with <2% coverage | 1,348 (37.6%) |
| Images with >15% coverage | 626 (17.5%) |

### Verdict

Pipeline unchanged and sound.

---

## 3. Model Architecture Review

**Identical to all synthesis runs:**

| Component | Detail |
|-----------|--------|
| Model | TamperDetector (SMP UNet + ResNet34 + FC classifier) |
| Input | 4 channels (RGB + ELA) |
| Parameters | 24,571,347 |
| Loss | 1.5 * Focal(cls) + 1.0 * [0.5*BCE + 0.5*Dice](seg) + 0.3 * Edge(seg) |
| Class weights | [0.8420, 1.2310] (balanced) |
| Multi-GPU | DataParallel across 2 GPUs |

### Verdict

Zero architectural changes. Same architecture that has now failed in 4 prior runs.

---

## 4. Training Pipeline Review

### Training Execution Summary

| Metric | vK.12.0b | vK.12.0 (ref) | vK.11.4 (ref) |
|--------|---------|---------------|---------------|
| Total epochs | **16** | 16 | 25 |
| Best epoch | **6** | 6 | 15 |
| Best val Dice (tam) | **0.1412** | 0.1412 | 0.1412 |
| Initial train loss | 4.6129 | 4.5975 | 4.6517 |
| Final train loss | 1.6462 | 1.7305 | 1.8795 |

### Epoch-by-Epoch Training Dynamics

| Epoch | Train Loss | Val Acc | Val AUC | Val Dice(tam) | Notes |
|-------|-----------|---------|---------|---------------|-------|
| 1 | 4.6129 | 0.4059 | 0.6189 | 0.1221 | Encoder frozen |
| 2 | 1.6641 | 0.5090 | 0.6847 | 0.1224 | Encoder frozen |
| 3 | 1.6296 | 0.4059 | 0.6779 | 0.1132 | Encoder unfrozen -- **Dice DROPS** |
| 4 | 1.6252 | 0.4059 | 0.6428 | 0.1402 | Recovery |
| 5 | 1.6168 | 0.4434 | 0.6938 | 0.1265 | |
| **6** | **1.6010** | **0.4059** | **0.6035** | **0.1412** | **BEST** |
| 7 | 1.6333 | 0.4059 | 0.6141 | 0.1411 | |
| 8 | 1.6351 | 0.4059 | 0.6160 | 0.1408 | |
| 9 | 1.6385 | 0.4059 | 0.6020 | 0.1409 | |
| 10 | 1.6421 | 0.4059 | 0.5933 | 0.1406 | Checkpoint |
| 11 | 1.6436 | 0.4059 | 0.5985 | 0.1408 | |
| 12 | 1.6448 | 0.4059 | 0.5907 | 0.1408 | |
| 13 | 1.6454 | 0.4059 | 0.6155 | 0.1407 | |
| 14 | 1.6459 | 0.4059 | 0.5934 | 0.1405 | |
| 15 | 1.6470 | 0.4059 | 0.6083 | 0.1406 | |
| 16 | 1.6462 | 0.4059 | 0.5938 | 0.1406 | Early stopped |

### Critical Observations

1. **Val Accuracy locked at 0.4059 for 14 consecutive epochs (3-16).** The classification head has completely stalled -- it predicts the same class (likely "tampered") for every image. This is the most extreme classification collapse in the synthesis series.

2. **Best Dice at epoch 6** -- identical to vK.12.0's best epoch. Both 12.0 and 12.0b peaked at epoch 6 with val Dice(tam)=0.1412. This suggests the training trajectory is nearly deterministic for this CONFIG.

3. **Train loss increases from epoch 6 onward** (1.6010 → 1.6462). Same divergence pattern seen in vK.11.1-R2.

4. **Epoch 3 Dice DROP** (0.1224 → 0.1132). The moment the encoder unfreezes, segmentation Dice drops sharply before recovering at epoch 4. This is the encoder destabilization pattern.

5. **Val AUC degrades continuously** from epoch 5 (0.6938) to epoch 16 (0.5938). The classification loss is not just stalled -- it's getting worse.

### Verdict

**Training is nearly identical to vK.12.0** -- same best epoch, same best Dice, same stagnation pattern. The slight differences (initial loss 4.6129 vs 4.5975, final loss 1.6462 vs 1.7305) reflect batch size differences (32 vs 16) and GPU scheduling non-determinism.

---

## 5. Evaluation Metrics Review

### Test Set Results

| Metric | vK.12.0b | vK.12.0 | Delta |
|--------|---------|---------|-------|
| **Accuracy** | **0.4062** | 0.4062 | **0.0000** |
| **AUC-ROC** | **0.6175** | 0.5637 | **+0.0538** |
| Dice (all) | 0.0537 | 0.0537 | 0.0000 |
| **Dice (tam)** | **0.1322** | 0.1321 | +0.0001 |
| **IoU (tam)** | **0.0825** | 0.0825 | 0.0000 |
| **F1 (tam)** | **0.1322** | 0.1321 | +0.0001 |
| **Precision (tam)** | **0.0825** | 0.0825 | 0.0000 |
| **Recall (tam)** | **1.0000** | 1.0000 | 0.0000 |
| **Pixel Accuracy** | **0.0825** | 0.0825 | 0.0000 |
| **Pixel-AUC** | **0.4791** | 0.4952 | **-0.0161** |

**Recall = 1.0000, Precision = 0.0825** -- the model predicts EVERY pixel as tampered, identical to vK.12.0. The pixel-AUC of 0.4791 is worse than vK.12.0's 0.4952 and the second-worst in the series (after vK.11.1-R2's 0.4482).

The AUC-ROC improvement (+0.0538) and the Tam-Dice of 0.1322 (vs 0.1321) are within noise but noteworthy: 0.1322 breaks the exact 0.1321 match for the first time. This is likely due to the different effective batch size (128 vs 64).

### Threshold Optimization

| Parameter | vK.12.0b | vK.12.0 |
|-----------|---------|---------|
| Optimal threshold | **0.0500** | 0.5092 |
| F1 at optimal | **0.1321** | 0.1321 |

Despite vK.12.0b's optimal threshold being 0.05 (vs 12.0's 0.51), both converge to F1=0.1321. The radically different thresholds reflect different sigmoid output distributions, but the degenerate constant-output pattern remains.

### Per-Forgery-Type Breakdown (at optimal threshold)

| Forgery Type | vK.12.0b Dice | vK.12.0 Dice | Match? |
|-------------|--------------|-------------|--------|
| Splicing | 0.1016 | 0.1016 | **EXACT** |
| Copy-move | 0.1918 | 0.1918 | **EXACT** |

### Mask-Size Stratified Evaluation (at optimal threshold)

| Size Category | vK.12.0b Dice | vK.12.0 Dice | Match? |
|--------------|--------------|-------------|--------|
| Tiny (<2%) | 0.0190 | 0.0190 | **EXACT** |
| Small (2-5%) | 0.0630 | 0.0630 | **EXACT** |
| Medium (5-15%) | 0.1537 | 0.1537 | **EXACT** |
| Large (>15%) | 0.4859 | 0.4859 | **EXACT** |

### Shortcut Learning Detection

| Test | Baseline F1 | Modified F1 | Delta |
|------|------------|-------------|-------|
| Mask randomization | 0.1321 | 0.1321 | **0.0000** |
| Boundary erosion | 0.1321 | 0.1321 | **0.0000** |

### Robustness Testing

**NOT EXECUTED** -- blocked by the cell 108 crash. Same loss of robustness data as vK.12.0.

### Verdict

**Virtually identical to vK.12.0 across all metrics.** The fifth training run of the synthesis architecture produces the same constant-output pattern, the same bitwise identical stratified metrics, and the same degenerate precision/recall (Recall=1.0, Precision=0.0825). The 0.0001 difference in Tam-Dice (0.1322 vs 0.1321) is the first deviation from the universal constant, likely due to batch size differences.

---

## 6. The Crash: `AttributeError: 'Tensor' object has no attribute 'astype'`

### What Happened

The notebook crashed at **cell 108** (Section 12.3, "Difference Map and Contour Overlay") with:

```
AttributeError: 'Tensor' object has no attribute 'astype'
```

The `show_enhanced_viz()` function calls `img.astype(np.uint8)` on line 38, but `img` is returned from `denormalize()` as a PyTorch Tensor. The original vK.12.0 notebook had a guard:

```python
if hasattr(img, 'numpy'):
    img = img.numpy()
```

This guard was **removed** in vK.12.0b, introducing the crash.

### Comparison with vK.12.0's Crash

| Aspect | vK.12.0 | vK.12.0b |
|--------|---------|----------|
| Crash location | Cell 53 (Section 12.3) | Cell 108 (Section 12.3) |
| Error type | `KeyError: 'true_mask'` | `AttributeError: 'Tensor'.astype` |
| Root cause | Wrong dictionary key | Missing numpy conversion |
| Cells blocked | 42 (28%) | 16 (10.6%) |
| Evaluation sections blocked | Sections 11-19 | Sections 13-19 |
| Core eval (Section 11) | **Blocked** | **Not blocked** |

vK.12.0b's crash is significantly LESS severe -- it occurs much later in the notebook (cell 108 vs cell 53), meaning all core evaluation metrics (precision/recall, threshold optimization, per-type, shortcut, robustness basics) were computed before the crash.

### Impact Assessment

Blocked sections:

| Section | Content | Severity |
|---------|---------|----------|
| 13.1 | Failure case analysis | Medium |
| 13.2 | FP/FN error analysis | High -- new vK.12.0 feature |
| 14 | Grad-CAM heatmaps | Medium |
| 15 | Enhanced robustness testing | High -- new vK.12.0 feature |
| 16.1 | Inference speed benchmark | High -- new vK.12.0 feature |
| 17 | Model card | Low |
| 18 | Reproducibility verification | Medium |
| 19 | Quick inference demo | Low |

### Fix

One line: add `if hasattr(img, 'numpy'): img = img.numpy()` before the `.astype(np.uint8)` call. Trivial bug, moderate impact.

---

## 7. Visualization Quality

### Executed Visualizations

| Visualization | Section | Status | Notes |
|--------------|---------|--------|-------|
| Mask coverage histogram + CDF | 6.4 | Rendered | Dataset insight |
| Augmentation preview grid | 6.x | Rendered | 5-panel |
| Architecture diagram | 7.1 | Rendered | Programmatic matplotlib |
| torchinfo summary | 7.2 | Rendered (text) | Full layer breakdown |
| Training curves (4 subplots) | 11.2 | Rendered | Shows flatline pattern |
| Threshold sweep plot | 11.3 | Rendered | Flat F1 across thresholds |
| Confusion matrix | 11.5 | Rendered | |
| ROC curve | 11.5 | Rendered | |
| PR curve | 11.5 | Rendered | |
| Mask-size F1 bar chart | 11.7 | Rendered | |
| Experiment comparison table | 11.9 | Rendered (text) | **MISLEADING** baselines |
| 4-panel prediction grid | 12 | Rendered | Original / GT / Predicted / Overlay |
| ELA visualization | 12.2 | Rendered | RGB vs ELA side-by-side |

### Crashed / Never Rendered

| Visualization | Section | Status |
|--------------|---------|--------|
| 6-panel enhanced viz (diff maps, contours) | 12.3 | **CRASHED** |
| FP/FN error analysis | 13.2 | Blocked |
| Failure case gallery | 13.1 | Blocked |
| Grad-CAM heatmaps | 14 | Blocked |
| Enhanced robustness bar charts | 15 | Blocked |
| Inference benchmark | 16.1 | Blocked |

### Executive Summary / Results Dashboard

Still shows "Final test metrics have not been computed yet" -- same broken execution-order issue as all synthesis notebooks.

### Verdict

**Same visualization profile as vK.12.0** -- the 13 rendered visualizations are identical in structure. The crash moved later but still blocks the same enhanced viz features.

---

## 8. Assignment Alignment Check

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **1. Dataset Selection** | PASS | CASIA v2.0, with mask coverage analysis |
| **1. Data Pipeline** | PASS | Complete pipeline with ELA |
| **1. Augmentation** | PASS | 7 transforms, augmentation preview rendered |
| **2. Architecture** | PASS | Documented + architecture diagram + torchinfo |
| **2. Resource Constraints** | PASS | Runs on Kaggle T4 GPUs |
| **3. Performance Metrics** | **FAIL** | Tam-F1=0.1322, Recall=1.0, pixel-AUC=0.4791 |
| **3. Visual Results** | **PARTIAL** | Core predictions rendered; enhanced viz crashed |
| **4. Single Notebook** | PASS | Everything in one .ipynb |
| **4. Model Weights** | **FAIL** | Weights saved but model is non-functional |
| **Bonus: Robustness** | **FAIL** | Enhanced suite blocked by crash |
| **Bonus: Subtle Tampering** | **FAIL** | Per-type breakdown near-random |

### Verdict

Same structural compliance as vK.12.0. Slightly better coverage (crash at cell 108 vs 53), but model still non-functional.

---

## 9. Engineering Quality

### Strengths

| Aspect | Rating | Notes |
|--------|--------|-------|
| Dataset exploration | **A** | Mask coverage analysis |
| Model documentation | **A** | torchinfo + architecture diagram |
| Evaluation breadth | **A** | Precision/Recall/PixelAcc computed |
| CONFIG system | **A** | Comprehensive |
| Checkpoint system | **A** | 3-file strategy |
| Core evaluation execution | **B+** | Better than vK.12.0 -- all Section 11 metrics computed |

### Weaknesses

| Issue | Severity | Detail |
|-------|----------|--------|
| **`AttributeError` crash** | **HIGH** | New bug introduced by removing img.numpy() guard |
| **Hardcoded comparison baselines** | **HIGH** | Same misleading ~0.350 Tam Dice for "v11.x" |
| ExecutiveSummary placeholder | Medium | Same broken execution-order |
| Model Card says "vK.11.1" | Low | Not updated |
| W&B not finished | Medium | `wandb.finish()` cell never executed due to crash |
| No CONFIG changes | Medium | Same hyperparameters that failed 4 times before |

### W&B Integration

| Property | Value |
|----------|-------|
| Mode | **Offline** (no API key on Kaggle) |
| Run finished | **No** -- `wandb.finish()` cell blocked by crash |

### Verdict

**Engineering quality matches vK.12.0.** The crash bug is different but equally preventable -- a 2-line deletion in a visualization function that was never tested. The core evaluation sections ran to completion, making this the most diagnostically complete vK.12.0 variant.

---

## 10. Roast Section

**"They Fixed One Bug and Introduced Another, Then Got the Same Results"**

The pitch for vK.12.0b was simple: take vK.12.0, fix the `KeyError: 'true_mask'` crash that killed 42 cells, and get a clean run. What actually happened: someone fixed the KeyError (or at least got past it), then deleted two lines from a visualization function that convert a PyTorch Tensor to NumPy. The result? A new crash, `AttributeError: 'Tensor' object has no attribute 'astype'`, at cell 108 instead of cell 53. Progress, technically. The crash happens 55 cells later now. At this rate, they'll need about 3 more vK.12.0 variants to reach the end of the notebook.

To be completely fair, the crash is less severe. vK.12.0 lost 42 cells (28% of the notebook). vK.12.0b lost only 16 (10.6%). All the core evaluation metrics ran -- precision, recall, per-type, mask-size, shortcut tests, everything. So we got more data. And what does the data say?

Recall = 1.0000. Precision = 0.0825. Pixel-AUC = 0.4791. Tam-F1 = 0.1322.

Wait -- 0.1322? Not 0.1321? Stop the presses. After four runs producing exactly 0.1321, vK.12.0b managed to eke out 0.1322 -- a 0.0001 improvement. That's one ten-thousandth of a point. If this project were a stock, that would be a 0.008% gain. The secret? A different effective batch size (128 vs 64) because this Kaggle session happened to get a slightly different GPU memory allocation. The model didn't learn anything different. The arithmetic just rounded differently.

The training dynamics tell the most depressing story yet. Val Accuracy: 0.4059 for **14 consecutive epochs** (3 through 16). The classification head didn't just fail -- it became a literal constant. It outputs the same prediction for every image for nearly the entire training run. Meanwhile, val AUC starts at 0.6847 (epoch 2, frozen) and slides to 0.5938 by epoch 16. The encoder is being slowly lobotomized by the multi-objective loss.

And then there's the experiment comparison table, still claiming "v11.x ~0.350 Tam Dice, ~0.380 Tam F1." The actual v11.x Tam-F1 is 0.1272-0.1321. Those hardcoded values are 3x inflated. They were wrong in vK.12.0 and they're still wrong in vK.12.0b because nobody changed them. Nobody changed anything -- except accidentally deleting two lines from a visualization helper.

The W&B run wasn't properly closed because `wandb.finish()` is in cell 150 and the crash happened at cell 108. Somewhere in Kaggle's storage, there's an orphaned offline W&B run, forever incomplete, recording the exact coordinates of yet another failure.

This is the fifth run of the synthesis architecture. Five. Same CONFIG. Same architecture. Same loss function. Same dataset. Same results. At what point does "running the experiment again" stop being science and start being a ritual? The constant-output attractor has been mapped with cartographic precision. Five independent paths through parameter space, five arrivals at the same dead end, splicing Dice 0.1016, copy-move Dice 0.1918, shortcut delta 0.0000. The model has spoken. It has nothing to say.

**Score: 3.5/10** (Same failed model as vK.12.0, new crash replacing old one, but better evaluation coverage -- all Section 11 metrics completed)

---

## Summary

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Dataset Pipeline | **A** | Sound pipeline + mask coverage analysis |
| Model Architecture | **B** | Unchanged from vK.12.0 |
| Training Pipeline | **C** | Same failure: epoch 6 peak, 14-epoch accuracy lock at 0.4059 |
| Evaluation Metrics | **A** (code) / **F** (results) | All Section 11 metrics computed + confirm Recall=1.0 |
| Visualization | **B-** | 13 of 19 rendered, crash at cell 108 |
| Assignment Alignment | **D+** | Structurally complete, model non-functional |
| Engineering Quality | **B** | Strong eval, but new crash bug is preventable |
| **Overall** | **3.5/10** | Fifth confirmation of synthesis architecture failure |

### Key Metrics vs Project History

| Run | Tam-F1 | Tam-IoU | Img Acc | AUC | Pixel-AUC | Recall |
|-----|--------|---------|---------|-----|-----------|--------|
| v6.5 (best) | **0.4101** | 0.3563 | 0.8246 | 0.8703 | — | — |
| vK.10.6 | 0.2213 | 0.1554 | 0.8357 | 0.9057 | 0.7083 | — |
| v8 | 0.2949 | 0.2321 | 0.7190 | 0.8170 | — | — |
| vK.11.4 | 0.1321 | 0.0825 | 0.4142 | 0.6434 | 0.4988 | — |
| vK.12.0 | 0.1321 | 0.0825 | 0.4062 | 0.5637 | 0.4952 | 1.0000 |
| **vK.12.0b** | **0.1322** | **0.0825** | **0.4062** | **0.6175** | **0.4791** | **1.0000** |

### What vK.12.0b Proves

1. **The constant-output pattern persists across 5 runs.** Stratified metrics are bitwise identical to all prior runs.
2. **The `KeyError: 'true_mask'` crash was fixable but introduced a new bug.** One crash replaced by another -- the visualization code was not tested end-to-end.
3. **Classification head fully collapsed.** Val Accuracy locked at 0.4059 for 14 epochs -- the most extreme stagnation yet.
4. **No amount of re-running fixes a structural problem.** Same CONFIG + same architecture = same failure.
