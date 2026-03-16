# Technical Audit: vK.11.1 (Run 02)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vk-11-1-tampered-image-detection-and-localization-run-02.ipynb` |
| **Platform** | Kaggle, 2x Tesla T4 GPUs (31.3 GB VRAM) |
| **Cells** | 102 total (49 code, 53 markdown) |
| **Executed** | 49 of 49 code cells (100%) |
| **Training** | 14 epochs (early stopped), best at epoch 4 |
| **Status** | **FULLY EXECUTED -- MODEL FAILED TO LEARN** |

---

## 1. Notebook Overview

vK.11.1 Run-02 is the **third fully executed run** of the synthesis architecture (after vK.11.4 and vK.11.5). Despite its "vK.11.1" version label, it is functionally identical to vK.11.4/11.5 in all code that matters -- same AMP bug fix, same CONFIG downgrade (max_epochs=50, patience=10), same W&B enhancements.

**Relationship to other runs:**
- Uses vK.11.1's notebook template (102 cells, no Results Dashboard, no Executive Summary section)
- Has vK.11.4's code changes: AMP fix, CONFIG downgrade, W&B prediction logging
- Correctly labeled as "vK.11.1" (unlike vK.11.4/11.5 which say "vK.11.1" in Model Card but are actually separate versions)
- W&B correctly set to `vK.11.1` project/run names (run-01 had incorrectly used `vK.11.0`)

**Key differences from vK.11.1 Run-01 (unexecuted):**

| # | Change | Impact |
|---|--------|--------|
| 1 | `max_epochs`: 100 → 50 | **Harmful** -- reduced training budget |
| 2 | `patience`: 20 → 10 | **Harmful** -- premature early stopping |
| 3 | Edge loss AMP fix (`.float()` + `autocast(enabled=False)`) | **Critical fix** -- unblocks training |
| 4 | `ReduceLROnPlateau(verbose=True)` removed | Cosmetic |
| 5 | W&B project/run renamed from `vK.11.0` to `vK.11.1` | Correctness fix |
| 6 | W&B sample prediction logging every 5 epochs | Monitoring improvement |
| 7 | Comprehensive W&B artifact logging (threshold sweep, ELA, Grad-CAM, robustness) | Monitoring improvement |

**No architectural or loss function changes.** Same TamperDetector, same loss weights, same optimizer config.

### CONFIG

```python
CONFIG = {
    'image_size': 256,
    'batch_size': 8,         # auto-scaled to 32 for 2xT4
    'max_epochs': 50,        # DOWNGRADED from run-01's 100
    'patience': 10,          # DOWNGRADED from run-01's 20
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'accumulation_steps': 4, # effective batch = 128
    'encoder_freeze_epochs': 2,
    'max_grad_norm': 5.0,
    'seed': 42,
}
```

---

## 2. Dataset Pipeline Review

**Identical to all vK.11.x runs:**

| Parameter | Value |
|-----------|-------|
| Dataset | CASIA v2.0 Upgraded (Kaggle) |
| Total Images | 12,614 (7,491 authentic + 5,123 tampered) |
| Split | 70/15/15 stratified (seed=42) |
| Train / Val / Test | 8,829 / 1,892 / 1,893 |
| Image Size | 256x256 |
| Input Channels | 4 (RGB + ELA grayscale at quality=90) |

Data leakage check **PASSED** -- 12,614 unique paths confirmed. Same augmentation pipeline (7 transforms), same DataLoader configuration (batch=32, 4 workers, persistent).

### Verdict

No changes from prior runs. Pipeline is sound.

---

## 3. Model Architecture Review

**Identical to all vK.11.x runs:**

| Component | Detail |
|-----------|--------|
| Model | TamperDetector (SMP UNet + ResNet34 + FC classifier) |
| Input | 4 channels (RGB + ELA) |
| Parameters | 24,571,347 |
| Loss | 1.5 * Focal(cls) + 1.0 * [0.5*BCE + 0.5*Dice](seg) + 0.3 * Edge(seg) |
| Class weights | [0.8420, 1.2310] (balanced) |
| Multi-GPU | DataParallel across 2 GPUs |

### Verdict

Architecture unchanged. Same multi-objective loss that failed in vK.11.4, vK.11.5, and vK.12.0.

---

## 4. Training Pipeline Review

### Training Execution Summary

| Metric | vK.11.1-R2 | vK.11.4 (ref) | vK.11.5 (ref) | vK.12.0 (ref) |
|--------|-----------|---------------|---------------|---------------|
| Total epochs | **14** | 25 | 13 | 16 |
| Best epoch | **4** | 15 | 3 | 6 |
| Best val Dice (tam) | **0.1362** | 0.1412 | 0.1364 | 0.1412 |
| Initial train loss | 4.6094 | 4.6517 | 4.6402 | 4.5975 |
| Final train loss | 1.6950 | 1.8795 | 1.6958 | 1.7305 |

### Epoch-by-Epoch Training Dynamics

| Epoch | Train Loss | Val Acc | Val AUC | Val Dice(tam) | Notes |
|-------|-----------|---------|---------|---------------|-------|
| 1 | 4.6094 | 0.4091 | 0.6670 | 0.1246 | Encoder frozen |
| 2 | 1.6779 | 0.5650 | 0.7077 | 0.1254 | Encoder frozen |
| 3 | 1.6728 | 0.5159 | 0.7054 | 0.1269 | Encoder unfrozen |
| **4** | **1.6699** | **0.5285** | **0.6465** | **0.1362** | **BEST -- first unfrozen epoch to improve** |
| 5 | 1.6778 | 0.5761 | 0.6868 | 0.1346 | |
| 6 | 1.6788 | 0.4202 | 0.6850 | 0.1358 | |
| 7 | 1.6799 | 0.4360 | 0.6799 | 0.1338 | LR reduction ~here |
| 8 | 1.6820 | 0.4477 | 0.6430 | 0.1327 | |
| 9 | 1.6856 | 0.4133 | 0.6614 | 0.1332 | |
| 10 | 1.6868 | 0.4619 | 0.6861 | 0.1320 | LR reduction ~here |
| 11 | 1.6879 | 0.4186 | 0.6625 | 0.1345 | |
| 12 | 1.6932 | 0.4234 | 0.6317 | 0.1323 | |
| 13 | 1.6964 | 0.4434 | 0.6524 | 0.1300 | |
| 14 | 1.6950 | 0.4366 | 0.6483 | 0.1320 | Early stopped |

### Critical Observations

1. **Best epoch at 4** -- only 2 epochs after encoder unfreeze. Like vK.11.5 (epoch 3) and vK.12.0 (epoch 6), the model's best moment is shortly after unfreezing. This is now a 4-run pattern.

2. **Train loss increases from epoch 5 onward** (1.6778 → 1.6964). This is unusual -- the model is not even overfitting. It's actively getting worse at fitting the training set while validation metrics also degrade.

3. **Two LR reductions** confirmed by W&B summary (final encoder_lr=3e-05, decoder_lr=2.5e-04 -- both reduced 2x from initial by factor 0.5). The scheduler fired at approximately epochs 7 and 10 based on patience=3 from best epoch 4.

4. **Val AUC peaks at epoch 2 (0.7077, frozen) then declines** -- consistent with the encoder-unfreeze-destroys-features pattern seen in vK.11.5.

5. **Only 14 epochs** -- the shortest run in the synthesis series (vs 25 for vK.11.4, 13 for vK.11.5, 16 for vK.12.0). The model exhausted patience=10 quickly because it peaked so early.

### Verdict

**Same failure mode as all synthesis runs, but faster.** The model peaks moments after encoder unfreeze, then degrades for 10 consecutive epochs. This is the fourth independent confirmation that the synthesis architecture's encoder learning rate destroys pretrained features.

---

## 5. Evaluation Metrics Review

### Test Set Results (at default threshold 0.5)

| Metric | vK.11.1-R2 | vK.11.4 | vK.11.5 | vK.12.0 |
|--------|-----------|---------|---------|---------|
| **Accuracy** | **0.5235** | 0.4142 | 0.4194 | 0.4062 |
| **AUC-ROC** | **0.6550** | 0.6434 | 0.6466 | 0.5637 |
| Dice (all) | 0.0517 | 0.0537 | 0.0517 | 0.0537 |
| **Dice (tam)** | **0.1274** | 0.1321 | 0.1272 | 0.1321 |
| **IoU (tam)** | **0.0780** | 0.0825 | 0.0768 | 0.0825 |
| **F1 (tam)** | **0.1274** | 0.1321 | 0.1272 | 0.1321 |
| **Pixel-AUC** | **0.4482** | 0.4988 | 0.5215 | 0.4952 |

**At default threshold (0.5), vK.11.1-R2 has the WORST Tam-F1 of all synthesis runs** (0.1274, worse than vK.11.5's 0.1272 after threshold optimization). The pixel-AUC of 0.4482 is also the worst -- significantly below random chance, indicating the model's continuous probability outputs are anti-correlated with ground truth.

**Interestingly, image-level accuracy (0.5235) and AUC-ROC (0.6550) are the BEST in the synthesis series.** This suggests the classification head learned slightly better image-level features, but at the expense of segmentation -- reinforcing the multi-objective conflict hypothesis.

### Threshold Optimization

| Parameter | vK.11.1-R2 | vK.11.4 | vK.11.5 |
|-----------|-----------|---------|---------|
| Optimal threshold | **0.0500** | 0.4939 | 0.0500 |
| F1 at optimal | **0.1321** | 0.1321 | 0.1321 |

At the optimized threshold of 0.05 (the floor of the sweep range), F1 recovers to the universal 0.1321. Like vK.11.5, the model's sigmoid outputs are concentrated near zero, requiring an extremely low threshold to produce any overlap.

### Test Metrics at Optimal Threshold (0.05)

| Metric | Value |
|--------|-------|
| Dice (tam) | 0.1321 |
| IoU (tam) | 0.0825 |
| F1 (tam) | 0.1321 |

### Per-Forgery-Type Breakdown (at optimal threshold)

| Forgery Type | Dice | IoU | F1 |
|-------------|------|-----|-----|
| Splicing (509) | **0.1016** | 0.0614 | 0.1016 |
| Copy-move (260) | **0.1918** | 0.1239 | 0.1918 |

### Mask-Size Stratified Evaluation (at optimal threshold)

| Size Category | Count | Dice | F1 |
|--------------|-------|------|-----|
| Tiny (<2%) | 289 | **0.0190** | 0.0190 |
| Small (2-5%) | 190 | **0.0630** | 0.0630 |
| Medium (5-15%) | 171 | **0.1537** | 0.1537 |
| Large (>15%) | 119 | **0.4859** | 0.4859 |

### Shortcut Learning Detection

| Test | Baseline F1 | Modified F1 | Delta |
|------|------------|-------------|-------|
| Mask randomization | 0.1321 | 0.1321 | **0.0000** |
| Boundary erosion | 0.1321 | 0.1321 | **0.0000** |

### Robustness Testing (at threshold 0.05)

| Condition | F1 | Delta |
|-----------|-----|-------|
| Clean | 0.1321 | — |
| JPEG QF=70 | 0.1321 | 0.0000 |
| JPEG QF=50 | 0.1321 | 0.0000 |
| Noise σ=10 | 0.1321 | 0.0000 |
| Noise σ=25 | 0.1321 | 0.0000 |
| Blur k=3 | 0.1321 | 0.0000 |
| Blur k=5 | 0.1321 | 0.0000 |
| Resize 0.75 | 0.1322 | +0.0001 |

**Note:** Albumentations API deprecation warnings indicate `ImageCompression` and `GaussNoise` parameters were silently ignored. The transforms likely applied as identity operations.

### The Identical-Output Confirmation

Once again, after threshold optimization, all stratified metrics are **bitwise identical** to vK.11.4, vK.11.5, and vK.12.0:

| Metric | vK.11.1-R2 | vK.11.4 | vK.11.5 | vK.12.0 | Match? |
|--------|-----------|---------|---------|---------|--------|
| Splicing Dice | 0.1016 | 0.1016 | 0.1016 | 0.1016 | **ALL EXACT** |
| Copy-move Dice | 0.1918 | 0.1918 | 0.1918 | 0.1918 | **ALL EXACT** |
| Tiny Dice | 0.0190 | 0.0190 | 0.0190 | 0.0190 | **ALL EXACT** |
| Small Dice | 0.0630 | 0.0630 | 0.0630 | 0.0630 | **ALL EXACT** |
| Medium Dice | 0.1537 | 0.1537 | 0.1537 | 0.1537 | **ALL EXACT** |
| Large Dice | 0.4859 | 0.4860 | 0.4859 | 0.4859 | Within rounding |
| Shortcut delta | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **ALL EXACT** |

This is now the **fourth independent training run** converging to bitwise identical stratified metrics. The constant-output hypothesis is confirmed beyond any statistical doubt.

### Verdict

**Worst raw performance in the series (Tam-F1=0.1274 at default threshold), recovers to the familiar 0.1321 after optimization.** The pixel-AUC of 0.4482 is the lowest yet, confirming the model is increasingly anti-correlated with ground truth as more runs occur. The better classification metrics (Acc=0.5235, AUC=0.6550) confirm the multi-objective conflict: the classification head benefits at segmentation's expense.

---

## 6. Visualization Quality

### Executed Visualizations

| Visualization | Status | Notes |
|--------------|--------|-------|
| Training loss curve | Rendered | Shows flatline/divergence |
| Threshold sweep plot | Rendered | Flat F1 across thresholds |
| W&B sample predictions | Logged (every 5 epochs) | 2 tampered samples per log |
| ELA visualization | Logged to W&B | RGB vs ELA side-by-side |
| Grad-CAM | Logged to W&B | |
| Failure case analysis | Logged to W&B | |
| Robustness table | Logged to W&B | |
| Confusion Matrix + ROC/PR | **SILENT FAILURE** | Cell 62 executed (ec=31) but produced zero output |
| Model Card results table | Rendered | Shows all test metrics |

### Key Issue

Cell 62 (Confusion Matrix + ROC/PR Curves) has execution_count=31 but **zero outputs** -- no plots, no text, nothing. This is anomalous and suggests the plotting code failed silently or output was stripped during execution.

### W&B Integration

| Property | Value |
|----------|-------|
| Mode | **Online** (synced to wandb.ai) |
| Project | `vK.11.1-tampered-image-detection-assignment` |
| Run name | `vK.11.1-smp-resnet34-ela-seed42-kaggle` |
| Run ID | `t6czno53` |
| Run finished | Yes (clean) |

This is the only synthesis run with **online W&B** (all others were offline on Kaggle). This provides a live dashboard for reviewing training curves and predictions.

### Verdict

**Comprehensive W&B logging is the standout feature.** Nearly every visualization pushes to W&B, making this the most W&B-instrumented run in the series. The silent failure of the Confusion Matrix cell is the only notable gap.

---

## 7. Assignment Alignment Check

| Requirement | Status | Notes |
|-------------|--------|-------|
| **1. Dataset Selection** | PASS | CASIA v2.0 |
| **1. Data Pipeline** | PASS | Complete pipeline with ELA |
| **1. Augmentation** | PASS | 7 transforms |
| **2. Architecture** | PASS | TamperDetector documented |
| **2. Resource Constraints** | PASS | Kaggle T4 compatible |
| **3. Performance Metrics** | **FAIL** | Tam-F1=0.1274, pixel-AUC=0.4482 -- below random |
| **3. Visual Results** | **PARTIAL** | Confusion matrix cell produced no output |
| **4. Single Notebook** | PASS | Everything in one .ipynb |
| **4. Model Weights** | **FAIL** | Weights saved but model is non-functional |
| **Bonus: Robustness** | **FAIL** | Present but Albumentations API deprecated + constant output |
| **Bonus: Subtle Tampering** | **FAIL** | Per-type breakdown near-random |

### Verdict

Same structural compliance as all synthesis runs. Model performance is the worst in the series at default threshold. Online W&B logging does not compensate for a non-functional model.

---

## 8. Engineering Quality

### Strengths

| Aspect | Rating | Notes |
|--------|--------|-------|
| CONFIG centralization | A | Complete, documented |
| W&B integration | **A** | **Best in series** -- online, comprehensive artifact logging |
| Checkpoint system | A | 3-file strategy |
| Data leakage prevention | A | Explicit overlap verification |
| Evaluation suite | A | Threshold sweep, shortcut, robustness, per-type/size |
| AMP implementation | B+ | Edge loss bug fixed |
| Version labeling | **A** | Correctly uses "vK.11.1" everywhere (unlike other runs) |

### Weaknesses

| Issue | Severity | Detail |
|-------|----------|--------|
| Confusion Matrix cell silent failure | Medium | Cell 62 produced zero outputs despite execution |
| Model Card effective batch error | Low | States "effective batch = 32" when actual is 128 |
| Albumentations API deprecation | Medium | Robustness transforms silently fall back to identity |
| `train_dice` always 0.0 | Medium | Never computed, misleading in training history |
| Train loss divergence | Medium | Loss increases from epoch 5 -- model is unstable |

### Verdict

**Engineering quality matches vK.11.4.** The online W&B integration and correct version labeling are improvements. The silent Confusion Matrix failure is a minor gap. Overall infrastructure is professional-grade -- supporting a model that learned nothing.

---

## 9. Roast Section

**"The Fourth Experiment That Proved the First Three Weren't Flukes"**

Let's review. vK.11.4: Tam-F1=0.1321. vK.11.5: 0.1272. vK.12.0: 0.1321. And now vK.11.1 Run-02: 0.1274. Four training runs. Four different epoch counts (25, 13, 16, 14). Four different best epochs (15, 3, 6, 4). And at the end, the same bitwise-identical stratified metrics down to the fourth decimal place. Splicing Dice 0.1016. Copy-move Dice 0.1918. Tiny mask Dice 0.0190. Every. Single. Time.

At this point, the synthesis architecture has achieved something remarkable: **perfect reproducibility of failure**. The constant-output attractor in this loss landscape is so strong that it doesn't matter when you start, how long you train, or which GPU you get -- you end up in the same dead end. Four independent random walks through parameter space, all converging to the same constant prediction.

This run's pixel-AUC of 0.4482 is the new project low. Not just below random (0.50) -- meaningfully below. The model has learned a slight anti-correlation: the more likely a pixel is to be tampered, the LESS likely the model says it is. If you negated every pixel's probability, you'd get 0.5518 -- not great, but better than anything the model produces as-is.

The irony is that vK.11.1-R2 has the **best classification metrics in the synthesis series**: Acc=0.5235 (vs 0.4062-0.4194 for others) and AUC=0.6550 (vs 0.5637-0.6466). The classification head actually learned something. The segmentation head got worse. This is RC-1 from the comparison report in living color: the 1.5x-weighted Focal loss is winning the gradient tug-of-war, steering the encoder toward image-level features, away from pixel-level ones.

The train loss trajectory tells the same story from a different angle. After epoch 5, training loss actually **increases** -- 1.6778 to 1.6964 over 9 epochs. The model is getting worse at fitting the training set. That's not overfitting. That's not underfitting. That's the optimizer finding that minimizing one component of the multi-objective loss forces another component upward. Three losses pulling in different directions, the model stuck in the middle, slowly getting worse at everything.

The ReduceLROnPlateau fired twice (final lr encoder=3e-05, decoder=2.5e-04). It tried. It reduced the learning rate by 75% total. Nothing changed. The problem isn't the learning rate magnitude -- it's the direction. No step size will help when the gradient is pointing at a constant-output basin.

The W&B integration is the one bright spot: online sync, artifact logging, prediction visualizations, the works. It means someone can go to wandb.ai and watch, in beautiful interactive plots, exactly how the model converged to predicting nothing. The perfect crime scene documentation for a model that was dead on arrival.

The Confusion Matrix cell (cell 62) executed with ec=31 and produced... nothing. Zero outputs. Not an error, not a warning -- just silence. It's almost poetic. The model has nothing meaningful to confuse.

At this point, running a fifth synthesis experiment without changing the architecture, loss function, or hyperparameters would not be science. It would be archaeology -- digging through the same failure mode hoping to find something different. The constant-output attractor has been mapped. The loss landscape has been surveyed. The encoder unfreeze dynamics have been documented across 4 independent runs. The data is in. The synthesis architecture does not work.

**Score: 3.5/10** (Worst raw Tam-F1 in series, worst pixel-AUC in project history, but correct version labeling and best W&B integration)

---

## Summary

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Dataset Pipeline | **A-** | Identical to series, sound |
| Model Architecture | **B** | Unchanged, same multi-objective conflict |
| Training Pipeline | **C** | Epoch 4 peak → 10-epoch decline → early stop at 14 |
| Evaluation Metrics | **A** (code) / **F** (results) | Comprehensive suite proves model failure |
| Visualization | **B** | Best W&B logging, but Confusion Matrix silent failure |
| Assignment Alignment | **D** | Structurally complete, functionally non-compliant |
| Engineering Quality | **B+** | Online W&B, correct version labels |
| **Overall** | **3.5/10** | Fourth confirmation of synthesis architecture failure |

### Key Metrics vs Project History

| Run | Tam-F1 | Tam-IoU | Img Acc | AUC | Pixel-AUC |
|-----|--------|---------|---------|-----|-----------|
| v6.5 (best) | **0.4101** | 0.3563 | 0.8246 | 0.8703 | — |
| vK.10.6 | 0.2213 | 0.1554 | 0.8357 | 0.9057 | 0.7083 |
| v8 | 0.2949 | 0.2321 | 0.7190 | 0.8170 | — |
| vK.11.4 | 0.1321 | 0.0825 | 0.4142 | 0.6434 | 0.4988 |
| vK.12.0 | 0.1321 | 0.0825 | 0.4062 | 0.5637 | 0.4952 |
| vK.11.5 | 0.1272 | 0.0768 | 0.4194 | 0.6466 | 0.5215 |
| **vK.11.1-R2** | **0.1274** | **0.0780** | **0.5235** | **0.6550** | **0.4482** |

### What vK.11.1-R2 Proves

1. **The constant-output attractor is universal.** Four independent training runs with different trajectories converge to identical stratified metrics.
2. **The multi-objective conflict is real.** Best classification metrics (Acc, AUC) in the series, worst segmentation metrics (Pixel-AUC). The objectives directly compete.
3. **LR reduction does not help.** Two scheduler interventions reduced lr by 75% -- the model continued degrading.
4. **No more synthesis runs should be attempted without architectural changes.** The failure is structural, not stochastic.
