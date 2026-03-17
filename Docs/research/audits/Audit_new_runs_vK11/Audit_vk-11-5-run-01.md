# Technical Audit: vK.11.5 (Run 01)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vk-11-5-tampered-image-detection-and-localization-run-01.ipynb` |
| **Platform** | Kaggle, 2x Tesla T4 GPUs (31.3 GB VRAM) |
| **Cells** | 135 total (61 code, 74 markdown) |
| **Executed** | 57 of 61 code cells |
| **Training** | 13 epochs (early stopped), best at epoch 3 |
| **Status** | **FULLY EXECUTED -- MODEL FAILED TO LEARN (WORSE THAN vK.11.4)** |

---

## 1. Notebook Overview

vK.11.5 is the **second fully executed run** of the synthesis architecture and a **near-exact copy of vK.11.4** with one structural addition: a Results Dashboard section (cells 8-15, 4 code + 4 markdown cells).

**Changes from vK.11.4:**
1. Added "Results Dashboard" section (cells 8-15) with Metrics Summary Table, Training Curve Visualization, and Example Localization panels
2. Updated Table of Contents to include Results Dashboard link
3. Updated title/conclusion version labels from "vK.11.4" to "vK.11.5"

**Zero code changes** to architecture, CONFIG, dataset, training, loss functions, or evaluation. The functional ML code is 100% identical to vK.11.4.

**Unexecuted cells** (4 of 61): Quick Inference Demo (3 cells) and W&B finish.

### CONFIG

Identical to vK.11.4:
```python
CONFIG = {
    'img_size': 256,
    'batch_size': 8,         # auto-scaled to 32 for 2xT4
    'max_epochs': 50,
    'patience': 10,
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'accumulation_steps': 4,
    'encoder_freeze_epochs': 2,
    'max_grad_norm': 5.0,
    'seed': 42,
}
```

---

## 2. Dataset Pipeline Review

**Identical to vK.11.4** in every respect:

| Parameter | Value |
|-----------|-------|
| Dataset | CASIA v2.0 Upgraded (Kaggle) |
| Total Images | 12,614 (7,491 authentic + 5,123 tampered) |
| Split | 70/15/15 stratified (seed=42) |
| Train / Val / Test | 8,829 / 1,892 / 1,893 |
| Image Size | 256x256 |
| Input Channels | 4 (RGB + ELA grayscale) |

Data leakage check **PASSED** -- zero overlap confirmed.

Same augmentation pipeline (7 transforms), same DataLoader configuration (batch=32, 4 workers, persistent).

### Verdict

No changes from vK.11.4. Pipeline is sound.

---

## 3. Model Architecture Review

**Identical to vK.11.4:**

| Component | Detail |
|-----------|--------|
| Model | TamperDetector (SMP UNet + ResNet34 + FC classifier) |
| Input | 4 channels (RGB + ELA) |
| Parameters | 24,571,347 |
| Loss | 1.5*Focal(cls) + 1.0*[0.5*BCE + 0.5*Dice](seg) + 0.3*Edge(seg) |

No architectural changes whatsoever. Same edge loss AMP fix as vK.11.4.

### Verdict

Architecture unchanged. See vK.11.4 audit for detailed analysis.

---

## 4. Training Pipeline Review

### Training Execution Summary

| Metric | vK.11.5 | vK.11.4 (reference) |
|--------|---------|---------------------|
| Total epochs | **13** | 25 |
| Best epoch | **3** | 15 |
| Best val Dice (tam) | **0.1364** | 0.1412 |
| Initial train loss | 4.6402 | 4.6517 |
| Final train loss | 1.6958 | 1.8795 |

### Epoch-by-Epoch Training Dynamics

| Epoch | Train Loss | Val Acc | Val AUC | Val Dice(tam) | Notes |
|-------|-----------|---------|---------|---------------|-------|
| 1 | 4.6402 | 0.4059 | 0.6368 | 0.1232 | Encoder frozen |
| 2 | 1.7408 | 0.5312 | 0.6906 | 0.1355 | Encoder frozen |
| **3** | **1.7327** | **0.4175** | **0.6331** | **0.1364** | **BEST -- first unfrozen epoch** |
| 4 | 1.7316 | 0.4926 | 0.7037 | 0.1352 | Declining from peak |
| 5 | 1.7280 | 0.4101 | 0.6541 | 0.1247 | Further decline |
| 8 | 1.7097 | 0.4116 | 0.6311 | 0.1113 | Significant degradation |
| 10 | 1.7037 | 0.4090 | 0.6303 | 0.1139 | |
| 13 | 1.6958 | 0.4212 | 0.6356 | 0.1178 | Early stop triggered |

### Critical Observations

1. **The model peaks at epoch 3 -- the very first epoch after encoder unfreeze.** Val Dice(tam) hits 0.1364 at epoch 3 and then DECLINES for 10 consecutive epochs, dropping to 0.1113 by epoch 8. This is the opposite of learning.

2. **Encoder unfreeze is destructive.** The pattern is clear: during the 2 frozen epochs, val Dice climbs from 0.1232 → 0.1355. The moment the encoder unfreezes at epoch 3, Dice peaks at 0.1364 (marginal gain) then degrades. The pretrained ResNet34 features are being corrupted, not fine-tuned.

3. **Train loss continues to decrease** (1.73 → 1.70) while validation metrics degrade. This is a textbook overfitting pattern, except the model is not even overfitting to anything useful -- it's memorizing training-set patterns that do not generalize.

4. **Compared to vK.11.4**: Training lasted only 13 epochs vs 25, with the best epoch at 3 vs 15. The non-determinism between runs (different GPU scheduling, data loading order) produced a worse trajectory. Both converged to the same constant-output behavior, but vK.11.5 got there faster.

### Encoder Unfreeze Analysis

| Epoch | Encoder Status | Val Dice(tam) | Trend |
|-------|---------------|---------------|-------|
| 1 | Frozen | 0.1232 | Improving |
| 2 | Frozen | 0.1355 | Improving |
| 3 | **Unfrozen** | 0.1364 | Peak (+0.0009) |
| 4 | Unfrozen | 0.1352 | **Declining** |
| 5+ | Unfrozen | ≤0.1247 | **Declining** |

**The encoder learning rate of 1e-4 is too high for fine-tuning.** Standard practice for fine-tuning pretrained encoders on small datasets is 1e-5 to 1e-6. At 1e-4, the encoder's ImageNet features are being overwritten rather than adapted. Combined with the 4th ELA channel (which requires the first conv layer to learn entirely new weights), the encoder is destabilized.

### Verdict

**Training confirms the same failure mode as vK.11.4, but faster.** The model peaks at the moment of encoder unfreeze and degrades from there. This is strong evidence that the encoder learning rate is destructive.

---

## 5. Evaluation Metrics Review

### Test Set Results

| Metric | vK.11.5 | vK.11.4 | Delta |
|--------|---------|---------|-------|
| **Accuracy** | **0.4194** | 0.4142 | +0.0052 |
| **AUC-ROC** | **0.6466** | 0.6434 | +0.0032 |
| Dice (all) | 0.0517 | 0.0537 | -0.0020 |
| IoU (all) | 0.0312 | 0.0335 | -0.0023 |
| **Dice (tampered)** | **0.1272** | **0.1321** | **-0.0049** |
| **IoU (tampered)** | **0.0768** | **0.0825** | **-0.0057** |
| **F1 (tampered)** | **0.1272** | **0.1321** | **-0.0049** |

**vK.11.5 is WORSE than vK.11.4** on all segmentation metrics. Despite slightly better accuracy and AUC (within noise), the tampered-only Dice/IoU/F1 are all lower.

### Threshold Sweep

| Parameter | vK.11.5 | vK.11.4 |
|-----------|---------|---------|
| Optimal threshold | **0.0500** | 0.4939 |
| F1 at optimal | 0.1321 | 0.1321 |

**The optimal threshold hit the floor at 0.05** (the lowest tested value). This means the model's probability outputs are clustered near zero -- the sigmoid output for most pixels is below 0.05. Despite this dramatically different threshold, the resulting F1 is **identical to vK.11.4's** -- 0.1321 in both cases. This confirms both models produce the same degenerate output pattern.

### Pixel-Level AUC-ROC

**0.5215** (vs vK.11.4's 0.4988). Both within noise of 0.50 (random chance). No meaningful discriminative power.

### The Identical-Output Evidence

The following metrics are **bitwise identical** between vK.11.4 and vK.11.5 despite different training durations and different best epochs:

| Metric | vK.11.4 | vK.11.5 | Match? |
|--------|---------|---------|--------|
| Per-forgery Dice (splicing) | 0.1016 | 0.1016 | **EXACT** |
| Per-forgery Dice (copy-move) | 0.1918 | 0.1918 | **EXACT** |
| Mask-size tiny Dice | 0.0190 | 0.0190 | **EXACT** |
| Mask-size small Dice | 0.0630 | 0.0630 | **EXACT** |
| Mask-size medium Dice | 0.1537 | 0.1537 | **EXACT** |
| Mask-size large Dice | 0.4860 | 0.4859 | Within rounding |
| Shortcut baseline F1 | 0.1321 | 0.1321 | **EXACT** |
| Shuffled-mask delta | 0.0000 | 0.0000 | **EXACT** |
| Robustness (all 8 conditions) | 0.1321 | 0.1321 | **EXACT** |

**Two different training runs producing bitwise identical per-type and per-size metrics is statistically near-impossible -- unless both models output the same constant prediction.** The models have converged to the same fixed prediction pattern: a near-zero-valued mask that produces ~0.13 Dice when thresholded against the test set's ground truth distribution.

### Shortcut Learning Detection

| Test | Baseline F1 | Modified F1 | Delta |
|------|------------|-------------|-------|
| Mask randomization | 0.1321 | 0.1321 | **0.0000** |
| Boundary erosion | 0.1321 | 0.1321 | **0.0000** |

Same result as vK.11.4. The model ignores image content.

### Robustness Testing

All 8 conditions produce F1=0.1321 with zero delta. Same Albumentations API deprecation warnings as vK.11.4.

### Verdict

**Confirms vK.11.4's findings with additional evidence.** The identical cross-run metrics are the strongest proof that both models converge to the same constant prediction. vK.11.5 is marginally worse on primary metrics but the difference is noise -- both models are non-functional.

---

## 6. Visualization Quality

### Results Dashboard (NEW in vK.11.5)

The Results Dashboard (cells 8-15) is the only structural addition over vK.11.4. It includes:

| Dashboard Panel | Cell | Output |
|----------------|------|--------|
| Metrics Summary Table | ec=2 | **"Final test metrics have not been computed yet"** |
| Training Curve Visualization | ec=3 | **"Training history not yet available"** |
| Example Localization | ec=4 | **"Example localization not available yet"** |
| Interpretation | ec=5 | **Markdown only (static text)** |

**ALL dashboard code cells show placeholder messages.** This is because they execute as cells ec=2-5, before the training loop (which starts around ec=29) and test evaluation (which defines `FINAL_TEST_METRICS` around ec=40+).

The dashboard is designed for a workflow where the user re-runs individual cells after training completes. This is reasonable in Jupyter/Colab interactive mode, but **Kaggle runs notebooks top-to-bottom once**. The dashboard is architecturally broken for the Kaggle execution environment.

### Other Visualizations

Same as vK.11.4:
- Executive Summary metrics: placeholder text
- W&B prediction panels: logged (offline)
- Training loss curve: shows flatline/decline
- Threshold sweep plot: flat F1 across thresholds
- Robustness bar chart: all bars identical
- Confusion matrix, sample predictions, Grad-CAM: cells exist but no image output

### Verdict

**The Results Dashboard adds visual polish but provides zero value in execution.** It is a presentation feature that cannot present anything because it runs before anything happens. The underlying issue (model producing constant output) means even a correctly-ordered dashboard would show embarrassing numbers.

---

## 7. Assignment Alignment Check

| Requirement | Status | Notes |
|-------------|--------|-------|
| **1. Dataset Selection** | PASS | CASIA v2.0 |
| **1. Data Pipeline** | PASS | Complete pipeline with ELA |
| **1. Augmentation** | PASS | 7 transforms |
| **2. Architecture** | PASS | Dual-head TamperDetector documented |
| **2. Resource Constraints** | PASS | Kaggle T4 compatible |
| **3. Performance Metrics** | **FAIL** | Tam-F1=0.1272, pixel-AUC=0.5215 -- near-random |
| **3. Visual Results** | **PARTIAL** | Dashboard shows placeholders, some visuals missing |
| **4. Single Notebook** | PASS | Everything in one .ipynb |
| **4. Model Weights** | **FAIL** | Weights saved but model is non-functional |
| **Bonus: Robustness** | **FAIL** | Present but API broken + constant output |
| **Bonus: Subtle Tampering** | **FAIL** | Per-type breakdown near-random |

### Verdict

Same as vK.11.4: structurally complete, functionally non-compliant. The Results Dashboard adds presentational polish but cannot compensate for a model that does not work.

---

## 8. Engineering Quality

### Strengths (carried from vK.11.4)

All vK.11.4 engineering strengths apply: CONFIG centralization, reproducibility infrastructure, checkpoint system, data leakage prevention, comprehensive evaluation suite, AMP implementation, W&B integration.

### Results Dashboard Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Concept | A | Excellent idea -- quick overview for reviewers |
| Implementation | D | Runs before training, always shows placeholders on Kaggle |
| Suggested fix | — | Move dashboard cells to after Section 11 (test evaluation), or use conditional rendering that detects whether `FINAL_TEST_METRICS` exists |

### Additional Issues

| Issue | Severity | Detail |
|-------|----------|--------|
| Dashboard execution order | **High** | All dashboard panels show placeholder text |
| Model Card says "vK.11.1" | Low | Not updated from base notebook |
| `train_dice` always 0.0 | Medium | Never computed, misleading in training history |
| Albumentations deprecation | Medium | Robustness transforms use deprecated API |

### Verdict

**Engineering quality remains high, but the Results Dashboard implementation is a significant miss.** The dashboard was the right idea (quick metrics overview for reviewers), but executing it before training exists defeats its purpose entirely. This is the kind of UI/UX bug that suggests the notebook was designed for interactive use but deployed in a batch execution environment.

---

## 9. Roast Section

**"The Second Run That Proved the First Wasn't a Fluke"**

vK.11.5 exists to answer one question: was vK.11.4's failure a random bad run? The answer is no. It's worse.

The model peaked at epoch 3 -- the very first epoch after the encoder was unfrozen -- and then spent 10 epochs getting progressively worse until early stopping finally put it out of its misery. That's not training. That's a pretrained ResNet34 slowly forgetting how to see, one gradient update at a time.

The epoch 3 peak is the most informative result in the entire vK.11.x series. During the frozen phase (epochs 1-2), the decoder slowly learns to produce segmentation output using fixed ImageNet features. The moment the encoder unfreezes, those features start getting corrupted by gradients from three competing objectives -- classification (1.5x weight!), segmentation, and edge detection -- flowing through a 4-channel input layer that was awkwardly patched onto a 3-channel pretrained network. The model's best moment is the instant before the encoder learns anything. That tells you everything about the optimization landscape.

The Results Dashboard is the crown jewel of vK.11.5. Four beautifully formatted cells designed to give reviewers a quick overview of the model's performance. Metrics Summary Table: "Final test metrics have not been computed yet." Training Curve Visualization: "Training history not yet available." Example Localization: "Example localization not available yet." The dashboard runs as cells 2-5 in a 135-cell notebook where training doesn't start until cell 29. It is a presentation layer with nothing to present.

But the most damning finding isn't the performance -- it's the forensics. vK.11.4 and vK.11.5 produce **bitwise identical** per-forgery-type Dice, mask-size stratification, shortcut test results, and robustness metrics. Two different training runs. Different epoch counts (25 vs 13). Different best epochs (15 vs 3). Different final losses (1.88 vs 1.70). And yet the evaluation outputs match to 4 decimal places across every stratified analysis.

This isn't "similar results from similar models." This is two models that independently converged to the exact same constant prediction. The training loop spent ~3 GPU-hours (across both runs) to discover that the optimal strategy for the vK.11 loss function is to output approximately zero for every pixel. The model didn't learn to detect tampering. It learned that outputting nothing minimizes the combined Focal + BCE + Dice + Edge loss when 59% of images are authentic (and thus have zero-mask ground truth).

The fix for this is not "more epochs" or "better augmentation" or "a fancier dashboard." The fix is to recognize that combining five architectural changes simultaneously without ablation is not engineering -- it's gambling. And the house won.

**Score: 3.5/10** (same engineering quality as 11.4, worse results, broken dashboard)

---

## Summary

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Dataset Pipeline | **A-** | Identical to vK.11.4, sound |
| Model Architecture | **B** | Unchanged, same concerns |
| Training Pipeline | **C+** | Model degrades after encoder unfreeze (epoch 3 peak) |
| Evaluation Metrics | **A** (code) / **F** (results) | Comprehensive suite proves model failure |
| Visualization | **D+** | Dashboard added but permanently shows placeholders |
| Assignment Alignment | **D** | Structurally complete, functionally non-compliant |
| Engineering Quality | **B+** | High baseline, dashboard implementation broken |
| **Overall** | **3.5/10** | Worse than vK.11.4 with a non-functional cosmetic addition |

### Key Metrics vs Project History

| Run | Tam-F1 | Tam-IoU | Img Acc | AUC | Pixel-AUC |
|-----|--------|---------|---------|-----|-----------|
| v6.5 (best) | **0.4101** | 0.3563 | 0.8246 | 0.8703 | — |
| vK.10.6 | 0.2213 | 0.1554 | 0.8357 | 0.9057 | 0.7083 |
| v8 (regressed) | 0.2949 | 0.2321 | 0.7190 | 0.8170 | — |
| vK.11.4 | 0.1321 | 0.0825 | 0.4142 | 0.6434 | 0.4988 |
| **vK.11.5** | **0.1272** | **0.0768** | **0.4194** | **0.6466** | **0.5215** |

**vK.11.5 is the worst pretrained-encoder run in project history, surpassing vK.11.4 for the distinction.**
