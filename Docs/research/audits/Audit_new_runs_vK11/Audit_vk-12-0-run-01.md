# Technical Audit: vK.12.0 (Run 01)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vk-12-0-tampered-image-detection-and-localization-run-01.ipynb` |
| **Platform** | Kaggle, 2x Tesla T4 GPUs (~30 GB VRAM) |
| **Cells** | 151 total (69 code, 82 markdown) |
| **Executed** | 53 of 69 code cells (76.8%) |
| **Crashed** | Yes -- `KeyError: 'true_mask'` at cell 53 (Section 12.3) |
| **Cells killed by crash** | 42 cells never ran (Sections 13-19) |
| **Training** | 16 epochs (early stopped), best at epoch 6 |
| **Runtime** | 2,152.7 seconds (~35.9 minutes) |
| **Status** | **CRASHED MID-EXECUTION -- MODEL FAILED TO LEARN (SAME AS vK.11.x)** |

---

## 1. Notebook Overview

vK.12.0 is the **latest version** of the project, built on top of vK.11.5. It adds 10 planned improvement areas focused on **evaluation and presentation** -- not on architecture or training. The functional ML code (model, CONFIG, loss, training loop) is 100% identical to vK.11.4/vK.11.5.

### What's New in vK.12.0

| # | Improvement | Executed? | Status |
|---|-------------|-----------|--------|
| 1 | Better Localization Metrics (Precision, Recall, Pixel Accuracy) | Yes | Working |
| 2 | Enhanced Robustness Testing (Dice+IoU+F1 per condition) | No | Blocked by crash |
| 3 | FP/FN Error Analysis with contour overlays | No | Blocked by crash |
| 4 | Improved Visualizations (difference maps, contour overlays) | **CRASHED** | `KeyError: 'true_mask'` |
| 5 | Dataset Exploration (mask coverage distribution) | Yes | Working |
| 6 | Architecture Diagram (programmatic matplotlib) | Yes | Working |
| 7 | Training Strategy Explanation | Yes | Working |
| 8 | Experiment Comparison Table | Yes | **Misleading** (hardcoded baselines are inflated) |
| 9 | Model Complexity Section (torchinfo) | Yes | Working |
| 10 | Inference Speed Test | No | Blocked by crash |

**Result: 6 of 10 improvements executed. 1 crashed. 3 blocked.**

### CONFIG

Identical to vK.11.4/vK.11.5:

```python
CONFIG = {
    'image_size': 256,
    'batch_size': 8,              # auto-scaled to 16 for 2xT4
    'max_epochs': 50,
    'patience': 10,
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'accumulation_steps': 4,      # effective batch = 64
    'encoder_freeze_epochs': 2,
    'alpha': 1.5,                 # classification loss weight
    'beta': 1.0,                  # segmentation loss weight
    'gamma': 0.3,                 # edge loss weight
    'seed': 42,
    # ... all other params unchanged
}
```

**Zero CONFIG changes from vK.11.5.** The same hyperparameters that produced Tam-F1=0.1272 in vK.11.5 are used again.

---

## 2. Dataset Pipeline Review

Identical to vK.11.5:

| Parameter | Value |
|-----------|-------|
| Dataset | CASIA v2.0 Upgraded (Kaggle) |
| Total Images | 12,614 (7,491 authentic + 5,123 tampered) |
| Split | 70/15/15 stratified (seed=42) |
| Train / Val / Test | 8,829 / 1,892 / 1,893 |
| Image Size | 256x256 |
| Input Channels | 4 (RGB + ELA grayscale, quality=90) |

Data leakage check PASSED.

### New: Dataset Exploration (Section 6.4)

vK.12.0 adds a mask coverage analysis for tampered training images:

| Statistic | Value |
|-----------|-------|
| Images analyzed | 3,586 tampered (training set) |
| Coverage range | 0.0% -- 96.3% |
| Mean coverage | 9.1% |
| Median coverage | 3.1% |
| Images with <2% coverage | 1,348 (37.6%) |
| Images with >15% coverage | 626 (17.5%) |

**Insight**: 37.6% of tampered images have mask coverage below 2%. This means over a third of the training signal consists of tiny tampered regions that are extremely difficult to localize. The median of 3.1% explains why the model struggles -- the typical forgery covers less than 1/30th of the image.

This is a genuinely useful addition. It provides context for why tiny-mask Dice is 0.019 and why the model may be biased toward predicting zero (the majority of pixels in most tampered images ARE authentic).

### Verdict

Pipeline unchanged. Dataset exploration is a valuable new analysis that contextualizes model performance.

---

## 3. Model Architecture Review

**Identical to vK.11.5** -- no changes:

| Component | Detail |
|-----------|--------|
| Model | TamperDetector (SMP UNet + ResNet34 + FC classifier) |
| Input | 4 channels (RGB + ELA) |
| Total Parameters | 24,571,347 (all trainable) |
| Model Size | 93.7 MB (FP32) |
| Forward/Backward | 143.66 MB |
| Total mult-adds | 7.87 GFLOPS |
| Loss | 1.5*Focal(cls) + 1.0*[0.5*BCE + 0.5*Dice](seg) + 0.3*Edge(seg) |

### New: Model Complexity Section (Section 7.2)

vK.12.0 adds a `torchinfo` summary with full layer-by-layer parameter breakdown. Key layers:

| Layer | Output Shape | Params |
|-------|-------------|--------|
| Stem Conv2d (4→64) | [1, 64, 128, 128] | 12,544 |
| ResNet layer1 | [1, 64, 64, 64] | 221,952 |
| ResNet layer2 | [1, 128, 32, 32] | 1,116,416 |
| ResNet layer3 | [1, 256, 16, 16] | 6,822,400 |
| ResNet layer4 | [1, 512, 8, 8] | 13,114,368 |
| UNet Decoder | [1, 16, 256, 256] | 3,151,552 |
| Segmentation Head | [1, 1, 256, 256] | 145 |
| Classifier FC | [1, 2] | 131,842 |

The decoder outputs 16 channels before the segmentation head reduces to 1. This is a relatively narrow decoder bottleneck, though standard for SMP's default UNet.

### New: Architecture Diagram (Section 7.1)

A programmatic matplotlib diagram was generated and rendered. This adds visual documentation.

### Verdict

Architecture unchanged. The model complexity section and architecture diagram are useful documentation additions but do not affect model performance.

---

## 4. Training Pipeline Review

### Training Execution Summary

| Metric | vK.12.0 | vK.11.5 (reference) | vK.11.4 (reference) |
|--------|---------|---------------------|---------------------|
| Epochs | 16 | 13 | 25 |
| Best epoch | **6** | 3 | 15 |
| Best val Dice(tam) | **0.1412** | 0.1364 | 0.1412 |
| Initial train loss | 4.5975 | 4.6402 | 4.6517 |
| Final train loss | 1.7305 | 1.6958 | 1.8795 |
| Early stopped | Yes (patience=10) | Yes (patience=10) | Yes (patience=10) |

### Epoch-by-Epoch Training Dynamics

| Epoch | Train Loss | Val Acc | Val AUC | Val Dice(tam) | Notes |
|-------|-----------|---------|---------|---------------|-------|
| 1 | 4.5975 | 0.4059 | 0.6557 | 0.1263 | Encoder frozen |
| 2 | 1.7269 | 0.5687 | 0.6978 | 0.1269 | Encoder frozen |
| 3 | 1.7213 | 0.5248 | 0.7244 | 0.1252 | Encoder unfrozen; AUC peaks |
| 4 | 1.7246 | 0.4191 | 0.6710 | 0.1222 | AUC drops after unfreeze |
| 5 | 1.7203 | 0.5624 | 0.6981 | 0.1217 | |
| **6** | **1.6906** | **0.4059** | **0.5882** | **0.1412** | **BEST Dice -- but AUC collapsed** |
| 7-16 | ~1.72-1.73 | ~0.41-0.46 | ~0.55-0.63 | 0.1412 (flat) | 10 epochs, zero improvement |

### Critical Observations

1. **Val Dice(tam) flatlines at EXACTLY 0.1412 from epoch 6 through 16.** Ten consecutive epochs with identical Dice to 4 decimal places. This is the same constant-output collapse seen in vK.11.4 and vK.11.5.

2. **Interesting timing**: Best Dice (epoch 6) coincides with worst AUC (0.5882). The model's best segmentation moment comes precisely when its classification ability collapses. This supports the hypothesis that the classification and segmentation objectives compete -- when classification degrades, segmentation briefly benefits.

3. **Val AUC peak at epoch 3** (0.7244) then degrades after encoder unfreeze -- same pattern as vK.11.5. The pretrained encoder features are being corrupted, not fine-tuned.

4. **Training accuracy oscillates 41-52%** -- near random for binary classification. The model never learns a useful decision boundary.

### Verdict

Same failure mode as vK.11.4/vK.11.5. The model converges to a constant prediction within 6 epochs. The only new insight is the inverse correlation between AUC and Dice at epoch 6, which supports the loss-conflict hypothesis.

---

## 5. Evaluation Metrics Review

### Test Results

| Metric | vK.12.0 | vK.11.4 | vK.11.5 | Assessment |
|--------|---------|---------|---------|------------|
| **Accuracy** | **0.4062** | 0.4142 | 0.4194 | Worst of all three -- below random (59.4%) |
| **AUC-ROC** | **0.5637** | 0.6434 | 0.6466 | **Significant degradation** (-0.08) |
| Dice (all) | 0.0537 | 0.0537 | 0.0517 | Near-identical |
| IoU (all) | 0.0335 | 0.0335 | 0.0312 | Near-identical |
| **Dice (tampered)** | **0.1321** | **0.1321** | 0.1272 | **EXACT match** with vK.11.4 |
| **IoU (tampered)** | **0.0825** | **0.0825** | 0.0768 | **EXACT match** with vK.11.4 |
| **F1 (tampered)** | **0.1321** | **0.1321** | 0.1272 | **EXACT match** with vK.11.4 |

### New Metrics Added in vK.12.0

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision (tampered)** | **0.0825** | Only 8.25% of predicted tampered pixels are actually tampered |
| **Recall (tampered)** | **1.0000** | Model predicts ALL pixels as tampered |
| **Pixel Accuracy (tampered)** | **0.0825** | Same as precision -- only 8.25% correct |
| **Pixel-Level AUC-ROC** | **0.4952** | **Below random (0.50)** -- anti-correlated with truth |

**Recall = 1.0000 is the definitive proof of the failure mode.** The model predicts every pixel as tampered. This produces:
- Perfect recall (all tampered pixels are "found" because ALL pixels are marked)
- Terrible precision (8.25% -- only the ground truth fraction is correct)
- Pixel-AUC below 0.50 -- the probability outputs are actually ANTI-correlated with ground truth

The new precision/recall metrics in vK.12.0 provide the clearest diagnostic yet of the model's behavior. They are the most informative addition in the vK.12.0 notebook.

### Threshold Sweep

| Parameter | Value |
|-----------|-------|
| Optimal threshold | 0.5092 |
| F1 at optimal | 0.1321 |

### Per-Forgery-Type Breakdown

| Forgery Type | Dice | F1 | Match vK.11.4? |
|-------------|------|-----|----------------|
| Splicing | 0.1016 | 0.1016 | **EXACT** |
| Copy-move | 0.1918 | 0.1918 | **EXACT** |

### Mask-Size Stratification

| Size | Dice | F1 | Match vK.11.4? |
|------|------|-----|----------------|
| Tiny (<2%) | 0.0190 | 0.0190 | **EXACT** |
| Small (2-5%) | 0.0630 | 0.0630 | **EXACT** |
| Medium (5-15%) | 0.1537 | 0.1537 | **EXACT** |
| Large (>15%) | 0.4859 | 0.4859 | **EXACT** |

### Shortcut Learning Detection

| Test | Baseline F1 | Modified F1 | Delta |
|------|------------|-------------|-------|
| Mask randomization | 0.1321 | 0.1321 | **0.0000** |
| Boundary erosion | 0.1321 | 0.1321 | **0.0000** |

**Same result as vK.11.4 and vK.11.5.** Three separate training runs, three different epoch counts, three different best epochs -- identical evaluation metrics across every stratified analysis. The model converges to the same constant prediction.

### Robustness Testing

**NOT EXECUTED** -- blocked by the Section 12.3 crash. The enhanced robustness suite (8 conditions with Dice+IoU+F1) is defined but never ran.

### Verdict

The new precision/recall/pixel-accuracy metrics are the most valuable addition in vK.12.0. They unambiguously prove the model predicts ALL pixels as tampered (Recall=1.0, Precision=0.0825). The pixel-AUC of 0.4952 (below random) shows the continuous probability outputs are anti-correlated with ground truth. All stratified metrics are bitwise identical to vK.11.4.

---

## 6. The Crash: `KeyError: 'true_mask'`

### What Happened

The notebook crashed at **cell 53** (Section 12.3, "Difference Map and Contour Overlay") with:

```
KeyError: 'true_mask'
```

The code attempts `s['true_mask'].squeeze().numpy()`, but the sample dictionary uses a different key for ground truth masks (likely `'mask'` or `'gt_mask'`).

### Impact Assessment

This single bug killed **42 of 151 cells** (28% of the notebook), preventing execution of:

| Blocked Section | Content | Severity |
|----------------|---------|----------|
| 13.1 | Failure case analysis (top-10 worst predictions) | Medium |
| **13.2** | **FP/FN error analysis with contour overlays** | **High -- new vK.12.0 feature** |
| 14 | Grad-CAM heatmaps | Medium |
| **15** | **Enhanced robustness testing (8 conditions, Dice+IoU+F1)** | **High -- new vK.12.0 feature** |
| **16.1** | **Inference speed benchmark** | **High -- new vK.12.0 feature** |
| 17 | Model card | Low |
| 18 | Reproducibility verification | Medium |
| 19 | Quick inference demo | Low |

**Three of the ten vK.12.0 improvements were never tested** because of a dictionary key typo.

### Fix

One line: change `s['true_mask']` to match the actual key used when populating the sample dictionary. This is a trivial bug with catastrophic impact on the notebook's execution.

---

## 7. Visualization Quality

### Executed Visualizations (12 of 20)

| Visualization | Section | Status | Notes |
|--------------|---------|--------|-------|
| Mask coverage histogram + CDF | 6.4 | Rendered | **NEW** -- valuable dataset insight |
| Augmentation preview grid | 6.x | Rendered | 5-panel (original + 4 augmented) |
| Architecture diagram | 7.1 | Rendered | **NEW** -- programmatic matplotlib |
| torchinfo summary | 7.2 | Rendered (text) | **NEW** -- full layer breakdown |
| Training curves (4 subplots) | 11.2 | Rendered | Loss, accuracy, Dice, LR schedule |
| Threshold sweep plot | 11.3 | Rendered | Flat F1 across thresholds |
| Confusion matrix | 11.5 | Rendered | |
| ROC curve | 11.5 | Rendered | |
| PR curve | 11.5 | Rendered | |
| Mask-size F1 bar chart | 11.7 | Rendered | F1 by size bucket |
| Experiment comparison table | 11.9 | Rendered (text) | **MISLEADING** -- see Section 8 |
| 4-panel prediction grid | 12 | Rendered | Original / GT / Predicted / Overlay |
| ELA visualization | 12.2 | Rendered | RGB vs ELA side-by-side |

### Crashed / Never Rendered (8 of 20)

| Visualization | Section | Status |
|--------------|---------|--------|
| 6-panel enhanced viz (diff maps, contours) | 12.3 | **CRASHED** |
| FP/FN error analysis | 13.2 | Blocked |
| Failure case gallery | 13.1 | Blocked |
| Grad-CAM heatmaps | 14 | Blocked |
| Robustness bar charts | 15 | Blocked |
| Degradation condition gallery | 15 | Blocked |
| Inference benchmark results | 16.1 | Blocked |
| Single-image demo | 19 | Blocked |

### Executive Summary

Still shows "Final test metrics have not been computed yet" -- same broken execution-order issue as vK.11.5's Results Dashboard.

### Verdict

The executed visualizations are the most comprehensive in the project. The mask coverage histogram, architecture diagram, and model complexity summary add genuine value. However, the crash blocked the most visually impactful new additions (difference maps, contour overlays, FP/FN analysis, Grad-CAM) -- which was the primary visual improvement promised in vK.12.0.

---

## 8. Assignment Alignment Check

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **1. Dataset Selection** | PASS | CASIA v2.0, with mask coverage analysis |
| **1. Data Pipeline** | PASS | Complete pipeline with ELA |
| **1. Augmentation** | PASS | 7 transforms, augmentation preview rendered |
| **2. Architecture** | PASS | Documented + architecture diagram + torchinfo |
| **2. Resource Constraints** | PASS | Runs on Kaggle T4 GPUs in 36 minutes |
| **3. Performance Metrics** | **FAIL** | Tam-F1=0.1321, Recall=1.0 (predicts everything), pixel-AUC=0.4952 |
| **3. Visual Results** | **PARTIAL** | 4-panel predictions rendered; enhanced viz crashed |
| **4. Single Notebook** | PASS | Everything in one .ipynb |
| **4. Model Weights** | **FAIL** | Weights saved but model is non-functional |
| **Bonus: Robustness** | **FAIL** | Code present but never executed (crash blocked) |
| **Bonus: Subtle Tampering** | **FAIL** | Per-type breakdown shows near-random F1 |

### Verdict

Same structural compliance as vK.11.x. The additional documentation (architecture diagram, model complexity, dataset exploration) strengthens the presentation, but the model remains non-functional. The crash blocking robustness testing is particularly harmful to the bonus requirement.

---

## 9. Engineering Quality

### Strengths

| Aspect | Rating | Notes |
|--------|--------|-------|
| Dataset exploration | **A** | Mask coverage analysis provides genuine insight |
| Model documentation | **A** | torchinfo + architecture diagram = best in project |
| Evaluation breadth | **A** | Precision/Recall/PixelAcc are correctly computed and revealing |
| Experiment comparison | **B-** | Table structure is good but historical baselines are fabricated |
| CONFIG system | **A** | Carried from vK.11.x, comprehensive |
| Checkpoint system | **A** | 3-file strategy |
| Reproducibility infrastructure | **B+** | Seeds set, but verification section didn't execute |

### Weaknesses

| Issue | Severity | Detail |
|-------|----------|--------|
| **`KeyError: 'true_mask'` crash** | **CRITICAL** | One typo killed 42 cells and 3 of 10 new features |
| **Hardcoded comparison baselines** | **HIGH** | Section 11.9 claims "v11.x ~0.350 Tam Dice, ~0.380 Tam F1, ~0.850 Acc, ~0.900 AUC" but vK.11.5's actual metrics are 0.127, 0.127, 0.419, 0.647 -- inflated by 3-4x |
| Executive Summary placeholder | Medium | Same broken execution-order as vK.11.5 |
| Model Card says "vK.11.1" | Low | Likely still not updated |
| No CONFIG changes attempted | Medium | Same hyperparameters that failed in vK.11.4/11.5 |
| No architecture changes attempted | Medium | Same model that failed in vK.11.4/11.5 |
| `train_dice` always 0.0 | Medium | Never computed |

### The Hardcoded Baselines Problem

The Experiment Comparison Table (Section 11.9) includes hardcoded approximate historical baselines alongside live vK.12.0 metrics. The hardcoded values are:

| Version | Hardcoded Tam Dice | Hardcoded Tam F1 | Actual Tam F1 | Inflation |
|---------|-------------------|------------------|---------------|-----------|
| v7 | ~0.200 | ~0.200 | ~0.20 (est) | ~OK |
| v10.x | ~0.220 | ~0.230 | 0.2213 | ~OK |
| **v11.x** | **~0.350** | **~0.380** | **0.1272** | **3.0x inflated** |

The v11.x hardcoded values appear to be aspirational targets, not actual results. This creates a misleading comparison that makes vK.12.0 look worse than it is relative to a false baseline, while simultaneously making the project history look better than it actually is.

### Verdict

**Engineering is strong in evaluation and documentation, weak in testing and validation.** The `KeyError` crash is the kind of bug that a single dry-run would have caught. The hardcoded comparison baselines are intellectually dishonest (even if unintentional). The decision to re-run identical CONFIG and architecture without any changes, knowing they failed in vK.11.4/11.5, raises questions about experimentation discipline.

---

## 10. Roast Section

**"The Model Still Predicts Everything as Tampered, but Now We Have a Pretty Architecture Diagram"**

Let's get the good news out of the way first: vK.12.0 nailed the presentation. The mask coverage histogram reveals that 37.6% of tampered images have less than 2% coverage -- that's genuine insight, the first time anyone in this project has asked "what does the training signal actually look like?" The `torchinfo` layer breakdown is useful. The architecture diagram is clean. The precision/recall metrics are the most diagnostic numbers this project has ever produced.

Recall = 1.0000. Let that sink in. The model predicts **every single pixel** as tampered. 100% recall. Zero selectivity. The nuclear option of segmentation. It's the machine learning equivalent of a fire alarm that never stops ringing -- technically it will catch every fire, but you couldn't call it useful.

Precision = 0.0825. That means for every pixel the model says is tampered, there's an 8.25% chance it actually is. You'd get better odds throwing darts at the image. Actually, you'd need a pixel-level CASIA expertise to even get 8.25% by random chance -- that's approximately the mean mask coverage of the dataset. The model learned exactly one thing: the prior probability that any pixel is tampered. Then it applied that to everything.

Pixel-AUC = 0.4952. **Below** random. The continuous probability outputs are anti-correlated with ground truth. The model is not just failing to localize tampering -- it is slightly better at predicting where tampering ISN'T than where it IS. If you negated the output probabilities, you'd get pixel-AUC = 0.5048, which would technically be the project's best pixel-level discriminator. That's not a recommendation.

But the real star of this notebook is the `KeyError: 'true_mask'` crash. One dictionary key typo -- `'true_mask'` instead of whatever key the sample dictionary actually uses -- killed 42 of 151 cells. That's 28% of the notebook. Three of the ten planned vK.12.0 improvements (enhanced robustness, FP/FN analysis, inference speed) NEVER EXECUTED because nobody test-ran the visualization code before submission. The enhanced robustness suite -- the single feature that would have provided the most useful new data -- was defined but never produced a number. The inference speed benchmark that would have answered "how fast is this (non-functional) model?" sits there in elegant Python, gathering digital dust.

The Experiment Comparison Table is... creative. It includes hardcoded baselines claiming "v11.x ~0.350 Tam Dice, ~0.380 Tam F1." The actual vK.11.5 Tam-F1 is 0.1272. That's not a rounding difference -- that's a 3x inflation. If this were a financial audit, someone would be getting a phone call. Whether these are aspirational targets, planned results, or numbers from a parallel universe where the synthesis architecture works, they don't belong in a comparison table presented alongside real metrics.

And then there's the strategic question: **why run vK.12.0 with the exact same CONFIG?** The same `max_epochs=50`, `patience=10`, `encoder_lr=1e-4`, `alpha=1.5`, `beta=1.0`, `gamma=0.3` that failed in vK.11.4 and vK.11.5. Not a single hyperparameter changed. The definition of insanity is doing the same thing and expecting different results -- but to be fair, the results ARE almost the same: Tam-F1=0.1321 again, pixel-AUC=0.4952 again, shortcut test delta=0.0000 again. The model is remarkably consistent in its failure.

The 10 improvement areas were all about making the notebook look better -- better charts, better metrics, better documentation. Zero were about making the MODEL better. No architecture changes. No loss function modifications. No hyperparameter search. No ELA validation. No encoder LR reduction. No classification weight ablation. The notebook evolved; the model didn't.

vK.12.0 is the most professionally presented non-functional model in this project's history. It's a masterclass in evaluation infrastructure wrapped around a model that has learned exactly one thing: predict 1 for every pixel and let God sort out the Dice score.

**Score: 4/10** (Best documentation and evaluation infrastructure in the project, but same failed model, plus a crash that blocked 28% of the notebook)

---

## Summary

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Dataset Pipeline | **A** | Sound pipeline + new mask coverage analysis |
| Model Architecture | **B** | Unchanged, well-documented (torchinfo + diagram) |
| Training Pipeline | **C** | Same failure: constant output, epoch 6 flatline |
| Evaluation Metrics | **A** (code) / **F** (results) | Precision/Recall are the most diagnostic metrics yet |
| Visualization | **B-** | 12 of 20 rendered, but crash blocked the best new ones |
| Assignment Alignment | **D+** | Best documentation, but model still non-functional |
| Engineering Quality | **B** | Strong eval, but `KeyError` crash is inexcusable |
| **Overall** | **4/10** | Best presentation of the worst-performing model family |

### Key Metrics vs Project History

| Run | Tam-F1 | Tam-IoU | Img Acc | AUC | Pixel-AUC | Recall |
|-----|--------|---------|---------|-----|-----------|--------|
| v6.5 (best) | **0.4101** | 0.3563 | 0.8246 | 0.8703 | — | — |
| vK.10.6 | 0.2213 | 0.1554 | 0.8357 | 0.9057 | 0.7083 | — |
| v8 | 0.2949 | 0.2321 | 0.7190 | 0.8170 | — | — |
| vK.11.4 | 0.1321 | 0.0825 | 0.4142 | 0.6434 | 0.4988 | — |
| vK.11.5 | 0.1272 | 0.0768 | 0.4194 | 0.6466 | 0.5215 | — |
| **vK.12.0** | **0.1321** | **0.0825** | **0.4062** | **0.5637** | **0.4952** | **1.0000** |

### What vK.12.0 Proves

1. **The synthesis architecture consistently fails.** Three training runs (vK.11.4, vK.11.5, vK.12.0) with identical CONFIG produce Tam-F1 of 0.1321, 0.1272, and 0.1321 respectively.
2. **The model predicts all pixels as tampered.** Recall=1.0, Precision=0.0825. This is the clearest description of the failure mode.
3. **Presentation improvements cannot compensate for model failure.** vK.12.0 has the best documentation, visualization, and evaluation infrastructure in the project -- all proving the model does nothing.
4. **The same experiment should not be re-run without changes.** Using identical CONFIG after two prior failures was not experimentation -- it was repetition.
