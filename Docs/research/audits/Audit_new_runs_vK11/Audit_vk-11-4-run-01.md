# Technical Audit: vK.11.4 (Run 01)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vk-11-4-tampered-image-detection-and-localization-run-01.ipynb` |
| **Platform** | Kaggle, 2x Tesla T4 GPUs (31.3 GB VRAM) |
| **Cells** | 127 total (58 code, 69 markdown) |
| **Executed** | 54 of 58 code cells |
| **Training** | 25 epochs (early stopped), best at epoch 15 |
| **Status** | **FULLY EXECUTED -- MODEL FAILED TO LEARN** |

---

## 1. Notebook Overview

vK.11.4 is the **first fully executed run** of the synthesis architecture. It inherits vK.11.1's TamperDetector design (SMP UNet + ResNet34 + ELA + EdgeLoss + FC classifier) with three fixes and three structural additions:

**Code changes from vK.11.1:**
1. `edge_loss` AMP bug fixed (added `.float()` cast and `autocast(enabled=False)` wrapper)
2. `max_epochs` reduced 100 → 50, `patience` reduced 20 → 10
3. `ReduceLROnPlateau verbose=True` removed (deprecated parameter)
4. W&B sample prediction logging added (every 5 epochs, 2 tampered validation images)

**Structural additions:**
- Executive Summary section (cells 0-7) with problem statement, dataset overview, architecture diagram, training strategy, final metrics placeholder
- Reproducibility Verification section (section 18, cells 107-117)
- Quick Inference Demo section (section 19, cells 118-124, unexecuted)

**Unexecuted cells** (4 of 58): Quick Inference Demo (3 cells) and W&B finish.

### CONFIG

```python
CONFIG = {
    'img_size': 256,
    'batch_size': 8,         # auto-scaled to 32 for 2xT4
    'max_epochs': 50,        # DOWNGRADED from 11.1's 100
    'patience': 10,          # DOWNGRADED from 11.1's 20
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

### Dataset Configuration

Identical to vK.11.1:

| Parameter | Value |
|-----------|-------|
| Dataset | CASIA v2.0 Upgraded (Kaggle) |
| Total Images | 12,614 (7,491 authentic + 5,123 tampered) |
| Split | 70/15/15 stratified (seed=42) |
| Train / Val / Test | 8,829 / 1,892 / 1,893 |
| Image Size | 256x256 |
| Input Channels | 4 (RGB + ELA grayscale) |

### Data Leakage Check

**PASSED** -- zero overlap between splits (cell 30, ec=11). The single-block data pipeline eliminates the Block 1 leakage that plagued vK.1-vK.7.1.

### ELA Processing

ELA computed at JPEG quality=90 on BGR images before RGB conversion. The 4-channel stacking (RGB + ELA gray) was executed and verified through the model's shape check output.

### Augmentations

Same 7 transforms as vK.11.1. Applied with `additional_targets={'ela': 'image'}` for synchronized augmentation of ELA and RGB channels. Mask augmented in sync via Albumentations' built-in mask handling.

### DataLoader Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 32 (auto-scaled from 8 for 2xT4) |
| Workers | 4 per loader |
| Persistent workers | True |
| Drop last (train) | True |
| Worker init fn | Seeded (base_seed + worker_id) |

### Verdict

**Pipeline is sound and executed correctly.** Data leakage verified, ELA processed, augmentations applied. No issues identified in the data pipeline itself.

---

## 3. Model Architecture Review

Identical to vK.11.1:

| Component | Detail |
|-----------|--------|
| Encoder | ResNet34, ImageNet pretrained (SMP) |
| Decoder | UNet (SMP default) |
| Input | 4 channels (RGB + ELA) |
| Seg output | (B, 1, 256, 256) logits |
| Cls output | (B, 2) logits via GAP → FC(512→256→2) |
| Parameters | 24,571,347 (all trainable) |
| Attention | None |
| Runtime | `nn.DataParallel` on 2x T4 |

The model was instantiated (cell 45, ec=19) and shape-verified with a dummy 4-channel input. Output shapes confirmed: segmentation `(1, 1, 256, 256)`, classification `(1, 2)`.

### Loss Function

```
Total = 1.5 * FocalLoss(cls) + 1.0 * [0.5*BCE + 0.5*Dice](seg) + 0.3 * EdgeLoss(seg)
```

**Edge loss AMP fix** (the only code change affecting training):
```python
# vK.11.1 (buggy):
return F.binary_cross_entropy(pred_edge, gt_edge)

# vK.11.4 (fixed):
with torch.amp.autocast('cuda', enabled=False):
    return F.binary_cross_entropy(pred_edge.float(), gt_edge.float())
```

### Architecture Verdict

The architecture is **unchanged from vK.11.1** and is theoretically sound. The fix to the edge loss AMP casting was necessary for training to proceed. However, the architecture's failure to learn (see Section 4-5) suggests deeper issues with the multi-head loss weighting or 4-channel input adaptation.

---

## 4. Training Pipeline Review

### Training Execution Summary

| Metric | Value |
|--------|-------|
| Total epochs | 25 (early stopped at patience=10) |
| Best epoch | 15 |
| Best val Dice (tampered) | 0.1412 |
| Initial train loss | 4.6517 |
| Final train loss | 1.8795 |
| Training time | Full Kaggle session |

### Epoch-by-Epoch Training Dynamics

| Epoch | Train Loss | Val Acc | Val AUC | Val Dice(tam) | LR (enc) | Notes |
|-------|-----------|---------|---------|---------------|----------|-------|
| 1 | 4.6517 | 0.4064 | 0.6684 | 0.1180 | 1e-4 | Encoder frozen |
| 2 | 1.7397 | 0.5264 | 0.7038 | 0.1239 | 1e-4 | Encoder frozen |
| 3 | 1.7529 | 0.4799 | 0.7258 | 0.1227 | 1e-4 | Encoder unfrozen |
| 5 | 1.7740 | 0.4587 | 0.6993 | 0.1199 | 1e-4 | |
| 10 | 1.7774 | 0.4164 | 0.6613 | 0.1325 | 1e-4 | |
| 15 | 1.7687 | 0.4149 | 0.6437 | **0.1412** | 1e-4 | **BEST** |
| 20 | 1.8310 | 0.4149 | 0.5978 | 0.1412 | 5e-5 | LR reduced |
| 25 | 1.8795 | 0.4101 | 0.5645 | 0.1412 | 2.5e-5 | Early stop |

### Critical Observations

1. **Train loss NEVER meaningfully decreases after epoch 2.** The initial drop from 4.65 to 1.74 is just the model learning mean predictions. After that, loss oscillates in the 1.73-1.88 range -- the model is not learning.

2. **Val Dice (tampered) flatlines at 0.1412 from epoch 15-25.** Ten consecutive epochs with identical Dice to 4 decimal places means the model has converged to a fixed prediction pattern.

3. **Train loss INCREASES from epoch 15 onward** (1.77 → 1.88) while val metrics are frozen. This is a hallmark of training collapse -- the optimizer is moving weights around but not improving any objective.

4. **Val accuracy oscillates around 0.41-0.53** -- worse than the 59.4% majority-class baseline (always predict "authentic"). The classification head is not learning.

5. **Val AUC DECREASES from 0.73 (epoch 3) to 0.56 (epoch 25).** The model's discriminative ability is getting WORSE over training.

### Training Pipeline Assessment

| Feature | Implementation | Issue? |
|---------|---------------|--------|
| Gradient accumulation | 4-step with correct partial-window flush | OK |
| AMP | autocast + GradScaler + edge loss fix | OK |
| Encoder freeze | First 2 epochs | OK mechanically, but unfreezing may cause issues (see vK.11.5) |
| ReduceLROnPlateau | patience=3, factor=0.5 | Fires twice (epoch ~18 and ~21), no improvement |
| Early stopping | patience=10 on val Dice(tam) | Functions correctly, triggers at epoch 25 |
| W&B logging | Offline mode (no API key on Kaggle) | No cloud sync, local only |

### Verdict

**The training pipeline executes correctly but the model does not learn.** All components (AMP, accumulation, freezing, scheduling, checkpointing) work as designed. The failure is not in the training infrastructure -- it is in the model's ability to optimize the combined loss toward useful segmentation.

---

## 5. Evaluation Metrics Review

### Test Set Results

| Metric | Value | Assessment |
|--------|-------|------------|
| **Accuracy** | **0.4142** | **WORSE than random** (majority baseline = 59.4%) |
| **AUC-ROC** | **0.6434** | Poor |
| Dice (all samples) | 0.0537 | Extremely poor |
| IoU (all samples) | 0.0335 | Extremely poor |
| F1 (all samples) | 0.0537 | Extremely poor |
| **Dice (tampered only)** | **0.1321** | **Near-random** |
| **IoU (tampered only)** | **0.0825** | **Near-random** |
| **F1 (tampered only)** | **0.1321** | **Near-random** |

### Threshold Sweep

| Parameter | Value |
|-----------|-------|
| Range | 0.05 to 0.80 (50 points) |
| Optimal threshold | 0.4939 |
| F1 at optimal | 0.1321 |
| F1 at default (0.5) | 0.1321 |

**The threshold sweep produces the same F1 regardless of threshold value.** This confirms the model's predictions are clustered in a narrow range -- changing the threshold does not change which pixels are classified as tampered.

### Pixel-Level AUC-ROC

**0.4988** -- This is **statistically indistinguishable from 0.50 (random chance).** The model's pixel-level probability outputs carry zero discriminative information about tampering. This is the single most definitive metric proving the model has not learned.

### Per-Forgery-Type Breakdown

| Forgery Type | Dice | F1 | Images |
|-------------|------|-----|--------|
| Splicing | 0.1016 | 0.1016 | ~500 |
| Copy-move | 0.1918 | 0.1918 | ~269 |

Both types near-random. Copy-move slightly higher (0.19 vs 0.10), but still far below usefulness.

### Mask-Size Stratified Evaluation

| Size Category | Mask Coverage | Dice | Count |
|--------------|---------------|------|-------|
| Tiny | <2% of pixels | 0.0190 | — |
| Small | 2-5% | 0.0630 | — |
| Medium | 5-15% | 0.1537 | — |
| Large | >15% | 0.4860 | — |

Only large forgeries (>15% of image area) achieve a remotely non-trivial Dice (0.49), but this is likely an artifact of the model predicting a near-constant mask that overlaps with large tampered regions by chance.

### Shortcut Learning Detection

| Test | Baseline F1 | Modified F1 | Delta | Interpretation |
|------|------------|-------------|-------|----------------|
| Mask randomization | 0.1321 | 0.1321 | **0.0000** | **MODEL IGNORES IMAGE CONTENT** |
| Boundary erosion | 0.1321 | 0.1321 | **0.0000** | Model not boundary-sensitive |

**This is the smoking gun.** When ground truth masks are randomly shuffled (assigning the wrong mask to each image), the F1 score does not change AT ALL. This means the model's predictions are not correlated with image content. It is producing the same output regardless of which image it sees.

The notebook itself flags this: `[WARN] Mask randomization: F1 stable -> possible shortcut learning`

### Robustness Testing

| Condition | F1 | Delta from Baseline |
|-----------|-----|-------------------|
| Baseline (clean) | 0.1321 | — |
| JPEG QF=70 | 0.1321 | 0.0000 |
| JPEG QF=50 | 0.1321 | 0.0000 |
| Gaussian noise | 0.1321 | 0.0000 |
| Gaussian blur | 0.1321 | 0.0000 |
| Resize 50% | 0.1321 | 0.0000 |

**ALL conditions produce identical F1.** There are two compounding issues:
1. **Albumentations API deprecation**: Output warnings show `quality_lower`, `quality_upper` (for ImageCompression) and `var_limit` (for GaussNoise) are no longer valid parameters. The transforms may not be applying the intended degradations.
2. **Even if transforms applied correctly**, a model that produces constant output would show identical F1 for all conditions. The robustness test is measuring nothing because the model does nothing.

### Verdict

**Every evaluation metric confirms the model has not learned.** Pixel-AUC of 0.4988 is definitive: zero discriminative power at the pixel level. The shortcut test proves the model ignores image content. The robustness test confirms constant output. This is not a poorly-performing model -- it is a non-functional one.

---

## 6. Visualization Quality

### Executed Visualizations

| Visualization | Cell | Has Image Output | Notes |
|--------------|------|-----------------|-------|
| Executive Summary metrics | ec=1 | Text only | Shows placeholders (runs before training) |
| Model shape check | ec=19 | Yes (text) | Confirms (1,1,256,256) + (1,2) |
| W&B prediction panels | ec=29 | Yes (images) | Logged every 5 epochs |
| Training loss curve | ec=30 | Yes (plot) | Shows flatlined losses |
| Threshold sweep plot | ec=35 | Yes (plot) | Flat F1 across all thresholds |
| Robustness bar chart | ec=51 | Yes (plot) | All bars identical height |

### Missing Visualizations

| Visualization | Status |
|--------------|--------|
| Confusion matrix | Cell exists but no image output |
| ROC / PR curves | Cell exists but no image output |
| Sample prediction panels (Original/GT/Pred/Overlay) | Cell exists but no image output |
| Failure case analysis | Cell exists but no image output |
| Grad-CAM heatmaps | Cell exists but no image output |
| ELA visualization | Cell exists but no image output |

Several key visualization cells appear to have executed (have execution counts) but produced no visible image output. This may be because:
- matplotlib figures were not explicitly shown (`plt.show()` missing or not called in notebook context)
- Output was logged to W&B instead of inline display
- Cells encountered silent errors

### Executive Summary Placeholders

The Executive Summary section (cells 0-7) runs as cells ec=1 through ec=7 -- **before training begins**. All metric displays show placeholder text ("Final test metrics have not been computed yet"). This defeats the purpose of an executive summary.

### Verdict

**Visualization coverage is partial.** Training curves and threshold sweep plots were generated, confirming the model's failure. But critical visualizations (confusion matrix, sample predictions, Grad-CAM) are missing, which would have provided additional diagnostic insight.

---

## 7. Assignment Alignment Check

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **1. Dataset Selection** | PASS | CASIA v2.0, publicly available |
| **1. Data Pipeline** | PASS | Cleaning, preprocessing, 70/15/15 split, mask alignment |
| **1. Augmentation** | PASS | 7 transforms including JPEG compression |
| **2. Architecture** | PASS | TamperDetector documented, loss functions justified |
| **2. Resource Constraints** | PASS | Runs on Kaggle T4 GPUs |
| **3. Performance Metrics** | **FAIL** | Metrics computed but results are near-random |
| **3. Visual Results** | **PARTIAL** | Some visualizations present, key ones missing |
| **4. Single Notebook** | PASS | Everything in one .ipynb |
| **4. Model Weights** | **FAIL** | Weights saved but model is non-functional |
| **Bonus: Robustness** | **FAIL** | Tests present but Albumentations API broken + model constant |
| **Bonus: Subtle Tampering** | **FAIL** | Per-type breakdown present but results near-random |

### Verdict

**Structurally compliant but functionally non-compliant.** The notebook checks every assignment box in terms of code structure. But the assignment expects a model that "detect[s] and localize[s] tampered regions" -- this model does neither. A trained model with Tam-F1=0.13 and pixel-AUC=0.50 does not demonstrate "strong problem-solving skills" or "rigorous evaluation methodologies" in the intended sense.

---

## 8. Engineering Quality

### Strengths

| Aspect | Rating | Notes |
|--------|--------|-------|
| CONFIG centralization | A | Single dict, complete, documented |
| Reproducibility | A- | Seeds set, split determinism verified (Reproducibility section 18) |
| Checkpoint system | A | 3-file strategy (best/last/periodic) |
| Data leakage prevention | A | Explicit overlap verification |
| Evaluation suite | A | 12+ analysis features (threshold, pixel-AUC, forgery-type, shortcut, robustness) |
| AMP implementation | B+ | Edge loss bug fixed with proper autocast disabling |
| W&B integration | B | Prediction logging every 5 epochs, but offline only |

### Weaknesses

| Issue | Severity | Detail |
|-------|----------|--------|
| Executive Summary runs before training | Medium | Always shows placeholders in top-to-bottom execution |
| Model Card says "vK.11.1" | Low | Not updated from base notebook |
| `train_dice` always 0.0 | Medium | `history['train_dice'].append(0.0)` -- never computed |
| Section numbering inconsistent | Low | Markdown headers jump between numbering schemes |
| Albumentations API deprecation | Medium | Robustness transforms use deprecated parameters |
| Missing visualization outputs | Medium | Several executed cells produce no visible output |
| Quick Inference Demo unexecuted | Low | Section 19 cells have no execution count |

### Verdict

**Engineering quality is the highest in the project.** The CONFIG system, reproducibility infrastructure, checkpoint strategy, and evaluation suite are all excellent. The weaknesses are cosmetic (model card version, section numbering) or execution-order issues (executive summary placeholders). The engineering quality makes the model's failure all the more puzzling -- the infrastructure is professional-grade, but the model it supports learned nothing.

---

## 9. Roast Section

**"The Model Learned to Predict a Constant Mask, and the Notebook Validated It with a Test Suite That Confirmed the Model Does Nothing"**

Let's start with the headline: this is the synthesis architecture. The one that was supposed to combine v6.5's pretrained encoder (Tam-F1=0.41) with v8's per-sample Dice, vK.10.6's evaluation rigor, and three new components (ELA, edge loss, classification head). The prediction was Tam-F1 0.50-0.65. The result? **0.1321.** That's not a regression -- that's a collapse. vK.10.6 trained from scratch for 100 epochs hit 0.22. v8 with a broken pos_weight=30 hit 0.29. vK.11.4 with every advantage in the book hit 0.13.

**The pixel-level AUC is 0.4988.** For anyone keeping score at home, 0.50 is what you get from `random.random()`. The model's pixel predictions are literally random. You would get the same diagnostic accuracy by flipping a coin at each pixel.

**The shortcut test is the smoking gun.** When you shuffle the ground truth masks -- assigning each image some other image's mask -- the F1 score doesn't change. Not "barely changes." Not "changes within noise." **Zero delta to four decimal places.** The predictions are not correlated with image content. The model has learned to produce a fixed output pattern.

**The robustness testing is a comedy of errors.** All 8 conditions produce F1=0.1321. The notebook interprets this as "robust to perturbations." In reality, it is robust because the model is not looking at the input. Adding Gaussian noise to an image that the model ignores does not change the model's output. Also, the Albumentations API changed and the degradation transforms print deprecation warnings -- they may not even be applying. So the "robustness test" is testing whether a constant-output model produces constant output when given potentially-unmodified inputs. The answer is yes.

**The CONFIG downgrade is baffling.** vK.11.1 had `max_epochs=100, patience=20`. The one thing every prior audit agreed on was that more training time was critical -- vK.10.3b-10.5 collapsed precisely because patience=10 killed training at epoch 10, while vK.10.6 thrived with patience=30 and 100 epochs. So vK.11.4 takes the exact architecture that needs generous training... and cuts the budget in half. The model peaked at epoch 15 and flatlined for 10 more. What would have happened with 100 epochs? We'll never know.

**The Executive Summary section is peak optimism.** Eight cells of formatted markdown describing the problem, the architecture, the training strategy, and a placeholder for "Final Test Metrics." It executes as cells 1-7, before the model ever trains. In, the actual metrics would fill in. On Kaggle, which runs top-to-bottom once, it shows "not yet available" -- forever.

The Model Card still says "TamperDetector vK.11.1." If you're going to ship a broken model, at least label it correctly.

**What went wrong?** The most likely cause is that adding ELA + edge loss + classification head simultaneously created a loss landscape where the segmentation task cannot find gradient signal. The classification loss (weighted 1.5x) likely dominated encoder updates, pushing features toward image-level discrimination at the expense of pixel-level localization. The edge loss (while individually useful) added a third objective competing for the same encoder parameters. And the ELA channel -- whose signal quality on CASIA images was never validated -- may have added noise rather than information.

The lesson is harsh: combining five good ideas does not produce a good result. It produces vK.11.4.

**Score: 4/10** (model fails completely, but engineering and evaluation infrastructure are excellent)

---

## Summary

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Dataset Pipeline | **A-** | Sound, leakage-free, ELA integrated |
| Model Architecture | **B** | Correct synthesis, but multi-head loss failed |
| Training Pipeline | **B+** | All infrastructure works, model does not learn |
| Evaluation Metrics | **A** (code) / **F** (results) | Comprehensive suite proves model failure |
| Visualization | **C** | Partial coverage, key visuals missing |
| Assignment Alignment | **D** | Structurally complete, functionally non-compliant |
| Engineering Quality | **A-** | Best in project, minor cosmetic issues |
| **Overall** | **4/10** | Excellent infrastructure around a non-functional model |

### Key Metrics vs Project History

| Run | Tam-F1 | Tam-IoU | Img Acc | AUC | Pixel-AUC |
|-----|--------|---------|---------|-----|-----------|
| v6.5 (best) | **0.4101** | 0.3563 | 0.8246 | 0.8703 | — |
| vK.10.6 | 0.2213 | 0.1554 | 0.8357 | 0.9057 | 0.7083 |
| v8 (regressed) | 0.2949 | 0.2321 | 0.7190 | 0.8170 | — |
| **vK.11.4** | **0.1321** | **0.0825** | **0.4142** | **0.6434** | **0.4988** |

**vK.11.4 is the worst pretrained-encoder run in project history.**
