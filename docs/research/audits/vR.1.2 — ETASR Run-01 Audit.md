# Technical Audit: vR.1.2 — ETASR Run-01 (Data Augmentation)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-1-2-etasr-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (15,511 MB VRAM) |
| **Cells** | 31 total (19 code, 12 markdown) |
| **Executed** | 19 of 19 code cells (all executed) |
| **Training** | 6 epochs (early stopped), best at epoch 1 |
| **Version** | vR.1.2 — Data Augmentation |
| **Change** | `ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=15, fill_mode='nearest')` |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |
| **Verdict** | **NEGATIVE — Significant regression across all metrics. Change rejected.** |

---

## 1. Notebook Overview

vR.1.2 adds real-time data augmentation to the training pipeline via Keras `ImageDataGenerator`. The augmentation includes horizontal flip, vertical flip, and random rotation ±15°. All other parameters are frozen from vR.1.1.

### Notebook Structure (31 cells)

Same 12-section structure as vR.1.1. Changed cells:
- Cell 0: Updated title and pipeline diagram (adds "Augment" step)
- Cell 1: Updated change log (vR.1.1 → vR.1.2 diff table)
- Cell 2: Added `ImageDataGenerator` import, `AUGMENTATION` config dict
- Cell 15: Updated training pipeline markdown (mentions augmentation)
- Cell 17: Replaced `model.fit(X_train, ...)` with `model.fit(train_generator, ...)`
- Cell 28: Updated ablation tracking table with vR.1.1 historical row
- Cell 29: Updated discussion section
- Cell 30: Updated model save filename

### Structural Completeness

| Requirement | Present? | Quality |
|-------------|----------|---------|
| Introduction | ✅ | Good — change clearly described, pipeline updated |
| ETASR architecture explanation | ✅ | Unchanged from vR.1.1 |
| Dataset explanation | ⚠️ Partial | Still no CASIA provenance description |
| Preprocessing pipeline | ✅ | ELA + augmentation both explained |
| Model architecture | ✅ | Unchanged — correctly frozen |
| Training pipeline | ✅ | Augmentation method clearly documented |
| Evaluation metrics | ✅ | Per-class, macro, ROC-AUC, confusion matrix |
| Results discussion | ✅ | Discussion previews next version |

### Missing / Weak Sections

1. **No augmentation visualization** — Should show examples of augmented ELA images to verify augmentation doesn't destroy ELA signal
2. **Dataset description still thin** — No CASIA 2.0 provenance
3. **No per-epoch formatted table** — Only raw Keras progress bar output

---

## 2. Strengths

1. **Single controlled change** — Only augmentation was added. All other parameters frozen. Clean ablation.
2. **Augmentation correctly applied to training only** — Validation and test sets are not augmented. No data leakage.
3. **All 19 cells execute without errors** — Clean run, no crashes.
4. **Proper evaluation maintained** — Per-class metrics, macro average, ROC-AUC, confusion matrix on held-out test set all carried forward from vR.1.1.
5. **Version tracking** — `VERSION='vR.1.2'`, `CHANGE` string, ablation comparison table all present.
6. **Model saved** — `vR.1.2_ela_cnn_model.keras` persisted.
7. **Cumulative change log** — Cell 1 lists both vR.1.1 and vR.1.2 changes for traceability.

---

## 3. Weaknesses and Issues

### MAJOR

| ID | Issue | Severity | Impact |
|----|-------|----------|--------|
| **M1** | **Catastrophic performance regression** — Test accuracy dropped from 88.38% to 85.53% (-2.85pp). Macro F1 dropped from 0.8805 to 0.8505 (-3.00pp). ROC-AUC dropped from 0.9601 to 0.9011 (-0.059). Every single metric degraded. | **Critical** | Augmentation harmed the model |
| **M2** | **Training never improved beyond epoch 1** — Best val_accuracy was 0.8531 at epoch 1. The model declined every subsequent epoch. Early stopping restored epoch 1 weights after only 6 epochs. This means the augmented data actively prevented learning. | **Critical** | Augmentation incompatible with this LR/architecture |
| **M3** | **FN rate increased to 16.6%** — Up from 11.7% (vR.1.1). The model now misses 128 of 769 tampered images (1 in 6). This is the worst tampered recall in the entire ablation series. | High | Tampered detection significantly worse |
| **M4** | **No localization** — Still classification only. | Critical | Assignment requirement failure |

### MINOR

| ID | Issue | Severity | Impact |
|----|-------|----------|--------|
| m1 | **29 training samples dropped per epoch** — `steps_per_epoch = 8829 // 32 = 275` drops remainder of 29. Minor data loss per epoch. | Low | Negligible |
| m2 | **No augmented sample visualization** — Critical omission. Cannot verify whether augmentation preserves ELA signal integrity. | Medium | Debugging blind spot |
| m3 | **Keras display artifacts** — Epochs 2, 4, 6 show "0s / 2ms/step" timing. Kaggle progress bar rendering issue, not affecting results. | Low | Display-only |
| m4 | **FP rate increased to 13.0%** — Up from 11.6% (vR.1.1). Authentic classification also degraded. | Medium | Both classes worse |
| m5 | **Warnings suppressed** — Same blanket `filterwarnings('ignore')` inherited from vR.1.1. | Low | Debugging hazard |

---

## 4. Paper Reproduction Fidelity

Architecture is unchanged from vR.1.1. All layers match ETASR Table III exactly.

| Aspect | Verdict |
|--------|---------|
| ELA preprocessing | Exact match + augmentation (new) |
| Image size 128×128 | Exact match (frozen) |
| CNN architecture | Exact match (frozen) |
| Optimizer Adam(0.0001) | Exact match (frozen) |
| Dropout 0.25/0.5 | Exact match (frozen) |
| Loss categorical_crossentropy | Exact match (frozen) |
| Data augmentation | **Deviation — paper does not specify augmentation** |

The paper does not mention data augmentation. Adding it is an intentional ablation change, not a fidelity issue. However, the paper may have used augmentation without documenting it.

**Paper Reproduction Score: 7/10** — Architecture faithful, but augmentation caused results to diverge further from paper claims (85.53% vs 96.21% = 10.68pp gap, worse than vR.1.1's 7.83pp gap).

---

## 5. Dataset Pipeline Review

| Aspect | vR.1.1 | vR.1.2 | Change |
|--------|--------|--------|--------|
| Total images | 12,614 | 12,614 | Unchanged |
| Train split | 8,829 (70%) | 8,829 (70%) | Unchanged |
| Val split | 1,892 (15%) | 1,892 (15%) | Unchanged |
| Test split | 1,893 (15%) | 1,893 (15%) | Unchanged |
| Augmentation | None | H-flip, V-flip, Rot ±15° | **New** |
| Augmentation target | N/A | Training only | Correct |
| Training method | `model.fit(X_train)` | `model.fit(train_generator)` | Changed to generator |
| Steps per epoch | Auto (276) | 275 (floor division) | 29 samples dropped |

### Data Leakage Check

- **No leakage.** Augmentation applied only to training batches. Val/test data untouched.
- Split ratios and class distributions identical to vR.1.1.

### Critical Pipeline Concern

**Augmentation may destroy ELA signal.** ELA maps encode compression artifacts as subtle brightness patterns. Rotation with `fill_mode='nearest'` introduces edge artifacts (border pixels replicated). More critically, horizontal/vertical flips may be irrelevant for ELA analysis — the spatial pattern of compression artifacts is not orientation-dependent. The augmentation may be adding noise without meaningful diversity.

---

## 6. Training Pipeline Review

### Configuration

| Parameter | Value | Status |
|-----------|-------|--------|
| Optimizer | Adam(lr=0.0001) | Frozen |
| Loss | categorical_crossentropy | Frozen |
| Batch size | 32 | Frozen |
| Max epochs | 50 | Frozen |
| Early stopping | val_accuracy, patience=5 | Frozen |
| Steps per epoch | 275 | New (generator-based) |
| Augmentation | H-flip, V-flip, Rot ±15° | **NEW** |

### Epoch-by-Epoch Training Log

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|-------|-----------|-----------|----------|---------|-------|
| **1** | 0.5021 | 0.7132 | 0.3824 | **0.8531** | **Best epoch — restored** |
| 2 | 0.2928 | 0.9062 | 0.3975 | 0.8478 | Val declining |
| 3 | 0.3776 | 0.8462 | 0.4273 | 0.8330 | Val declining |
| 4 | 0.2066 | 0.9688 | 0.4387 | 0.8277 | Val declining |
| 5 | 0.3472 | 0.8600 | 0.4630 | 0.8177 | Val declining |
| 6 | 0.2872 | 0.9062 | 0.4418 | 0.8288 | Early stopping triggers |

### Training Dynamics Analysis

**This is a failed training run.** Key observations:

1. **Best epoch is epoch 1** — The model never improved after the very first epoch. This has never happened in any prior run (all had best epoch at 8).

2. **Val accuracy monotonically decreased** — 0.8531 → 0.8478 → 0.8330 → 0.8277 → 0.8177 → 0.8288. Continuous decline with one minor bounce at epoch 6.

3. **Val loss monotonically increased** — 0.3824 → 0.3975 → 0.4273 → 0.4387 → 0.4630 → 0.4418. The model got progressively worse at generalization.

4. **Train accuracy oscillated wildly** — 0.7132, 0.9062, 0.8462, 0.9688, 0.8600, 0.9062. This is augmentation-induced stochasticity — each batch sees different random transforms, causing noisy gradient estimates.

5. **Only 6 epochs ran** (vs 13 in vR.1.1) — Early stopping correctly detected no improvement after epoch 1.

### Root Cause Analysis

**The learning rate (0.0001) is too low for augmented training.** Without augmentation, the model sees the same images every epoch and can make steady progress. With augmentation, each batch contains randomly transformed images, creating a noisier loss landscape. The current learning rate fails to converge fast enough to learn the augmented distribution before early stopping kills training.

Additionally, the massive Flatten→Dense(256) layer (29.5M params) amplifies this problem — the layer memorizes exact pixel patterns rather than learning rotation/flip-invariant features. Augmented images appear as "new" data to this layer, confusing it.

---

## 7. Performance Summary

### Final Test Metrics (Epoch 1 Weights)

| Metric | Authentic | Tampered | Macro |
|--------|-----------|----------|-------|
| **Precision** | 0.8843 | 0.8145 | 0.8494 |
| **Recall** | 0.8701 | 0.8336 | 0.8518 |
| **F1** | 0.8771 | 0.8239 | 0.8505 |

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 85.53% |
| **ROC-AUC** | 0.9011 |
| **Macro F1** | 0.8505 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|---|---|---|
| **True Au** (1,124) | 978 | 146 |
| **True Tp** (769) | 128 | 641 |

| Rate | vR.1.1 | vR.1.2 | Delta |
|------|--------|--------|-------|
| FP rate | 11.6% | 13.0% | +1.4pp worse |
| FN rate | 11.7% | 16.6% | **+4.9pp worse** |

### Performance Comparison: vR.1.1 vs vR.1.2

| Metric | vR.1.1 (Parent) | vR.1.2 (This Run) | Delta | Verdict |
|--------|-----------------|-------------------|-------|---------|
| Test Accuracy | 88.38% | 85.53% | **-2.85pp** | **NEGATIVE** |
| Au Precision | 0.9170 | 0.8843 | -0.0327 | Worse |
| Au Recall | 0.8843 | 0.8701 | -0.0142 | Worse |
| Au F1 | 0.9004 | 0.8771 | -0.0233 | Worse |
| Tp Precision | 0.8393 | 0.8145 | -0.0248 | Worse |
| Tp Recall | 0.8830 | 0.8336 | **-0.0494** | **Significantly worse** |
| Tp F1 | 0.8606 | 0.8239 | -0.0367 | Worse |
| Macro F1 | 0.8805 | 0.8505 | **-0.0300** | **NEGATIVE** |
| ROC-AUC | 0.9601 | 0.9011 | **-0.0590** | **Significantly worse** |
| FP rate | 11.6% | 13.0% | +1.4pp | Worse |
| FN rate | 11.7% | 16.6% | +4.9pp | **Much worse** |
| Epochs | 13 (best 8) | 6 (best 1) | — | Training failed |

**Every single metric regressed.** This is an unambiguous negative result.

### vs Paper Claims

| Metric | Paper | vR.1.2 | Gap |
|--------|-------|--------|-----|
| Accuracy | 96.21% | 85.53% | **-10.68pp** (widened from -7.83pp) |

---

## 8. Result Analysis

### Did vR.1.2 Achieve Its Goal?

**No.** The stated goal was to "reduce overfitting and improve test-set performance." Instead:
- Accuracy dropped 2.85pp
- Overfitting was replaced by underfitting (model couldn't learn at all)
- ROC-AUC dropped 0.059 (from excellent to merely good)
- Training was effectively useless (best at epoch 1)

### Overfitting Indicators

| Indicator | vR.1.1 | vR.1.2 | Assessment |
|-----------|--------|--------|------------|
| Train-val gap at best | 4.25pp | ~13pp (epoch 1: train 71% vs val 85%) | **Reversed** — model couldn't even overfit |
| Val loss divergence | Yes (after epoch 8) | Yes (from epoch 2 onward) | Worse |
| Late-epoch collapse | Epochs 12–13 | N/A (only 6 epochs) | Not applicable |
| Best epoch | 8 | **1** | Training failed |

### Why Augmentation Failed

Three reinforcing causes:

1. **Learning rate too low for noisy gradients** — Augmentation introduces stochasticity. At lr=0.0001, the optimizer cannot follow the noisier loss surface. The model makes progress in epoch 1 (on the raw data in the generator's first pass), then struggles to improve as augmented variants dominate subsequent batches.

2. **Architecture incompatibility** — The Flatten→Dense(256) layer stores 29.5M parameters that encode pixel-exact spatial patterns. Flipped and rotated ELA images activate completely different neurons in this layer. The architecture lacks spatial invariance.

3. **ELA signal fragility** — ELA maps encode compression artifacts as precise spatial brightness patterns with specific pixel-level structure. Rotation fills borders with `nearest` values, creating artificial edges. The augmentation may destroy the very signal the model is trying to learn.

### Ablation Protocol Decision

Per the master ablation plan:
> **NEGATIVE** (−): Test accuracy or macro F1 dropped by > 0.5%

vR.1.2 dropped 2.85pp accuracy and 3.00pp macro F1. This is **NEGATIVE**.

Per the plan:
> If negative: the change is rejected and future versions branch from the last positive version

**vR.1.2 augmentation is REJECTED. vR.1.3 branches from vR.1.1 (the last positive version).**

---

## 9. Paper Reproduction Score

**7/10** — Architecture still faithful, but results have diverged further from paper (10.68pp gap vs 7.83pp). Augmentation was not mentioned in the paper.

---

## 10. Assignment Readiness Score

**4/10** — Below vR.1.1 (5/10). Performance regression makes the notebook less ready for submission.

| Requirement | Status |
|-------------|--------|
| Image tampering detection | ⚠️ 85.53% accuracy (below vR.1.1) |
| ELA preprocessing | ✅ Implemented and visualized |
| CNN model | ✅ Paper architecture reproduced |
| Proper evaluation | ✅ Test set, per-class, ROC-AUC |
| Data augmentation | ✅ Implemented (but harmful) |
| Class imbalance handling | ❌ Missing |
| Localization / masks | ❌ Missing |

---

## 11. Recommended Improvements

### Immediate Action: Skip to vR.1.3 (Class Weights)

Since augmentation is rejected, proceed with the next item in the ablation roadmap: **class weights**. This branches from vR.1.1 (not vR.1.2).

Class weights address the 1.46:1 class imbalance (W7) and may:
- Balance FP/FN rates
- Improve tampered precision and recall
- Be a lower-risk change than augmentation

### Future: Revisit Augmentation After Architecture Changes

Augmentation may work after:
- **vR.1.4 (BatchNorm)** — Stabilizes training, may tolerate augmentation noise
- **vR.1.7 (GAP replaces Flatten)** — Dramatically reduces param count, adds spatial invariance

After architectural improvements, a future version could re-test augmentation with the improved architecture.

### Alternative Augmentation Strategies (for future consideration)

If augmentation is re-attempted:
1. **Flip-only (no rotation)** — Rotation with fill_mode may damage ELA signal
2. **Higher learning rate (0.001)** — May converge faster with noisy augmented data
3. **Warm-up training** — Train without augmentation for 5 epochs, then enable augmentation
4. **Lighter augmentation** — Only horizontal flip, no rotation
