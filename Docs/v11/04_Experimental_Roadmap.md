# Docs11: Experimental Roadmap

A sequenced experiment plan with dependency ordering, GPU budgets, and success/failure criteria. Each phase builds on the previous one, with explicit gates and fallback strategies.

---

## Dependency Graph

```
Phase 0: Pre-Training Validation (CPU, ~10 min)
  ├── E0.1  Data leakage path check
  └── E0.2  pHash near-duplicate check
  [GATE: Zero cross-split leaks. If leaks found → fix splits before any training.]

Phase 1: Baseline Run (GPU, ~3 hrs on 2×T4)
  └── E1.1  Run vK.10.5 as-is for 50 epochs
  [GATE: Tampered Dice > 0.40. If < 0.20 after 20 epochs → abort, investigate.]

Phase 2: Free Evaluation Wins (Inference only, ~30 min)
  ├── E2.1  Threshold sweep on E1.1 model
  ├── E2.2  Forgery-type breakdown
  ├── E2.3  Mask-size stratification
  ├── E2.4  Confusion matrix + ROC/PR curves
  ├── E2.5  Failure case analysis
  └── E2.6  Shortcut learning checks
  [All E2.x are independent of each other; all depend on E1.1]
  [GATE: Record all metrics. This is the FALLBACK SUBMISSION if Phase 3 fails.]

Phase 3: Architecture Upgrade (GPU, ~4-8 hrs total)
  E3.1  SMP UNet + pretrained ResNet34 + classification head
    └── E3.2  E3.1 + ELA 4th channel
         └── E3.3  E3.2 + Edge supervision loss
  [Sequential: each builds on the last.]
  [GATE: At each step, tampered Dice must not regress > 0.05 from previous.]

Phase 4: Full Evaluation on Best Model (Inference, ~1 hr)
  ├── E4.1  Threshold sweep on Phase 3 best model
  ├── E4.2  Robustness testing (8 conditions)
  ├── E4.3  Grad-CAM explainability
  └── E4.4  Full eval suite (forgery breakdown + mask stratification + failure analysis + shortcuts)
  [All E4.x are independent of each other; all depend on Phase 3 best model.]

Phase 5: Ablations — Optional (GPU, ~8+ hrs)
  ├── E5.1  Pretrained vs from-scratch (compare E1.1 vs E3.1)
  ├── E5.2  ELA contribution (compare E3.1 vs E3.2)
  ├── E5.3  Edge loss contribution (compare E3.2 vs E3.3)
  └── E5.4  Gradient accumulation (run E3.3 + accum steps=4)
  [All E5.x are independent of each other.]

Phase 6: Colab Verification (GPU, ~1-2 hrs)
  └── E6.1  Run final notebook end-to-end on Colab single T4
  [GATE: Must complete without errors. This is the submission verification.]
```

---

## Phase 0: Pre-Training Validation

### E0.1: Data Leakage Path Check

**Goal:** Verify zero overlap between train/val/test image paths.

**Method:**
```python
train_paths = set(train_df['image_path'])
val_paths = set(val_df['image_path'])
test_paths = set(test_df['image_path'])
assert len(train_paths & val_paths) == 0, "Train/val overlap!"
assert len(train_paths & test_paths) == 0, "Train/test overlap!"
assert len(val_paths & test_paths) == 0, "Val/test overlap!"
print("Data leakage check: PASSED")
```

**Success criteria:** All assertions pass.
**Abort criteria:** If overlap found, fix split logic before proceeding.

### E0.2: pHash Near-Duplicate Check

**Goal:** Detect near-duplicate images across splits (e.g., same source image with different manipulations).

**Method:** Compute perceptual hashes for all images, flag pairs with Hamming distance < 10 that appear in different splits.

**Success criteria:** No critical duplicates. Low-distance pairs documented if found.

---

## Phase 1: Baseline Run

### E1.1: Run vK.10.5 As-Is

**Goal:** Establish a baseline with the current architecture and configuration.

**Configuration:** Exact vK.10.5 CONFIG — 256×256 RGB, custom UNet, 50 epochs, CosineAnnealing, AMP, early stopping on tampered Dice.

**Metrics to record:**
- Image accuracy, ROC-AUC
- Tampered-only: Dice, IoU, F1
- All-sample: Dice, IoU, F1
- Training curves (loss, accuracy, dice, LR)
- Total training time
- Convergence epoch

**Success criteria:** Tampered Dice > 0.40 at convergence.
**Concern threshold:** If tampered Dice < 0.20 after 20 epochs, investigate.
**Abort criteria:** If training loss does not decrease for 10 consecutive epochs.

**Estimated time:** 2-3 hours on 2×T4, 4-5 hours on 1×T4.

---

## Phase 2: Free Evaluation Wins

### E2.1: Threshold Sweep

**Goal:** Find optimal segmentation threshold.
**Method:** Sweep 0.05-0.80 in 0.05 steps on validation set. Select threshold maximizing tampered-only F1.
**Expected benefit:** +0.05-0.15 on all segmentation metrics (free improvement).
**Complexity:** Very Low (~30 lines).

### E2.2: Forgery-Type Breakdown

**Goal:** Report separate metrics for splicing vs copy-move.
**Method:** Parse CASIA filenames (`Tp_S_*` = splicing, `Tp_D_*` = copy-move).
**Expected benefit:** Reveals copy-move weakness (expected F1 < 0.20 based on v8 results).
**Complexity:** Low (~30 lines).

### E2.3: Mask-Size Stratification

**Goal:** Report metrics bucketed by tampered region size.
**Buckets:** Tiny (<2%), Small (2-5%), Medium (5-15%), Large (>15%).
**Expected benefit:** Reveals size-dependent failure modes.
**Complexity:** Low (~40 lines).

### E2.4: Confusion Matrix + ROC/PR Curves

**Goal:** Standard classification evaluation plots.
**Method:** `sklearn` confusion_matrix, roc_curve, precision_recall_curve.
**Complexity:** Very Low (~25 lines).

### E2.5: Failure Case Analysis

**Goal:** Display 10 worst predictions with GT mask, predicted mask, metadata.
**Method:** Sort test samples by per-sample F1, display worst 10.
**Complexity:** Low (~40 lines).

### E2.6: Shortcut Learning Checks

**Goal:** Verify model learned forensic features, not dataset artifacts.
**Method:**
1. Mask randomization — shuffle masks, re-evaluate. F1 should drop sharply.
2. Boundary sensitivity — erode/dilate predictions 1px, check metric stability.
**Complexity:** Low (~40 lines).

---

## Phase 3: Architecture Upgrade

### E3.1: SMP UNet + Pretrained ResNet34

**Goal:** Replace from-scratch encoder with ImageNet-pretrained backbone.

**Changes:**
- Replace `UNetWithClassifier` with `TamperDetector` (SMP UNet + classification head)
- Add `pip install segmentation-models-pytorch` to setup cell
- Update CONFIG: `in_channels=3`, `encoder_name='resnet34'`
- Add differential LR: encoder=1e-4, decoder=1e-3
- Add encoder freeze for first 2 epochs

**Success criteria:** AUC > 0.85 (exceeds v8's 0.817).
**Go/no-go:** If AUC < 0.80, proceed to E3.2 anyway (ELA may help).

**Estimated time:** 3-4 hours on 2×T4.

### E3.2: + ELA 4th Channel

**Goal:** Add Error Level Analysis as forensic preprocessing.

**Changes (on top of E3.1):**
- Add `compute_ela()` function
- Modify dataset `__getitem__` to compute and stack ELA channel
- Update CONFIG: `in_channels=4`
- SMP handles first-layer weight adaptation automatically

**Success criteria:** Tampered F1 improves by > 0.03 over E3.1.
**Go/no-go:** If no improvement, ELA may not help this architecture. Document finding and continue.

**Estimated time:** 3-4 hours on 2×T4 (full retrain).

### E3.3: + Edge Supervision Loss

**Goal:** Add boundary-aware loss component.

**Changes (on top of E3.2):**
- Add `edge_loss()` function using Sobel operator
- Modify total loss: add `+ γ * edge_loss(seg_logits, masks)`
- Add CONFIG parameter: `edge_loss_weight=0.3`

**Success criteria:** Boundary quality visually improves, tampered F1 does not regress.
**Go/no-go:** If training destabilizes, reduce γ to 0.1 or disable.

**Estimated time:** 3-4 hours on 2×T4 (full retrain).

---

## Phase 4: Full Evaluation on Best Model

Apply the complete evaluation suite to the best model from Phase 3.

### E4.1: Threshold Sweep (repeat of E2.1 on new model)
### E4.2: Robustness Testing

**Method:** For each of 8 conditions, apply degradation to test images, run inference, compute tampered-only metrics, report delta from clean baseline.

**Conditions:**
| # | Degradation | Parameters |
|---|---|---|
| 1 | JPEG compression | QF=70 |
| 2 | JPEG compression | QF=50 |
| 3 | Gaussian noise | σ=10 |
| 4 | Gaussian noise | σ=25 |
| 5 | Gaussian blur | kernel=3 |
| 6 | Gaussian blur | kernel=5 |
| 7 | Resize-back | 0.75× |
| 8 | Resize-back | 0.50× |

**Output:** Grouped bar chart showing tampered-only F1 per condition + clean baseline.

### E4.3: Grad-CAM Explainability
**Method:** Hook encoder's deepest layer, compute gradient-weighted activation maps, overlay on sample images (3 authentic, 3 tampered with correct predictions, 3 tampered with incorrect).

### E4.4: Full Evaluation Suite
Repeat E2.2-E2.6 on the Phase 3 model.

---

## Phase 5: Ablations (Optional)

Each ablation compares two training runs to quantify the contribution of a specific component.

| Ablation | Compare | Hypothesis |
|---|---|---|
| E5.1 | E1.1 vs E3.1 | Pretrained encoder adds +0.10 AUC |
| E5.2 | E3.1 vs E3.2 | ELA adds +0.03-0.10 tampered F1 |
| E5.3 | E3.2 vs E3.3 | Edge loss improves boundary quality |
| E5.4 | E3.3 vs E3.3+accum | Gradient accumulation stabilizes training |

**Total GPU time for ablations:** ~12-16 hours (3-4 training runs × 3-4 hours each).

---

## Phase 6: Colab Verification

### E6.1: End-to-End Colab Run

**Goal:** Verify the final notebook runs from top to bottom on a single Colab T4 GPU.

**Options:**
- Full training (slow, 5-7 hours) — proves end-to-end capability
- Load pretrained weights + inference/evaluation only (fast, 30 min) — sufficient if weights are hosted

**Success criteria:** Zero runtime errors. All outputs generated. All visualizations render.

---

## GPU Budget Summary

| Phase | Est. Time (2×T4 Kaggle) | Est. Time (1×T4 Colab) |
|---|---|---|
| Phase 0 | 10 min | 10 min |
| Phase 1 | 2-3 hours | 4-5 hours |
| Phase 2 | 30 min | 30 min |
| Phase 3 | 4-8 hours (1-3 runs) | 7-14 hours (1-3 runs) |
| Phase 4 | 1 hour | 1 hour |
| Phase 5 | 8-12 hours | 12-16 hours |
| Phase 6 | 1-2 hours | 1-2 hours |
| **Minimum (Phases 0-4)** | **8-13 hours** | **13-21 hours** |
| **Full (Phases 0-6)** | **17-27 hours** | **25-39 hours** |

Kaggle provides 30 GPU-hours per week. The minimum path (Phases 0-4) fits in a single weekly quota. The full path including ablations requires two weeks.

---

## Fallback Strategy

If Phase 3 fails (pretrained encoder does not improve):

1. Use the vK.10.5 baseline from Phase 1
2. Apply all Phase 2 evaluation improvements (threshold sweep, forgery breakdown, etc.)
3. Add robustness testing and Grad-CAM from Phase 4
4. Submit with: vK.3-level metrics + v8-level evaluation methodology + vK.10.5 engineering

This fallback produces a notebook that scores 70-80/100 even without architectural upgrades — strong enough for a competitive submission.
