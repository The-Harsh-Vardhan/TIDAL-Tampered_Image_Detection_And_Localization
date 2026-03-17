# Experiment Description — vR.P.7: ELA + Extended Training

| Field | Value |
|-------|-------|
| **Version** | vR.P.7 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.3 (ELA as input, BN unfrozen) |
| **Change** | Extend max_epochs from 25 to 50, increase early stopping patience from 7 to 10 |
| **Encoder** | ResNet-34 (ImageNet, frozen body, BatchNorm unfrozen for domain adaptation) |
| **Input** | ELA 384×384×3 (RGB ELA map) |

---

## 1. Motivation

vR.P.3 was the breakthrough experiment: switching from RGB to ELA input produced +23.74pp Pixel F1 (0.4546 → 0.6920), the single largest improvement in either track. However, P.3 has a critical property: **it was still improving when training ended.**

- Best epoch = epoch 25 (the last epoch)
- Val loss was still decreasing: 0.7340 (epoch 1) → 0.4109 (epoch 25)
- Val Pixel F1 was still increasing: 0.4051 (epoch 1) → 0.7243 (epoch 25)
- LR was reduced twice (1e-3 → 5e-4 → 2.5e-4) with room for further reduction
- No early stopping was triggered (patience_counter never reached 7)

This means P.3 was **capacity-limited by the epoch budget, not by model capacity or data quality.** The model had not converged — there were more features to learn.

vR.P.7 tests a simple hypothesis: **giving the same model more time to train will improve results.** This is the lowest-risk, highest-expected-value experiment available.

---

## 2. What Changed from vR.P.3

| Aspect | vR.P.3 | vR.P.7 (This Version) |
|--------|--------|----------------------|
| **Max epochs** | 25 | **50** |
| **Early stopping patience** | 7 | **10** |
| **NUM_WORKERS** | 2 | **4** |
| **prefetch_factor** | default (2) | **2 (explicit)** |
| **Expected training time** | ~75-100 min | **~150-200 min** |

Everything else is identical: same model, same freeze strategy, same input pipeline, same loss, same optimizer, same LR scheduler, same seed.

---

## 3. What DID NOT Change (Frozen)

- Architecture: UNet + ResNet-34 (SMP)
- Input: ELA 384×384×3 (Q=90, brightness-scaled)
- Normalization: ELA-specific mean/std (computed from training set)
- Encoder: Frozen body, BN layers unfrozen
- IN_CHANNELS = 3
- Loss: SoftBCEWithLogitsLoss + DiceLoss
- Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
- LR Scheduler: ReduceLROnPlateau(factor=0.5, patience=3)
- Batch size: 16
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling (same 3-tier fallback)
- AMP + TF32 enabled
- Checkpoint save/resume
- Evaluation: pixel-level + image-level metrics

---

## 4. Single-Variable Justification

The ONLY variables changed are:
1. `EPOCHS`: 25 → 50
2. `PATIENCE`: 7 → 10 (necessary companion — with 50 epochs, patience=7 may trigger too early if the LR scheduler needs time to find a new plateau)

These are treated as a single logical change: "extended training budget." The patience increase is not an independent variable — it is the minimum adjustment needed to allow the extended epoch budget to be utilized. Without increased patience, the first plateau after epoch 25 would trigger early stopping before the model can benefit from further LR reductions.

---

## 5. Speed Optimizations (from P.1.5)

All speed optimizations from vR.P.1.5 are carried forward:

| Optimization | Implementation |
|-------------|---------------|
| AMP (Mixed Precision) | `autocast('cuda')` + `GradScaler('cuda')` in train/val |
| TF32 math | `torch.backends.cuda.matmul.allow_tf32 = True` |
| TF32 cuDNN | `torch.backends.cudnn.allow_tf32 = True` |
| Pin memory | `pin_memory=True` on all DataLoaders |
| Persistent workers | `persistent_workers=True` on all DataLoaders |
| Non-blocking transfer | `.to(device, non_blocking=True)` in train/val |
| Fast grad zeroing | `optimizer.zero_grad(set_to_none=True)` |
| Drop last (train) | `drop_last=True` on train_loader only |
| Scaler state checkpoint | Saved/restored in checkpoint dict |

**New in P.7:** `num_workers=4` (up from 2) and `prefetch_factor=2` (explicit). These enable the CPU to stay ahead of the GPU during the longer training run.

---

## 6. Experiment Lineage

```
vR.P.0 (RGB, divg07 dataset, ELA pseudo-masks)
  |
vR.P.1 (RGB, sagnikkayalcse52 dataset, GT masks) ← proper baseline
  |          \              \
vR.P.1.5     vR.P.5         vR.P.6
(speed)      (ResNet-50)    (EfficientNet-B0)
  |
vR.P.2 (gradual unfreeze, layer3+layer4)
  |
vR.P.3 (ELA input, BN unfrozen) ← BREAKTHROUGH: Pixel F1 = 0.6920
  |         \
vR.P.4      vR.P.7 ← THIS EXPERIMENT
(4ch)       (ELA + extended training, 50 epochs)
```

Full lineage: P.0 → P.1 → P.1.5 → P.2 → P.3 → **P.7**

P.7 branches from P.3 (not P.4) because:
- P.4's 4-channel fusion added marginal improvement (+1.33pp, below ±2pp significance threshold) at the cost of significant complexity (dual normalization, conv1 unfreeze, training instability)
- P.3's simpler ELA-only pipeline is the cleaner base for testing extended training
- If P.7 succeeds, the 4-channel approach can be re-tested as a future experiment on the P.7 model

---

## 7. Risk Assessment

**Overall Risk: LOW**

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Overfitting with more epochs | MODERATE | LOW | Early stopping (patience=10) + ReduceLROnPlateau will halt training if val_loss plateaus. P.3's train-val gap at epoch 25 was small. |
| Diminishing returns | LOW | MODERATE | Even +2pp Pixel F1 would be a POSITIVE verdict. Cost is only additional compute time. |
| Training instability at low LR | LOW | LOW | ReduceLROnPlateau reduces LR gradually (0.5× factor). Model is well-behaved at lower LRs in P.3. |
| Kaggle session timeout | MODERATE | LOW | Checkpoint save/resume handles this. 50 epochs at ~3-4 min/epoch = ~150-200 min, within T4 session limits. |
| P.3's NameError bug repeats | HIGH | HIGH | **MUST FIX** the `denormalize` → `denormalize_ela` bug in visualization cells before running. |

---

## 8. Hypothesis

**H0 (null):** Extended training does not improve Pixel F1 beyond ±2pp of P.3's 0.6920.

**H1 (alternative):** Extended training improves Pixel F1 by ≥ 2pp (to ≥ 0.7120) because the model was still learning at epoch 25.

**Evidence supporting H1:**
- P.3's best epoch was the last epoch (25/25)
- Val loss and Pixel F1 were both still improving monotonically
- LR still had room for 2-3 more reductions (2.5e-4 → 1.25e-4 → 6.25e-5)
- The freeze strategy (only 3.17M trainable params) limits overfitting risk

**Evidence supporting H0:**
- Diminishing returns are common — the easy features are learned first
- The ELA signal may have a ceiling at this resolution (384×384)
- The frozen encoder may not provide sufficiently rich features for further improvement
