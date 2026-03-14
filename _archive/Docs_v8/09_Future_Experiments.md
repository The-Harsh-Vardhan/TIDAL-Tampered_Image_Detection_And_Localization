# 09 — Future Experiments

## Purpose

Structured experiment roadmap beyond v8. Each experiment traces to a specific gap identified in the audit or Run01 analysis. Organized by priority tier and expected effort.

---

## Tier 1: v8 Ablation Studies

These should run alongside or immediately after the v8 baseline. They validate that the P0 fixes work as expected.

### Experiment 1.1: pos_weight Sensitivity

**Question:** What pos_weight value maximizes tampered-only F1?

**Design:**
- Train v8 with pos_weight ∈ {5, 10, 15, 20, computed_ratio}
- Hold all other v8 settings constant
- Report tampered-only F1, optimal threshold, and copy-move F1

**Expected outcome:** pos_weight in 10–20 range should normalize threshold to ~0.3–0.5.

**Effort:** Low — only change one hyperparameter per run.

### Experiment 1.2: Scheduler Comparison

**Question:** Which scheduler produces the best convergence behavior?

**Design:**
- ReduceLROnPlateau (factor=0.5, patience=3) vs CosineAnnealingWarmRestarts (T_0=10)
- Same v8 configuration otherwise
- Track: val F1 trajectory, final epoch, overfitting onset

**Expected outcome:** ReduceLROnPlateau should extend training to 30+ epochs.

**Effort:** Low — swap one component.

### Experiment 1.3: Augmentation Ablation

**Question:** Which augmentations contribute most to robustness improvement?

**Design:**
- v8 baseline (all augmentations)
- Remove ImageCompression → measure JPEG robustness gap
- Remove GaussNoise → measure noise robustness gap
- Remove ColorJitter → measure photometric robustness gap
- Geometric-only (Run01 config) → confirm Run01's weakness

**Expected outcome:** ImageCompression has the largest impact on JPEG robustness.

**Effort:** Medium — 5 training runs.

---

## Tier 2: Architecture Comparison (v9)

These experiments require a stable v8 training pipeline. Do not attempt until v8 produces well-calibrated, non-overfitting results.

### Experiment 2.1: Encoder Swap

**Question:** Does encoder choice meaningfully affect copy-move performance?

**Design:**
- ResNet34 (v8 baseline) vs EfficientNet-B0 vs ResNet50
- Same v8 loss, augmentation, scheduler
- Report: tampered-only F1, per-forgery F1, training time, GPU memory

**Expected outcome:** Minor differences. Architecture is not the primary bottleneck.

**Effort:** Low — SMP makes encoder swaps one-line changes.

### Experiment 2.2: DeepLabV3+ Comparison

**Question:** Does atrous spatial context improve copy-move F1?

**Design:**
- `smp.DeepLabV3Plus(encoder_name='resnet34', ...)` with v8 training pipeline
- Same evaluation protocol
- Report: tampered-only F1, copy-move F1, boundary quality

**Expected outcome:** May improve large-region predictions; unlikely to solve copy-move.

**Effort:** Low — SMP provides DeepLabV3+.

### Experiment 2.3: Encoder Freezing Ablation

**Question:** Does freezing the encoder for initial epochs improve stability?

**Design:**
- Freeze encoder for 0, 2, 5 epochs before unfreezing
- Track: grad norm, val loss variance, final F1

**Expected outcome:** 2-epoch freeze may reduce early instability.

**Effort:** Low.

---

## Tier 3: Advanced Architecture & Input (v9+)

### Experiment 3.1: Forensic Input Streams

**Question:** Do non-RGB input channels improve copy-move detection?

**Design:**
- Add SRM (Spatial Rich Model) residuals as 3 additional input channels (6-channel input)
- Alternative: ELA (Error Level Analysis) as 1 additional channel (4-channel input)
- Requires modifying `in_channels` and preprocessing pipeline

**Expected outcome:** Should improve copy-move F1 if artifacts are the missing signal.

**Effort:** HIGH — requires preprocessing pipeline changes, new data loading code.

### Experiment 3.2: Lightweight Transformer Encoder

**Question:** Does transformer-style global context help?

**Design:**
- Test SegFormer-B0 or MiT-B0 encoder via SMP or HuggingFace
- Same loss, augmentation, evaluation
- Report: F1, training time, memory

**Expected outcome:** Better global context may help, but T4 memory constraints limit model size.

**Effort:** Medium — may need custom integration.

### Experiment 3.3: Learned Image-Level Head

**Question:** Does a dedicated classification branch outperform max(prob_map)?

**Design:**
- Add a global average pooling → FC → sigmoid branch from the encoder bottleneck
- Multi-task loss: segmentation_loss + λ × classification_loss
- Compare: image AUC vs heuristic baseline

**Expected outcome:** Should improve image-level AUC from 0.87.

**Effort:** Medium — requires architecture modification and multi-task loss balancing.

---

## Tier 4: Evaluation & Generalization

### Experiment 4.1: Cross-Dataset Evaluation

**Question:** Does the model generalize beyond CASIA?

**Design:**
- Train on CASIA (v8 pipeline)
- Evaluate on Coverage dataset, CoMoFoD, or NIST'16 (zero-shot)
- Report: tampered-only F1 on each external dataset

**Expected outcome:** Significant performance drop expected. Valuable for honest scope claims.

**Effort:** Medium — requires dataset preparation and evaluation code adaptation.

### Experiment 4.2: Mask Quality Sensitivity

**Question:** How much does annotation noise in CASIA affect metrics?

**Design:**
- Manually review 50 random CASIA masks for quality
- Erode/dilate ground truth masks by 1–3 pixels
- Re-evaluate Run01/v8 under perturbed masks
- Report F1 sensitivity to mask perturbation

**Expected outcome:** Establishes error bars on metric interpretation.

**Effort:** Medium.

### Experiment 4.3: Multi-Seed Variance

**Question:** How stable are results across random seeds?

**Design:**
- Train v8 with seeds {42, 123, 456, 789, 1024}
- Report: mean ± std of tampered-only F1, copy-move F1, threshold

**Expected outcome:** Establishes confidence intervals for all conclusions.

**Effort:** High — 5× training time.

---

## Experiment Priority Matrix

| Experiment | Tier | Effort | Expected Impact | When |
|---|---|---|---|---|
| 1.1 pos_weight sweep | 1 | Low | High | With v8 |
| 1.2 Scheduler comparison | 1 | Low | High | With v8 |
| 1.3 Augmentation ablation | 1 | Medium | High | After v8 baseline |
| 2.1 Encoder swap | 2 | Low | Low-Medium | After v8 validated |
| 2.2 DeepLabV3+ | 2 | Low | Medium | After v8 validated |
| 2.3 Encoder freezing | 2 | Low | Low | After v8 validated |
| 3.1 Forensic streams | 3 | High | High | v9+ |
| 3.2 Transformer encoder | 3 | Medium | Medium | v9+ |
| 3.3 Classification head | 3 | Medium | Medium | v9+ |
| 4.1 Cross-dataset | 4 | Medium | High (credibility) | v9+ |
| 4.2 Mask quality study | 4 | Medium | Medium | Anytime |
| 4.3 Multi-seed variance | 4 | High | Medium | After v8 validated |

---

## Minimum Viable Experiment Set

If time is limited, the experiments that provide the most value per unit effort:

1. **pos_weight sweep** (Tier 1.1) — validates the most important fix
2. **Augmentation ablation** (Tier 1.3) — confirms which augmentations matter
3. **Encoder swap** (Tier 2.1) — answers "is architecture the bottleneck?"
4. **Multi-seed** (Tier 4.3) — establishes confidence in all reported numbers
