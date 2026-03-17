# 07 — Risk Assessment

## Purpose

Identify risks that could undermine v9 results or create misleading conclusions. Every risk has a mitigation already planned (or explicitly flagged as accepted).

---

## Risk 1: Dataset Shortcut Learning

**Description:** CASIA v2.0 may contain systematic differences between tampered and authentic images that the model exploits instead of learning genuine forgery localization. Known shortcuts include filename patterns, compression artifacts from dataset assembly, and color distribution shifts.

**Severity:** HIGH

**Evidence from Run01:**
- Optimal threshold (0.1327) is far below 0.5, suggesting the model does not produce confident predictions on truly tampered regions
- Overfitting onset at epoch 15 despite moderate augmentation
- Copy-move F1 significantly lower than splicing F1 (0.31 vs 0.59), which could reflect distribution imbalance or shortcut exploitation

**v9 Mitigations:**
| Mitigation | Status | Expected Effect |
|---|---|---|
| Mask randomization test | Approved | Directly measures shortcut reliance: if F1 drops <0.15, the model exploits image-level cues, not localization |
| pHash near-duplicate check | Approved | Detects content leakage across splits |
| ELA auxiliary channel | Approved | Provides a compression-history signal, reducing the model's need to infer it from pixel-level artifacts that may not generalize |
| Corrected dataset framing | Approved | Documents CASIA as a "chosen baseline with known limitations," not a definitive dataset |

**Residual risk:** Even with all mitigations, CASIA v2.0 is a limited benchmark. Results may not transfer to real-world tampered images. This is accepted and documented.

---

## Risk 2: Metric Inflation

**Description:** Reporting aggregate metrics (including authentic images with trivially correct all-zero predictions) inflates apparent performance. A model that simply predicts "nothing is tampered" achieves high aggregate F1 because authentic images dominate the dataset.

**Severity:** HIGH

**Evidence from Run01:**
- Aggregate Pixel F1 was meaningful only because tampered-only reporting was already adopted in v8
- Image-level detection via heuristic (max(prob_map) > τ) masked the lack of a principled detection mechanism

**v9 Mitigations:**
| Mitigation | Status | Expected Effect |
|---|---|---|
| Tampered-only F1 as primary metric | Retained | All headline numbers exclude authentic images |
| Learned classification head | Approved | Replaces heuristic with trainable image-level detection |
| Per-forgery-type F1 reporting | Retained | Copy-move and splicing reported separately — prevents splicing performance from hiding copy-move weakness |
| PR curves | Approved | Reveal the full precision-recall tradeoff, not just a single operating point |
| Multi-seed validation (3 seeds) | Approved | mean ± std prevents cherry-picking the best run |

**Residual risk:** Within tampered-only evaluation, large tampered regions still contribute disproportionately to F1. Mask-size stratified reporting (retained from v8) helps but does not fully resolve this.

---

## Risk 3: ELA Channel Overfitting

**Description:** ELA highlights compression inconsistencies, but many CASIA v2.0 images may share the same compression history, making ELA uninformative or misleading on some samples. The model could overfit to ELA patterns specific to this dataset.

**Severity:** MEDIUM

**Mitigations:**
- ImageCompression augmentation (QF 50–95) forces the model to handle varied compression levels
- ELA quality parameter (QF 90) is fixed and documented — not tuned per-dataset
- Augmentation ablation experiment will test model with vs without photometric augmentation, indirectly revealing ELA sensitivity

**Residual risk:** If CASIA v2.0 images all share similar compression history, ELA may add noise rather than signal. In that case, v9 results should show little improvement over v8. This is acceptable — the ablation reveals it.

---

## Risk 4: Multi-Task Loss Balancing

**Description:** The dual-task architecture introduces λ (classification loss weight). A poorly chosen λ can degrade segmentation quality by diverting capacity to the classification task, or make the classification head useless if λ is too small.

**Severity:** MEDIUM

**Mitigations:**
- Start with λ=0.5 (common default in multi-task forensic literature)
- Monitor both seg_loss and cls_loss via W&B — if one dominates, adjust λ
- Classification head is deliberately simple (GAP → FC) to minimize capacity diversion

**Residual risk:** λ may need manual adjustment mid-training. Not a fundamental risk — it's a hyperparameter that can be swept if time permits.

---

## Risk 5: Edge Loss Interaction with Small Masks

**Description:** Edge loss penalizes boundary pixels heavily. For very small tampered regions, nearly all pixels are "boundary" pixels, which could distort the loss landscape.

**Severity:** LOW-MEDIUM

**Mitigations:**
- edge_loss_lambda (0.3) is kept small relative to the main loss
- Morphological dilation/erosion kernel size (3) limits the definition of "boundary" to a thin band
- v8's mask-size stratified evaluation will reveal if small-mask performance degrades

**Residual risk:** Acceptable. Edge loss is a refinement — if it harms small masks, reduce edge_loss_lambda or disable it.

---

## Risk 6: Training Time Budget on Colab T4

**Description:** Adding ELA computation, dual-task forward pass, and edge loss increases per-step cost. Multi-seed validation (3 seeds × 50 epochs) and DeepLabV3+ comparison further increase total GPU time. Colab T4 sessions have a ~12-hour limit.

**Severity:** MEDIUM

**Estimated time budget:**
| Experiment | Seeds | Epochs | Estimated Time |
|---|---|---|---|
| v9 full pipeline (U-Net) | 3 | 50 | ~6h total |
| DeepLabV3+ comparison | 1 | 50 | ~2.5h |
| Augmentation ablation | 1 | 50 | ~2h |
| Evaluation + visualization | — | — | ~0.5h |
| **Total** | | | **~11h** |

**Mitigations:**
- Use Kaggle T4 × 2 or split across sessions for multi-seed runs
- DeepLabV3+ comparison is a comparison experiment, not mandatory — can be deferred if time is tight
- Early stopping (patience=10) may terminate runs before 50 epochs

**Residual risk:** If Colab sessions disconnect, partial training is lost. Checkpoint saving every 5 epochs mitigates this.

---

## Risk 7: Copy-Move Remains Weak

**Description:** Copy-move forgeries duplicate existing image content rather than introducing foreign content. The model may fundamentally struggle with copy-move detection because the duplicated region has identical statistics to the source.

**Severity:** MEDIUM (expected, documented)

**Evidence:**
- Run01 copy-move F1 = 0.31 vs splicing F1 = 0.59
- ELA provides weak signal on copy-move (same compression history as source region)
- No frequency-domain features or noise residuals approved for v9

**Mitigations:**
- ELA may still help if re-encoded regions have subtle compression differences
- Edge loss may improve boundary detection for copy-move regions
- Per-forgery-type loss tracking will quantify the gap

**Accepted outcome:** Copy-move F1 of 0.38-0.45 is a reasonable v9 target. Full copy-move parity with splicing requires techniques deferred to v10+ (SRM noise residuals, multi-branch architectures).

---

## Risk 8: Confirmation Bias in Evaluation

**Description:** Running multiple experiments and selecting the best result without pre-registering success criteria leads to cherry-picked conclusions.

**Severity:** LOW-MEDIUM

**Mitigations:**
- Success thresholds defined in advance (in `06_Notebook_V9_Implementation_Plan.md`):
  - Tampered-only F1 > 0.55
  - Image-level AUC > 0.88
  - Copy-move F1 > 0.38
  - Mask randomization drop > 0.15
- Multi-seed runs with reported mean ± std
- All runs logged to W&B with identical CONFIG except seed

**Residual risk:** Three seeds is minimal. More seeds would be better but are infeasible within the time budget.

---

## Risk Summary Matrix

| Risk | Severity | Mitigation Quality | Residual |
|---|---|---|---|
| Dataset shortcut learning | HIGH | Strong (4 mitigations) | Medium — CASIA limitations remain |
| Metric inflation | HIGH | Strong (5 mitigations) | Low |
| ELA channel overfitting | MEDIUM | Adequate | Medium — dataset-dependent |
| Multi-task loss balancing | MEDIUM | Adequate | Low |
| Edge loss + small masks | LOW-MEDIUM | Adequate | Low |
| Colab T4 time budget | MEDIUM | Adequate | Medium — session limits |
| Copy-move weakness | MEDIUM | Partial | Medium — fundamental limitation |
| Confirmation bias | LOW-MEDIUM | Good | Low |

---

## Bottom Line

The highest risks (shortcut learning, metric inflation) have the strongest mitigations. The mask randomization test is the single most important diagnostic in v9 — it directly answers whether the model has learned genuine localization or exploits dataset shortcuts. Copy-move weakness is an accepted limitation with honest documentation rather than a solvable problem at this stage.
