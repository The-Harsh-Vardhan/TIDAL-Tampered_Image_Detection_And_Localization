# Docs9 — Project Evolution Summary

## Purpose

Docs9 is the **technical decision record** for the Tampered Image Detection and Localization project. It determines which improvements from the accumulated design, critique, and research work are **actually feasible, useful, and aligned with the assignment requirements** — and which are not.

Docs9 does not blindly adopt every suggestion from Docs8, Audit8 Pro, or external research. It critically evaluates each proposal against four criteria:

1. Technical feasibility
2. Expected performance gain
3. Implementation complexity
4. Compatibility with Colab/Kaggle constraints

Each proposed improvement receives one of three verdicts: **Approved**, **Deferred**, or **Rejected**.

---

## Project Timeline

### Docs7 — Original System Design

Docs7 was the initial design documentation for the tampered image detection and localization pipeline. It specified:

- CASIA v2.0 dataset with stratified 70/15/15 splits
- U-Net with ResNet34 encoder (SMP), ImageNet pretrained
- BCE + Dice combined loss
- AdamW with differential learning rates (encoder 1e-4, decoder 1e-3)
- Minimal augmentation (HFlip, VFlip, RandomRotate90)
- CONFIG-driven pipeline with W&B experiment tracking

Docs7 was a competent design document, but it contained several unvalidated assumptions about loss design, augmentation, and evaluation methodology.

### Run01 — First Experimental Evidence

Run01 (`v6-5-tampered-image-detection-localization-run-01.ipynb`) produced the first real training evidence:

| Metric | Value |
|---|---|
| Tampered-only Pixel-F1 | 0.4101 |
| Copy-move F1 | 0.3105 |
| Splicing F1 | 0.5901 |
| Best threshold | 0.1327 |
| Overfitting onset | Epoch 15 |
| JPEG robustness gap | −0.13 |

Run01 confirmed several audit predictions: missing `pos_weight` caused threshold suppression, absent scheduler caused overfitting, and minimal augmentation led to robustness collapse.

### Docs8 — Design Blueprint for v8

Docs8 synthesized findings from Docs7, Audit7 Pro, and Run01 into a v8 implementation plan. Its key contributions:

- Centered tampered-only metrics as the primary evaluation measure
- Identified five failure categories (calibration, copy-move, overfitting, robustness collapse, complete prediction failure)
- Proposed P0/P1/P2 fixes: pos_weight, scheduler, augmentation, per-sample Dice, evaluation improvements
- Documented shortcut learning risk and mitigation strategy
- Created a concrete implementation checklist for Notebook v8

### Notebook v8 — Implementation of Docs8 Plan

Notebook v8 was created as two synchronized variants (Kaggle and Colab) implementing all P0 and P1 fixes from Docs8:

- BCE `pos_weight` computed from training masks
- ReduceLROnPlateau scheduler (patience=3, factor=0.5)
- Expanded augmentation (ColorJitter, ImageCompression, GaussNoise, GaussianBlur)
- Per-sample Dice loss computation
- Tampered-only metrics reported as primary
- Expanded threshold sweep (0.05–0.80)
- Mask-size stratification in evaluation
- Gradient norm and LR logging
- cudnn.benchmark contradiction fixed

### Audit8 Pro — Critical Review of Docs8

Audit8 Pro evaluated Docs8 with a principal-level lens. Key findings:

1. Docs8 is a better self-critique than Docs7 but still documents planned fixes rather than a corrected final submission
2. Single Colab notebook deliverable remains unverified
3. CASIA is still framed too aggressively as "the expected dataset"
4. Image-level detection is still a heuristic `max(prob_map)` rule
5. Copy-move performance remains weak (F1=0.31)
6. U-Net/ResNet34 is honestly defended as a baseline but still not justified as a strong forensic architecture
7. Shortcut-learning claims are stronger than the underlying validation evidence
8. Documentation lineage cites Audit6 Pro instead of Audit7 Pro

### Docs9 — This Document Set

Docs9 serves as the **final decision document before implementing the next training pipeline**. It:

1. Responds to every Audit8 Pro criticism with a concrete decision
2. Evaluates all candidate improvements against assignment constraints
3. Integrates relevant external research selectively
4. Produces a clear implementation blueprint for Notebook v9
5. Separates what is approved from what is deferred or rejected

---

## Docs9 Document Index

| Document | Focus |
|---|---|
| [01_Assignment_Alignment_Review.md](01_Assignment_Alignment_Review.md) | Re-evaluation against original assignment requirements |
| [02_Audit8_Pro_Response.md](02_Audit8_Pro_Response.md) | Point-by-point response to Audit8 Pro findings |
| [03_Feasible_Improvements.md](03_Feasible_Improvements.md) | Technical evaluation of all candidate improvements |
| [04_Improvement_Decision_Log.md](04_Improvement_Decision_Log.md) | Decision table for every proposed technique |
| [05_External_Resource_Integration.md](05_External_Resource_Integration.md) | Selective adoption from external research |
| [06_Notebook_V9_Implementation_Plan.md](06_Notebook_V9_Implementation_Plan.md) | Detailed blueprint for the next notebook version |
| [07_Risk_Assessment.md](07_Risk_Assessment.md) | Risks and mitigation strategies |
| [08_Future_Research_Directions.md](08_Future_Research_Directions.md) | Ideas beyond the current assignment scope |

---

## Key Decisions Preview

### Approved for v9

- Learned image-level classification head (dual-task architecture)
- ELA as auxiliary input channel (RGB+ELA 4-channel input)
- Auxiliary edge loss on tamper mask boundaries
- DeepLabV3+ comparison experiment
- Finer augmentation control with ablation tracking
- Multi-seed validation (3 seeds minimum)
- Near-duplicate content leak check (pHash)
- Corrected dataset framing (CASIA as chosen baseline, not assignment mandate)

### Deferred Beyond v9

- Full multi-branch forensic architecture (EMT-Net style)
- Transformer encoders (SegFormer, TransU2-Net)
- SRM noise residual input streams
- Cross-dataset evaluation (Coverage, CoMoFoD)
- Multi-scale training / multi-scale inference

### Rejected

- Full transformer backbone replacement (too complex, uncertain benefit for Colab constraints)
- Multi-stream forensic pipeline with 3+ input branches (exceeds assignment scope and compute)
- Large-scale dataset changes (replacing CASIA entirely)

---

## Expected Outcome for Notebook v9

| Metric | Run01 | v8 Expected | v9 Expected |
|---|---|---|---|
| Tampered-only Pixel-F1 | 0.41 | 0.50–0.60 | 0.55–0.65 |
| Splicing F1 | 0.59 | 0.65–0.72 | 0.68–0.75 |
| Copy-move F1 | 0.31 | 0.40–0.50 | 0.42–0.55 |
| Optimal threshold | 0.13 | 0.30–0.50 | 0.35–0.50 |
| Robustness Δ (JPEG) | −0.13 | −0.05 to −0.08 | −0.03 to −0.06 |
| Image-level AUC | 0.87 | ~0.87 | 0.90+ |

These are estimates grounded in the specific improvements approved. The dual-task head and ELA channel are expected to provide the largest incremental gains over v8.
