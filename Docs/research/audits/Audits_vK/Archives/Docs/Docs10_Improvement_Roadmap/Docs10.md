# Docs10

Source artifacts used for this roadmap:

- `Assignment.md`
- `Audits/Audit v7.1/Audit-vK.7.1.md`
- `Notebooks/vK.7.1 Image Detection and Localisation.ipynb`

This document defines a systematic improvement roadmap for the current project baseline, `vK.7.1 Image Detection and Localisation.ipynb`. The purpose is to improve the system through controlled, one-change-at-a-time experiments while preserving assignment compliance and avoiding uncontrolled notebook drift.

## 1. Overview of Improvement Strategy

The improvement philosophy is ablation-first.

- Each experimental notebook must change only one substantive variable relative to the accepted baseline.
- Every experiment must be compared against the same baseline control run unless the experiment itself changes the split policy.
- Only changes that produce measurable benefit, or clearly improve evaluation validity or reproducibility without harming performance, should be merged.
- Every experiment must preserve assignment compliance: authentic images, tampered images, ground-truth masks, image-level detection, pixel-level localization, and the required visual comparisons must remain present.

Operational rule:

- Measurement and artifact completeness are mandatory for every run. An experiment is not valid unless the saved notebook preserves its metric outputs and final visual outputs inside the `.ipynb` artifact.

The development flow is:

1. Execute `vK.7.1` unchanged once to establish the baseline control run.
2. Create one 10.x notebook per independent improvement.
3. Compare each notebook against the baseline control using the frozen evaluation protocol in Section 5.
4. Merge only successful changes.
5. Test combinations of successful changes later in a dedicated merge notebook rather than combining them immediately.

## 2. Baseline Definition

The baseline code configuration is `vK.7.1 Image Detection and Localisation.ipynb`.

### Dataset

- CASIA-style authentic (`Au`) and tampered (`Tp`) images with binary masks.
- Kaggle-first dataset root: `casia-spicing-detection-localization`.
- Metadata is created by pairing filenames across `IMAGE/<class>` and `MASK/<class>`.
- Data split is `70/15/15` using stratified `train_test_split`.

### Model

- Custom U-Net-style encoder-decoder.
- Encoder/decoder channel progression: `64 -> 128 -> 256 -> 512 -> 1024`.
- One-channel segmentation head.
- Classification head:
  - `AdaptiveAvgPool2d`
  - `Flatten`
  - `Linear(1024, 512)`
  - `ReLU`
  - `Dropout(0.5)`
  - `Linear(512, 2)`

### Training

- Image size: `256`
- Batch size: `8`
- Workers: `2`
- Image-level loss: focal-style classification loss with balanced class weights
- Segmentation loss: `0.5 * BCEWithLogits + 0.5 * Dice`
- `ALPHA = 1.5`
- `BETA = 1.0`
- Optimizer: `Adam(lr=1e-4)`
- Scheduler: `CosineAnnealingLR(T_max=10)`
- Epochs: `50`
- Gradient clipping: `max_norm=5.0`
- Best checkpoint rule: highest validation accuracy

### Current Evaluation State

- Implemented metrics in baseline code: accuracy, Dice, IoU, F1
- Missing from the saved baseline artifact: ROC-AUC and executed final four-panel proof
- Current localization metrics are not tampered-only
- Current best-model selection is based on detection accuracy rather than localization quality

### Baseline Prerequisite

Before any 10.x notebook is judged, `vK.7.1` must be run unchanged once and saved with outputs. That baseline control run becomes the comparison anchor for all later experiments.

## 3. Improvement Candidates From Audit

The audit findings produce three categories of improvement candidates.

### Highest Expected Impact

- Localization-aware checkpoint selection
- Pretrained encoder within the same dual-task U-Net pattern
- Leakage-aware split strategy
- Boundary-preserving augmentation policy

These are the most likely to affect final localization quality, generalization, or both.

### Medium Expected Impact

- Invalid-mask filtering and dataset integrity gate
- Tampered-only localization reporting plus ROC-AUC
- Global seeding and deterministic run manifest

These improve data quality, evaluation validity, and experiment reliability. Some may not increase raw scores, but they make the results more trustworthy.

### Lower Metric Impact but High Engineering Value

- AMP / mixed precision
- Output-complete execution discipline
- Hardcoded-path cleanup and notebook simplification

These primarily improve runtime efficiency, artifact quality, and maintainability.

Important interpretation:

- Some candidates target performance.
- Some target evaluation validity.
- Some target reproducibility or runtime efficiency.
- All should be documented.
- Only some should become permanent model changes.

## 4. Experimental Notebook Plan

Each notebook below changes one substantive variable relative to the accepted baseline.

### `vK.10.1 Image Detection and Localisation [Evaluation-Hardened Baseline].ipynb`

- Improvement implemented: standardized evaluation only
- Changes relative to baseline:
  - add tampered-only Pixel F1
  - add tampered-only IoU
  - add image ROC-AUC
  - require fixed four-panel output panels
  - require saved notebook outputs
- Expected impact:
  - more trustworthy and assignment-complete evidence
  - may lower reported scores by removing inflated or incomplete reporting
- Risks:
  - no direct performance gain
  - exposes weaknesses that were previously hidden by weak reporting

### `vK.10.2 Image Detection and Localisation [Missing-Mask Filter].ipynb`

- Improvement implemented: dataset integrity gate
- Changes relative to baseline:
  - exclude rows where `mask_exists == 0`
  - fail fast on invalid image-mask pairs before split and training
- Expected impact:
  - cleaner supervision
  - fewer silent dataset failures
- Risks:
  - reduced dataset size
  - possible score drop if invalid rows had been functioning as easy examples

### `vK.10.3 Image Detection and Localisation [Leakage-Aware Split].ipynb`

- Improvement implemented: split strategy only
- Changes relative to baseline:
  - replace plain label-stratified split with grouped or hash-aware split logic
  - reduce related-image leakage across train, validation, and test
- Expected impact:
  - more honest generalization estimate
  - better alignment with rigorous experimental design
- Risks:
  - raw scores may fall
  - this changes the benchmark, so later comparisons must be rerun on the accepted split

Roadmap rule:

- If `10.3` becomes the accepted split policy, all later performance notebooks must be rerun on that accepted split before merge decisions are made.

### `vK.10.4 Image Detection and Localisation [Localization-Aware Checkpointing].ipynb`

- Improvement implemented: checkpoint criterion only
- Changes relative to baseline:
  - select best checkpoint by validation tampered-only Pixel F1 instead of validation accuracy
- Expected impact:
  - stronger final masks without changing architecture or losses
- Risks:
  - slight drop in image-level accuracy

### `vK.10.5 Image Detection and Localisation [Boundary-Preserving Augmentations].ipynb`

- Improvement implemented: augmentation policy only
- Changes relative to baseline:
  - remove `GaussNoise`
  - remove `JpegCompression`
  - keep resize, horizontal flip, brightness/contrast, and shift-scale-rotate
- Expected impact:
  - sharper localization boundaries
  - less corruption of subtle forensic traces
- Risks:
  - weaker robustness to post-processing distortions

### `vK.10.6 Image Detection and Localisation [Pretrained Encoder].ipynb`

- Improvement implemented: encoder initialization only
- Changes relative to baseline:
  - replace the scratch encoder with an ImageNet-pretrained CNN encoder
  - preserve dual-task outputs and the same overall prediction tasks
- Expected impact:
  - stronger low-level and mid-level feature extraction
  - faster convergence
- Risks:
  - implementation mismatch with the current decoder or classifier head
  - added complexity may offset the benefit if integration is sloppy

### `vK.10.7 Image Detection and Localisation [Seeded Reproducible Run].ipynb`

- Improvement implemented: reproducibility only
- Changes relative to baseline:
  - add global seeds
  - add deterministic flags where practical
  - save run metadata needed to reproduce the result
- Expected impact:
  - lower run-to-run variance
  - better auditability
- Risks:
  - possible runtime slowdown
  - little or no direct metric gain

### `vK.10.8 Image Detection and Localisation [AMP Training].ipynb`

- Improvement implemented: mixed precision only
- Changes relative to baseline:
  - enable AMP while keeping model, data, and losses fixed
- Expected impact:
  - lower memory use
  - faster training
  - more practical repeated experiments
- Risks:
  - no guaranteed score gain
  - potential numerical edge cases

## 5. Evaluation Protocol

All 10.x notebooks must use the same frozen comparison protocol unless the experiment itself changes the split policy.

### Segmentation Metrics

- `Pixel F1` on tampered images only
- `IoU` on tampered images only
- Optional auxiliary metric: overall Dice

### Detection Metrics

- Image-level accuracy
- ROC-AUC

### Visual Outputs

Every experiment notebook must show:

- Original image
- Ground truth mask
- Predicted mask
- Overlay visualization

### Frozen Constants

These must remain fixed across experiments unless the single experimental variable is the split policy:

- same dataset source
- same image resolution
- same training budget
- same test set
- same mask threshold `0.5`

### Qualitative Panel Rule

Use the same saved sample IDs from the baseline control run across all experiments so qualitative comparisons remain fair and directly comparable.

### Artifact Completeness Rule

A notebook run is invalid unless:

- metrics are printed and saved in the notebook output
- final qualitative panels are rendered and saved in the notebook output
- the `.ipynb` artifact is preserved with outputs

## 6. Experiment Tracking Table

Use the following table for experiment tracking.

| Experiment | Single Change | Val Pixel F1 (Tampered-Only) | Val IoU (Tampered-Only) | Test Pixel F1 (Tampered-Only) | Test IoU (Tampered-Only) | Image Accuracy | ROC-AUC | Runtime / GPU Notes | Merge Decision | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| Baseline Control (vK.7.1 executed) | none |  |  |  |  |  |  |  |  |  |
| 10.1 | Evaluation-hardened baseline |  |  |  |  |  |  |  |  |  |
| 10.2 | Missing-mask filter |  |  |  |  |  |  |  |  |  |
| 10.3 | Leakage-aware split |  |  |  |  |  |  |  |  |  |
| 10.4 | Localization-aware checkpointing |  |  |  |  |  |  |  |  |  |
| 10.5 | Boundary-preserving augmentations |  |  |  |  |  |  |  |  |  |
| 10.6 | Pretrained encoder |  |  |  |  |  |  |  |  |  |
| 10.7 | Seeded reproducible run |  |  |  |  |  |  |  |  |  |
| 10.8 | AMP training |  |  |  |  |  |  |  |  |  |

## 7. Merge Strategy

Merge decisions must follow explicit rules.

### Rule 1: Performance-Oriented Changes

Performance changes should merge only if:

- at least one primary localization metric improves by at least `+0.01` absolute on validation
- the same primary metric also improves by at least `+0.01` absolute on test
- image accuracy does not drop by more than `0.01` absolute
- ROC-AUC does not drop by more than `0.01` absolute

If validation improves but test does not, the change does not merge.

### Rule 2: Engineering or Evaluation Changes

Engineering or evaluation changes can merge if they measurably improve compliance, evaluation validity, reproducibility, or runtime efficiency and do not cause more than `0.005` absolute degradation in any primary metric.

### Rule 3: Split-Policy Changes

If the leakage-aware split from `10.3` is accepted:

- rerun the baseline control on the new split
- rerun all later performance experiments on that same accepted split
- compare only within the new split regime

### Rule 4: Interaction Effects

If two winning changes appear complementary, do not merge them directly into the main notebook first. Create a later combined notebook, for example `vK.10.M1`, and test the interaction explicitly.

### Rule 5: Promotion to Mainline

The next main implementation should be created only after the merged candidate:

- passes the frozen evaluation protocol
- preserves all required visual outputs
- remains assignment-aligned
- retains a complete saved notebook artifact

## 8. Assignment Compliance Check

Every 10.x notebook must pass this checklist before it is considered valid.

- authentic images still present
- tampered images still present
- ground-truth masks still present
- image-level detection still active
- pixel-level localization still active
- required four-panel visual comparison still present
- localization metrics still reported
- notebook remains a single runnable submission artifact
- any dataset filtering or split change keeps the dataset publicly valid and documented

## Final Roadmap Position

The project should not jump directly to a redesigned model. The correct next move is to harden the baseline, make the evaluation honest, and then test targeted improvements one at a time. This roadmap keeps the work aligned with the assignment while building toward a stronger, evidence-backed successor to `vK.7.1`.
