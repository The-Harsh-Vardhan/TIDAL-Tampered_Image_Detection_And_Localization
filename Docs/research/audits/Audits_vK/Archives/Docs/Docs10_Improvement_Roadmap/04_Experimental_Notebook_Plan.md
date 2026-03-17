# 4. Experimental Notebook Plan

Each notebook below changes one substantive variable relative to the accepted baseline.

## `vK.10.1 Image Detection and Localisation [Evaluation-Hardened Baseline].ipynb`

- Improvement implemented: standardized evaluation only
- Changes relative to baseline:
  - add tampered-only Pixel F1
  - add tampered-only IoU
  - add image ROC-AUC
  - require fixed four-panel output panels
  - require saved notebook outputs
- Expected impact:
  - more trustworthy and assignment-complete evidence
  - possibly lower but more honest scores
- Risks:
  - no raw performance gain
  - exposes weaknesses rather than hiding them

## `vK.10.2 Image Detection and Localisation [Missing-Mask Filter].ipynb`

- Improvement implemented: dataset integrity gate
- Changes relative to baseline:
  - exclude rows where `mask_exists == 0`
  - fail fast on invalid image-mask pairs
- Expected impact:
  - cleaner supervision
  - fewer silent data issues
- Risks:
  - reduced dataset size
  - possible metric drop if invalid rows had been acting as easy cases

## `vK.10.3 Image Detection and Localisation [Leakage-Aware Split].ipynb`

- Improvement implemented: split strategy only
- Changes relative to baseline:
  - replace plain label-stratified split with grouped or hash-aware split logic
  - reduce related-image leakage across train, validation, and test
- Expected impact:
  - more trustworthy generalization estimate
- Risks:
  - raw scores may fall
  - this changes the benchmark itself

Roadmap rule:

- If `10.3` changes the accepted split policy, all later performance notebooks must be rerun on that accepted split before merge decisions are made.

## `vK.10.4 Image Detection and Localisation [Localization-Aware Checkpointing].ipynb`

- Improvement implemented: checkpoint criterion only
- Changes relative to baseline:
  - save the best checkpoint by validation tampered-only Pixel F1 instead of validation accuracy
- Expected impact:
  - better final masks without changing architecture
- Risks:
  - slight drop in image-level accuracy

## `vK.10.5 Image Detection and Localisation [Boundary-Preserving Augmentations].ipynb`

- Improvement implemented: augmentation policy only
- Changes relative to baseline:
  - remove `GaussNoise`
  - remove `JpegCompression`
  - keep resize, flip, brightness/contrast, and shift-scale-rotate
- Expected impact:
  - sharper boundaries
  - less corruption of subtle forensic traces
- Risks:
  - weaker robustness to compression or noise distortions

## `vK.10.6 Image Detection and Localisation [Pretrained Encoder].ipynb`

- Improvement implemented: encoder initialization only
- Changes relative to baseline:
  - replace the scratch encoder with an ImageNet-pretrained CNN encoder
  - preserve dual-task outputs
- Expected impact:
  - stronger features
  - faster convergence
- Risks:
  - implementation complexity
  - possible mismatch with the current decoder or classifier head

## `vK.10.7 Image Detection and Localisation [Seeded Reproducible Run].ipynb`

- Improvement implemented: reproducibility only
- Changes relative to baseline:
  - add global seeds
  - add deterministic flags where practical
  - save run metadata
- Expected impact:
  - lower run-to-run variance
  - better auditability
- Risks:
  - possible runtime slowdown
  - limited direct score gain

## `vK.10.8 Image Detection and Localisation [AMP Training].ipynb`

- Improvement implemented: mixed precision only
- Changes relative to baseline:
  - enable AMP while keeping model, data, and losses fixed
- Expected impact:
  - faster training
  - lower memory use
  - easier repeated experiments
- Risks:
  - no guaranteed metric gain
  - possible numerical edge cases
