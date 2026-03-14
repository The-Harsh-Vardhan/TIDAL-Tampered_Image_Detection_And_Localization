# 02_Dataset_and_Preprocessing.md Review

## Purpose

Specifies CASIA v2.0 usage, image-mask pairing, cleaning logic, split policy, and preprocessing steps before training.

## Technical Accuracy Score

`7/10`

## What Is Correct

- Technical correctness: The document makes the right baseline calls for this assignment: use CASIA, generate zero masks for authentic images, binarize masks, and keep nearest-neighbor interpolation for masks.
- Implementability on a single Colab notebook with T4: The proposed preprocessing path is straightforward to implement and operationally light enough for Colab.
- Assignment alignment: It directly covers the required dataset preparation tasks, including alignment checks and train/val/test splitting.

## Issues Found

- Contradictions: The split policy is consistent internally, but the surrounding docs do not define any leakage-prevention rule beyond stratification.
- Unsupported or hallucinated claims: The exact dataset counts and the hard-coded claim of "17 misaligned pairs" are plausible but unverified for the specific Kaggle mirror. These numbers should not be treated as guaranteed constants.
- Unnecessary complexity: None of the main preprocessing steps are over-engineered; the main issue is overconfidence rather than excess machinery.
- Missing technical details: There is no group-aware split policy, no near-duplicate handling, no corrupted-file handling, and no output audit of how many pairs were skipped or excluded.
- Additional implementation risk: The document says to use `train_test_split` with stratification for an 85 / 7.5 / 7.5 split, but it does not spell out the two-stage split procedure needed to achieve that ratio reliably.

## Recommendations

- Replace hard-coded mismatch counts with a dynamic validation step that logs all excluded pairs.
- Add a split-integrity note covering related images, filename-group heuristics, or at least a documented limitation if true grouping is unavailable.
- Document the exact two-step split process and persist the split manifest so results are reproducible.
