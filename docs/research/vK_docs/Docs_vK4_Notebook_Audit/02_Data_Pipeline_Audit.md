# 02 - Data Pipeline Audit

## Verdict
**Conditionally acceptable but not fully reliable.** The structure is reasonable, but critical safeguards are missing.

## What Works

1. Dataset discovery recursively searches under Kaggle input for IMAGE and MASK directories.
2. Train/val/test split uses stratified sampling by label.
3. Mask binarization uses mask > 0, which is standard for binary segmentation.

## Critical Weaknesses

## A) Image-mask pairing is permissive, not strict

- Missing mask files are tolerated and set to None.
- Dataset loader creates an all-zero mask when mask_path is missing.

### Why this is bad
If a tampered sample loses its mask due to naming mismatch or missing file, the model receives contradictory supervision and learns garbage boundaries.

### What senior engineers expect
Hard fail for tampered samples missing masks. No silent fallback.

## B) Validation helper exists but integrity enforcement is weak

- Image validation helper is defined.
- It is not used to gate the dataset construction pipeline.

### Why this is bad
Corrupted assets can slip in and fail late.

### Expected
Validate and exclude invalid files with explicit counts and audit log.

## C) Split leakage defense is not explicit

- Stratified split is present.
- There is no post-split assertion proving path-level disjointness.

### Why this is bad
Leakage can happen through accidental duplication or path collisions and inflate validation metrics.

### Expected
Explicit overlap assertions:
- train ∩ val = empty
- train ∩ test = empty
- val ∩ test = empty

## D) Resize policy may blur forgery boundaries

- All samples are resized to 256 x 256.

### Why this is bad
Tiny manipulation regions can be destroyed or softened, artificially lowering boundary quality and skewing model behavior.

### Expected
At least one check on resolution sensitivity or a boundary-aware resizing rationale.

## Reliability Score

- Structural design: 7/10
- Integrity enforcement: 4/10
- Leakage confidence: 5/10
- Overall data pipeline reliability: **5.5/10**

## Bottom Line

The data pipeline is good enough to run experiments, but not strong enough for a high-confidence submission without strict integrity assertions and failure rules.
