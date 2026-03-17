# 02_Dataset_and_Preprocessing.md Review

## Purpose

Documents dataset choice, dynamic pair discovery, mask handling, split policy, leakage caveats, and pre-training validation checks.

## Accuracy Score

`8/10`

## What Docs4 Fixed Since Audit 3

- Keeps dynamic discovery returning `(pairs, excluded)` instead of relying on hard-coded counts.
- Preserves the split manifest and the explicit leakage caveat.
- Keeps unknown filename patterns as warnings rather than silently defaulting them away.

## Research and Notebook Alignment

- CASIA v2.0 and the splicing/copy-move framing are well aligned with the relevant survey literature.
- Notebook v4 is stronger than the doc snippet here: it includes an explicit image-mask dimension check and logs `dimension_mismatch` exclusions.

## Issues Found

- The prose says discovery validates spatial alignment, but the shown code snippet does not actually perform the dimension check that notebook v4 implements.
- CASIA still limits the project to classical copy-move and splicing categories; generated tampering remains out of scope.
- The split policy is reproducible, but source-image leakage remains unavoidable because CASIA does not expose group identifiers.

## Suggested Improvements

- Copy the notebook's dimension-check helper into this doc so the code example matches the stated behavior.
- Add one sentence that GAN-generated or deepfake-style tampering is outside the current dataset scope.
- Keep the leakage caveat prominent because it materially affects how results should be interpreted.
