# Review: 02_Data_Pipeline.md

Source document path: `Docs/Overall Flow Docs/02_Data_Pipeline.md`

Purpose: Define dataset discovery, cleaning, splitting, and loading.

Validity score: 8/10

## Assignment alignment
- Strong alignment with dataset preparation, mask alignment, and train/val/test splitting.

## Technical correctness
- The image-mask pairing, zero-mask handling for authentic images, and on-the-fly loading are solid.
- The split logic stratifies by label type only and does not guard against leakage from related images or near-duplicates (lines 175-199).
- Expected counts are written as facts before validation runs (lines 86-91, 140, 204-206).

## Colab T4 feasibility
- Feasible and practical.
- Does not add unnecessary platform complexity.

## Issues found
- Moderate: Missing group-aware split or leakage checks (lines 175-199).
- Minor: Expected dataset counts are too certain for a generic guide (lines 86-91, 204-206).

## Contradictions with other docs
- Supports the main dataset docs well.
- Conflicts with shorter docs that recommend downsampling instead of full-data training.

## Recommendations
- Keep this as a core implementation doc.
- Add a dedup/source-leakage check and explicitly keep threshold tuning on validation only.

## Severity summary
- Moderate
