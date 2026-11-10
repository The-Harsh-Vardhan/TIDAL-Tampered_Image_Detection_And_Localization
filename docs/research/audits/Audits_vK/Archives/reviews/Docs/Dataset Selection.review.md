# Review: Dataset Selection.md

Source document path: `Docs/Dataset Selection.md`

Purpose: Provide a short recommendation on which datasets to use.

Validity score: 6/10

## Assignment alignment
- Aligned at a high level.
- Less reliable than the longer dataset docs.

## Technical correctness
- CASIA and COVERAGE are sensible recommendations.
- The file has visible formatting and count corruption around the CASIA summary (lines 10-11).
- The split recommendation changes to `70 / 30` validation/testing, which conflicts with the main dataset docs (lines 76-77).

## Colab T4 feasibility
- Dataset suggestions are feasible.
- The document does not create platform complexity by itself.

## Issues found
- Moderate: Broken formatting reduces confidence in the numeric summary (lines 10-11).
- Major: The split policy conflicts with the rest of the repo (lines 76-77).
- Moderate: Class-balancing advice leans toward undersampling without justification (lines 78-82).

## Contradictions with other docs
- `Docs/04_Best_Dataset.md` and `Docs/Overall Flow Docs/02_Data_Pipeline.md` use `85 / 7.5 / 7.5`.
- `Docs/Dataset.md` and `Docs/Overall Flow.md` also flirt with subset-based balancing.

## Recommendations
- Fix the formatting and unify the split policy with the main pipeline docs.
- Prefer full-data training and handle imbalance at the loss/metric level.

## Severity summary
- Major
