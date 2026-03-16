# Review: 06_Best_Practices.md

Source document path: `Docs/06_Best_Practices.md`

Purpose: Define engineering, training, and evaluation standards for the project.

Validity score: 6/10

## Assignment alignment
- Good focus on reproducibility, data validation, checkpointing, and reporting.
- Broadly useful for the assignment.

## Technical correctness
- Most engineering guidance is reasonable.
- The AMP examples use the older `torch.cuda.amp` style (lines 106-120), while other docs use `torch.amp`.
- AUC-ROC is labeled as a pixel metric here (lines 186-192), which conflicts with later image-level AUC guidance.

## Colab T4 feasibility
- The baseline practices are feasible.
- None of the recommended standards require leaving the Colab notebook path.

## Issues found
- Moderate: Metric definitions drift across the repo, especially around AUC (lines 186-192).
- Moderate: The AMP guidance should be updated to match the newer API style already used elsewhere (lines 106-120).
- Minor: "Always use AMP" is directionally right on T4 but written more absolutely than necessary.

## Contradictions with other docs
- `Docs/Overall Flow Docs/06_Performance_Metrics.md` defines AUC at the image level.
- `Docs/Code-Generation-Instructions.md` suggests best-model selection by IoU or Dice rather than validation F1.

## Recommendations
- Unify the metric taxonomy once across the repo.
- Keep the reproducibility and checkpointing guidance.
- Update code snippets to the current AMP style.

## Severity summary
- Moderate
