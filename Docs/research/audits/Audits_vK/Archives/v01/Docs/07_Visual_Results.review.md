# 07_Visual_Results.md Review

## Purpose

Specifies the qualitative prediction displays and supporting training and evaluation plots for the notebook.

## Technical Accuracy Score

`7/10`

## What Is Correct

- Technical correctness: The document correctly emphasizes original image, ground-truth mask, predicted output, and overlay-style visualization as central deliverables.
- Implementability on a single Colab notebook with T4: The plotting plan is practical for notebook use and does not add meaningful runtime cost.
- Assignment alignment: The focus on honest sample selection, training curves, and qualitative failure analysis is strong.

## Issues Found

- Contradictions: The document says the main 4-column grid is the primary visual output, but the later display order splits results into separate best, average, failure, and authentic grids. That is a presentational drift rather than a fatal issue, but it weakens the spec.
- Unsupported or hallucinated claims: None of the core plotting ideas are hallucinated, but the examples assume helper functions and imports that are not fully declared in the snippet.
- Unnecessary complexity: The extra ROC and threshold plots are reasonable, but they should remain secondary to the assignment-mandated mask outputs.
- Missing technical details: The main grid uses a predicted heatmap as the third panel instead of a binary predicted mask, even though the assignment explicitly asks for a predicted output mask.
- Additional implementation risk: The sample plotting code calls `.numpy()` on tensors without clearly ensuring they are on CPU, and it uses `torch.tensor(...)` without importing `torch` in the shown snippet.

## Recommendations

- Make binary predicted mask a required panel and keep the heatmap optional or supplementary.
- Decide whether the deliverable uses one combined grid or several curated grids, then state that choice once.
- Tighten the code snippet so it is runnable as shown, including CPU conversion and required imports.
