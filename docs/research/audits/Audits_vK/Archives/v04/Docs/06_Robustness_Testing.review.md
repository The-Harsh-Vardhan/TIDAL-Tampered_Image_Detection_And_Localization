# 06_Robustness_Testing.md Review

## Purpose

Defines the Phase 3 robustness protocol, the degradation suite, the resize wrapper, and the reporting format for bonus robustness results.

## Accuracy Score

`8/10`

## What Docs4 Fixed Since Audit 3

- Keeps the corrected resize-only-on-images design through `ResizeDegradationDataset`.
- Softens the research-context claims so they are no longer presented as if directly cited facts from the final doc set.
- Adds an explicit note that robustness evaluation is segmentation-focused and does not claim full image-level robustness analysis.

## Research and Notebook Alignment

- Notebook v4 implements the documented JPEG, noise, blur, and resize degradations, including the custom resize wrapper.
- The research papers broadly support testing post-processing degradations, but Docs4 still uses generic research-context language rather than tying the suite to specific repository papers.

## Issues Found

- The document does not include cropping robustness, even though cropping appears in the assignment examples.
- Image-level robustness behavior is intentionally left out, which is acceptable for a bonus section but still a limitation.
- The research context is directionally correct but remains generic rather than paper-specific.

## Suggested Improvements

- If bonus scope allows, add cropping as an optional extra degradation to match the assignment examples more closely.
- Keep the current segmentation-focused reporting, but state clearly that image-level robustness is not evaluated.
- Add one short citation map from the strongest repository papers to the robustness rationale.
