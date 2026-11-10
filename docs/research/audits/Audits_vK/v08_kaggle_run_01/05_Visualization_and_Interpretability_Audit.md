# Visualization And Interpretability Audit

## First, correct the false premise

There are visualizations inside Run-01. The prompt's "no image visualizations inside the notebook itself" claim is false for the executed artifact.

Run-01 includes:

- cell 18: sanity-check training batch
- cells 36 and 37: training curves and threshold curve
- cell 39: prediction grid with original image, ground truth, predicted mask, and overlay
- cell 43: Grad-CAM plus diagnostic TP/FP/FN overlays
- cell 44: failure-case analysis
- cell 55: saved plot inventory confirming `prediction_grid.png` and `gradcam_analysis.png`

So the problem is not "there are no visuals." The problem is "the visuals are still not doing enough work."

## What the notebook does right

Cell 39 is aligned with the assignment's visual-results requirement better than most internship notebooks. It explicitly builds:

- original image
- ground-truth mask
- predicted mask
- overlay

That is the correct basic visual stack.

Cell 43 also adds a diagnostic overlay:

- green = true positive
- red = false positive
- blue = false negative

That is useful because it shows error type, not just mask shape.

## Why the qualitative section is still not good enough

The notebook has visuals, but it still curates the story too gently.

Problems:

1. The prediction grid in cell 39 samples `best`, `median`, `worst`, and a couple of authentic cases. That is neat for a blog post, not sufficient for a forensic audit.
2. It does not explicitly separate copy-move failures from splicing successes, even though cell 33 shows a huge gap between them.
3. It does not explicitly surface tiny-mask failures, even though cell 33 shows that masks under 2 percent are a disaster.
4. It does not show false positives on authentic images as a dedicated failure category.
5. It does not show probability maps, only hard-thresholded masks. That hides calibration behavior.

When the model collapses on copy-move and small regions, the qualitative section should make that pain impossible to miss.

## Comparison with the reference notebook

The reference notebook `Pre-exisiting Notebooks/image-detection-with-mask.ipynb` is crude, but it gets one basic presentation habit right: it makes sample collection and inline image displays explicit and immediate.

Relevant pieces:

- cell 7 collects real and fake samples from the loader
- cell 8 shows overlayed predictions on sample images
- cell 9 and cell 10 show image plus predicted mask
- cell 12 and cell 13 expand the visualization grid to larger sample sets

That reference notebook is not perfect either. It still does not show ground-truth masks in the visualization flow, so it is not a gold standard. But it does not hide model behavior behind a sparse summary.

Run-01 is better in structure because it includes GT masks and diagnostic overlays. It is worse in one important sense: it does not turn the real failure modes into a disciplined visual review.

## Grad-CAM is fine, but the priority is backwards

Cells 42 and 43 add Grad-CAM on encoder layer 4. Fine.

But here is the blunt truth: when copy-move F1 is `0.1394`, tiny-mask F1 is `0.1432`, and worst-case F1 averages `0.0000`, adding Grad-CAM before fully exposing those failures is misplaced effort.

Senior expectation:

- first show clear segmentation successes and failures
- then show why the failures happen
- only then add attention heatmaps if they clarify something non-obvious

Right now Grad-CAM feels like garnish.

## What should be added before final submission

1. Separate visualization panels for splicing, copy-move, and authentic false positives.
2. A dedicated "tiny-mask failures" panel.
3. Probability heatmaps next to hard-thresholded masks.
4. A threshold comparison view, for example `0.5` versus `0.75`, on the same samples.
5. At least a few authentic images with false alarms highlighted.
6. A concise table next to the figure tying sample IDs to F1, IoU, mask area, and forgery type.

## Verdict

The notebook meets the assignment's minimum requirement to include visual results. It does not yet meet the standard of a serious segmentation review, because the visuals are present but not ruthless enough about exposing where the model actually fails.
