# Evaluation, Validation, and Explainability Teardown

This is the part most likely to impress a casual reviewer and annoy a serious one. The plots look complete. The methodology is not nearly as clean as the project wants it to look.

## 1. The image-level scoring rule is not stable across the repo

`Docs7/03_Model_Architecture.md:88-101` says image-level detection uses top-k mean.

`Docs7/05_Evaluation_Methodology.md:25` repeats top-k mean.

`Docs7/00_Master_Report.md:58` lists "top-k mean" as a known limitation.

The actual v6.5 notebook does this:

```python
tamper_score = probs[i].view(-1).max().item()
image_preds.append(int(tamper_score >= threshold))
```

Evidence: `notebooks/tamper_detection_v6.5_kaggle.ipynb:1236-1239`.

That is a material behavior difference, not documentation trivia.

### Why this matters

`max(prob_map)` is much more sensitive to isolated hot pixels than top-k mean. If the project cannot even keep that story consistent, every claim about image-level robustness and threshold behavior becomes suspect.

## 2. Mixed-set metrics are flattering the model

The empty-mask handling logic is friendly to the point of distortion.

### Implemented behavior

From `notebooks/tamper_detection_v6.5_kaggle.ipynb:856-891`:

- if GT is empty and prediction is empty:
  - F1 = 1.0
  - IoU = 1.0
  - precision = 1.0
  - recall = 1.0
- if GT is empty and prediction is positive:
  - precision = 0.0
  - recall = 1.0

### Why this is a problem

1. Authentic true negatives become perfect localization examples in mixed-set averages.
2. Empty-GT false positives still get recall `1.0`, which is semantically ridiculous if the reader treats recall as "did you avoid missing tampered content?"
3. The result is a reporting setup that can make mixed-set localization metrics look stronger than the actual tampered-region performance.

The docs explain the true-negative convention (`Docs7/05_Evaluation_Methodology.md:52-61`). Fine. They do not honestly confront how much that convention can flatter mixed-set results.

## 3. One threshold is doing too much work

The project uses a validation sweep after training to find the best segmentation threshold (`notebooks/tamper_detection_v6.5_kaggle.ipynb:1168-1198`).

Then that same threshold is reused for:

- test mask binarization
- image-level detection
- robustness evaluation

Docs7 says the same (`Docs7/05_Evaluation_Methodology.md:43-48`).

### Why this is weak

Segmentation and image-level detection are different decision problems. The threshold that maximizes validation Pixel-F1 is not automatically the threshold that gives the best image-level operating point.

The project picks convenience over calibration and then acts like that was methodological rigor.

## 4. The docs contradict themselves on training-time validation

`Docs7/12_Complete_Notebook_Structure.md:68` says:

"Threshold-aware validation: Early stopping uses best val Pixel-F1 from the sweep, not a fixed 0.5 threshold."

That is false for v6.5.

Actual behavior:

- `validate_model(..., threshold=0.5)` is defined with a fixed default (`notebooks/tamper_detection_v6.5_kaggle.ipynb:966-1008`)
- the training loop calls `validate_model(model, val_loader, criterion, device, CONFIG)` without passing a different threshold (`notebooks/tamper_detection_v6.5_kaggle.ipynb:1072-1075`)
- `Docs7/04_Training_Strategy.md:186` correctly says training uses fixed `0.5` and the sweep happens after training

So the structure doc is wrong, the training doc is right, and the notebook agrees with the training doc. That is exactly the kind of contradiction that makes reviewers stop trusting everything else.

## 5. Validation experiments are mostly hypothetical

`Docs7/13_Validation_Experiments.md` is smart on paper:

- mask randomization
- shortcut indicator checks
- prediction solidity
- boundary-band concentration

But the doc explicitly says these are not part of the standard training pipeline and require separate execution (`Docs7/13_Validation_Experiments.md:139-147`).

That means:

1. They are not evidence unless someone actually ran them.
2. The repo does not currently prove the model avoided shortcut learning.
3. The project is getting rhetorical benefit from validation work it has not clearly earned.

## 6. Robustness testing is narrow and misnamed

`Docs7/06_Robustness_Testing.md:29-40` defines eight conditions:

- clean
- JPEG 70
- JPEG 50
- light noise
- heavy noise
- blur
- resize 0.75x
- resize 0.5x

The notebook implements those (`notebooks/tamper_detection_v6.5_kaggle.ipynb:1845-1885`).

That is post-processing corruption robustness. It is not robustness to new forgery families.

So if the author says "the model is robust," the correct follow-up is:

"Robust to what, exactly?"

Certainly not to diffusion edits, content-aware fill, semantic inpainting, relighting harmonization, or GAN-based composites.

## 7. Per-image averaging hides useful failure structure

`Docs7/05_Evaluation_Methodology.md:16` and `120-124` prefer per-image averaging.

That choice is defensible, but incomplete. Without micro/global pixel metrics, the project cannot show whether failures are concentrated in small masks, large masks, or a handful of catastrophic misses.

A mature evaluation stack would report both macro and micro views, plus size-stratified analysis.

## 8. Explainability is handled cautiously in wording but weakly in implementation

The docs are technically careful:

- Grad-CAM is called diagnostic, not causal proof.

That is good.

The actual implementation is still weak as evidence:

1. Target scalar is `output.mean()` (`notebooks/tamper_detection_v6.5_kaggle.ipynb:1562-1565`)
2. Visualization samples are the highest-F1 tampered examples (`notebooks/tamper_detection_v6.5_kaggle.ipynb:1625-1629`)
3. There is no quantitative attribution evaluation

That produces nice-looking heatmaps with very little hard epistemic value.

## 9. Shortcut-learning risk remains unresolved

The repo itself lists plausible shortcut channels in `Docs7/13_Validation_Experiments.md:45-50` and `58-64`:

- JPEG quality mismatch
- resolution differences
- color-statistics shifts
- source-image overlap

Good. The problem is that the notebook does not execute a real falsification test for these risks.

So the honest state is:

"Shortcut risk is acknowledged, not disproven."

That is very different from claiming the model learned genuine forensic reasoning.

## Bottom line

The evaluation stack is good enough for an internship baseline demo. It is not good enough to support broad claims about reliable tamper detection.

The biggest problems are:

1. doc-code mismatch on core scoring behavior
2. mixed-set metric inflation risk
3. unexecuted validation experiments
4. narrow robustness scope
5. weak image-level calibration

## Immediate fixes

1. Standardize one image-level score and document it truthfully.
2. Make tampered-only localization the headline result.
3. Mark empty-GT recall as not applicable instead of 1.0 for false positives.
4. Report both macro and micro metrics.
5. Actually run at least one shortcut-learning falsification experiment.
6. Show false-positive-heavy authentic examples and hard failures in the explainability section.
