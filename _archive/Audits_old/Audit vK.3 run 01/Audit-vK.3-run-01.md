# Audit-vK.3-run-01

Review target: `Notebooks/vk-3-image-detection-and-localisation-run-01.ipynb`

Audit basis:

- assignment requirements from `Assignment.md`
- notebook artifact evidence
- repo-level asset evidence where relevant

## Executive Summary

This notebook is materially stronger as a submission artifact than the later `vK.7.x` family because it is actually executed, shows final metrics, prints the model, includes training curves, and renders the required four-panel qualitative outputs. At the artifact level, it does demonstrate a working dual-task system for image-level tamper detection and pixel-level localization.

That said, it is not clean. The single biggest technical flaw is that the notebook still contains a source-preserved prior experiment block with broken CSV wiring, where `TRAIN_CSV` points to `test_metadata.csv` and `TEST_CSV` points to `val_metadata.csv`. That means part of the notebook contains an invalid benchmark path, even though a later "effective submission" block corrects the split and produces the real final metrics. The second major compliance problem is deliverables: the repo contains trained weights and helper scripts, but no explicit Google Colab notebook link was found. So this is a credible baseline artifact, but not a fully compliant, reviewer-tight submission.

## Objective Compliance

The core objective is met.

- Image-level tampering detection is implemented and demonstrated through the classifier head and final test accuracy output.
- Pixel-level localization is implemented and demonstrated through the segmentation head, predicted masks, overlays, and the final four-panel prediction grid.

Evidence:

- model architecture printout is present and executed in cell `23`
- final effective test metrics are executed and saved
- final qualitative four-panel outputs are executed in cell `36`

Verdict: `PASS`

## Dataset Compliance

Dataset choice is compliant in principle. The notebook uses a CASIA-style authentic/tampered dataset with paired masks and explicitly documents authentic images, tampered images, binary masks, and a `70 / 15 / 15` split workflow.

Evidence:

- dataset explanation and split description are present in cells `7-9`
- metadata creation logs show `12614` images with authentic/tampered counts
- missing-mask preview is printed

Weaknesses:

- the metadata builder records `mask_exists` but does not clearly enforce a hard filter before the preserved old experiment path
- mask pairing is filename-based and assumed correct from folder symmetry rather than being robustly validated

Verdict: `PARTIAL`

## Data Pipeline Review

The pipeline does the right broad steps: metadata generation, train/validation/test splitting, resizing, normalization, and augmentation. The effective block uses the correct split files and reports `Train: 8829 Val: 1892 Test: 1893`.

The problem is not absence of a pipeline. The problem is that the notebook contains two different pipeline paths, and one of them is wrong.

Critical issue:

- preserved prior block:
  - `TRAIN_CSV = "/kaggle/working/test_metadata.csv"`
  - `VAL_CSV = "/kaggle/working/val_metadata.csv"`
  - `TEST_CSV = "/kaggle/working/val_metadata.csv"`
  - logged sizes: `Train: 1893 Val: 1892 Test: 1892`

That makes the old path unsuitable as a benchmark and undermines confidence in the notebook's internal consistency.

Augmentation is present and reasonable as a baseline:

- resize
- horizontal flip
- brightness/contrast
- Gaussian noise
- JPEG compression
- affine transform

But the notebook does not turn those into a real robustness study.

Verdict: `PARTIAL`

## Model Architecture Review

The model is a valid assignment-level baseline: a U-Net-style encoder-decoder with a classifier head attached to the bottleneck. That is a correct multitask design for "detect + localize."

Strengths:

- supports pixel-level segmentation directly
- supports image-level detection through the classifier branch
- architecture is fully printed and documented

Weaknesses:

- this is a generic CNN baseline, not a forensic-specific architecture
- no pretrained encoder
- no boundary-aware localization design
- checkpoint selection in the effective block is still based on validation accuracy, not localization quality

Verdict: `PASS`

## Resource Constraints / Colab Compatibility

This notebook is probably runnable on a Google Colab T4-class GPU. It uses `256 x 256` inputs, batch size `8`, and a standard PyTorch training loop. Nothing in the effective path looks obviously too large for Colab-class hardware.

But strict deliverable compliance is weaker than the notebook claims.

Evidence:

- the notebook describes itself as a Google Colab notebook
- the saved output explicitly shows `IN_COLAB: False`
- no explicit Colab link was found in repo search

So this is best described as "Colab-feasible" rather than "fully delivered as a Colab submission."

Verdict: `PARTIAL`

## Evaluation Metrics Review

The notebook does provide quantitative evaluation, and unlike later versions, it actually saves the final effective results in the artifact.

Effective final metrics:

- `Final Test Acc: 0.8986`
- `Final Test Dice: 0.5761`
- `Test IoU: 0.5526`
- `Test F1: 0.5761`

That satisfies the minimum requirement to evaluate both detection and localization.

The weak points are:

- no ROC-AUC, despite the assignment asking for standard industry metrics
- no confusion matrix, precision, or recall for image-level detection
- early preserved-run localization metrics are suspiciously flat around `0.5949`
- localization quality in the effective run is only moderate, not strong

This is enough to count as evaluated, but not enough to call rigorous.

Verdict: `PARTIAL`

## Visual Results Review

This is one of the notebook's strongest sections. The saved artifact includes:

- original image
- ground-truth mask
- predicted mask
- overlay visualization

Importantly, the final compact four-panel format is not just coded, it is executed and saved. That is a major positive relative to the other notebooks in this project line.

Verdict: `PASS`

## Deliverables & Documentation Compliance

The notebook includes the expected internal documentation:

- dataset explanation
- architecture description
- training strategy
- hyperparameter section
- evaluation outputs
- visualizations

So the notebook itself is well-covered as a documented artifact.

The compliance issue is overclaiming. The notebook presents itself as a "complete Google Colab notebook submission," but the saved execution is local and the explicit Colab link deliverable is missing from the repo evidence.

Verdict: `PARTIAL`

## Assets Review

Repo-level deliverable check:

- trained weights: present at `kaggle/working/best_model.pth`
- scripts used: present in `Notebooks/helper functions`
- Colab notebook link: not found

Because the assignment explicitly asks for the Colab link with access permissions, this section cannot be marked complete.

Verdict: `PARTIAL`

## Bonus Criteria Review

The notebook only lightly touches the bonus criteria.

What is present:

- JPEG compression appears in augmentation
- noise-style augmentation is present

What is missing:

- no dedicated robustness experiments for JPEG, resize, crop, or noise
- no explicit subtle-tampering evaluation for copy-move or splicing from similar textures

This does not count as bonus fulfillment. It counts as bonus awareness at best.

Verdict: `FAIL`

## Brutal Roast

This notebook is the first version in this repo that actually looks like a submission instead of a promise. It runs, it shows metrics, it shows the masks, and it finally gives the reviewer something concrete to look at.

And then it immediately sabotages itself by keeping a broken legacy training path inside the same notebook.

If you are going to call something "submission-ready," you do not leave a dead benchmark path in the artifact where the train split is literally wired to the test CSV. That is not a cute historical note. That is engineering sloppiness inside the submission itself.

The effective run is clearly the real one, and it is much better. But even there, the localization story is not impressive. `0.8986` test accuracy looks nice for classification. `0.5761` Dice and `0.5526` IoU are baseline-level numbers, not brag-worthy forensic localization. So the notebook clears the bar for "working," but not for "strong."

The other avoidable own-goal is compliance theater. The notebook says "Google Colab notebook," but the saved output says `IN_COLAB: False`, and the repo does not contain the explicit Colab link the assignment asked for. That is exactly the sort of detail a strict reviewer will use to downgrade an otherwise decent technical submission.

In short: this is a credible internship baseline, but it is not polished enough to deserve a clean pass.

## Final Compliance Scorecard

| Requirement | Status | Comments |
|---|---|---|
| Dataset selection | PASS | Public authentic/tampered dataset with masks is used and documented. |
| Data pipeline | PARTIAL | Full pipeline exists, but the preserved old path is miswired and unreliable. |
| Architecture | PASS | U-Net-style segmentation with classifier head is valid for detect + localize. |
| Colab compatibility | PARTIAL | Likely runnable on T4-class Colab, but saved execution is local and no Colab link was found. |
| Evaluation metrics | PARTIAL | Accuracy, Dice, IoU, and F1 are present, but ROC-AUC and stronger detection metrics are missing. |
| Visualizations | PASS | Required executed four-panel visualization is present in the artifact. |
| Documentation | PASS | Dataset, architecture, training strategy, hyperparameters, results, and visuals are documented. |
| Deliverables | PARTIAL | Single notebook artifact is present, but compliance is weakened by the missing Colab link and legacy broken block. |
| Assets | PARTIAL | Weights and scripts exist, but the explicit Colab link deliverable is missing. |
| Bonus criteria | FAIL | Only augmentation-level hints are present, not actual robustness or subtle-tampering evaluation. |

## Final Verdict

`FAIL`

Overall score: `66 / 100`

Why it fails:

- the notebook contains a materially broken legacy benchmark path
- the explicit Colab notebook link deliverable is missing
- localization evaluation is only moderate and not especially rigorous
- bonus criteria are not meaningfully addressed

Why it is still respectable:

- the effective run is executed and saved
- the final metrics are present
- the final qualitative panels are present
- the core detection + localization objective is actually demonstrated

Bottom line:

This is a real, reviewable baseline submission with working outputs, not vaporware. But it falls short of full assignment compliance and professional polish. It deserves credit for being substantially more complete than the later notebook artifacts, but not a pass.
