# Evaluation Methodology

## How to explain this in an interview

Start with this:

"Because this is a segmentation project, I evaluated both localization quality and image-level detection quality. The most important idea is that threshold selection was done on the validation set only, then frozen before test evaluation."

## What is being evaluated

This project has two evaluation targets:

- pixel-level localization
- image-level tamper detection

That is why the evaluation is split into:

- segmentation metrics like IoU and F1
- detection metrics like accuracy and AUC-ROC

## Why evaluation is a big part of this project

In tamper localization, a model can look good in one way and still fail badly in another.

For example:

- it might detect tampering but draw poor masks
- it might produce decent masks but too many false positives
- it might work only at one threshold

That is why the evaluation pipeline needs more than one metric.

## IoU

### What it is

IoU, or Intersection over Union, measures how much the predicted tampered region overlaps with the ground-truth region.

Formula idea:

- intersection = pixels both predicted and true as tampered
- union = pixels predicted or true as tampered

### What problem it solves

It directly measures spatial overlap, which is the real goal in localization.

### Why it was chosen here

IoU is especially important for segmentation because it penalizes both:

- missed tampered regions
- oversized predicted masks

That makes it a strong measure of mask quality.

### Alternatives

- Dice coefficient
- pixel accuracy
- boundary-focused metrics

### Why those were not enough alone

- Dice is useful, but IoU is often stricter and easier to discuss in segmentation interviews.
- Pixel accuracy can be misleading because background pixels dominate.
- Boundary metrics are interesting, but not necessary for the first baseline.

## Precision

### What it is

Precision asks:

"Of the pixels I predicted as tampered, how many were actually tampered?"

### What problem it solves

It measures false-positive control.

### Why it matters here

In forensic settings, too many false alarms reduce trust in the model.

## Recall

### What it is

Recall asks:

"Of the truly tampered pixels, how many did I successfully find?"

### What problem it solves

It measures missed tampered regions.

### Why it matters here

If recall is low, the model may miss subtle edits even if precision looks strong.

## F1 score

### What it is

F1 balances precision and recall.

### What problem it solves

It provides one summary number when both false positives and false negatives matter.

### Why it was chosen here

The project uses Pixel-F1 as the main selection metric because it gives a strong balance between:

- finding manipulated regions
- avoiding noisy masks

## Image-level accuracy and AUC-ROC

The project also evaluates whether the whole image is correctly classified as tampered or authentic.

Why this matters:

- some users need a quick decision
- some systems need ranking or triage before manual review

The project uses a top-k mean tamper score derived from the segmentation map, then reports:

- image accuracy
- image AUC-ROC

## Why IoU is particularly important for segmentation

If an interviewer asks why IoU matters so much, this is the simple answer:

"IoU tells me whether the predicted region actually overlaps the true manipulated region in a meaningful way. A model can get some positive pixels right by chance, but IoU forces it to localize the region more accurately."

That is why IoU is more informative than plain pixel accuracy in this project.

## Threshold tuning

### What it is

The model outputs probabilities, not hard binary masks. To get a final mask, I need a threshold.

### What problem it solves

A threshold controls the tradeoff between:

- precision
- recall

### Why it was chosen here

The project sweeps thresholds on the validation set only and chooses the threshold that gives the best validation Pixel-F1.

That threshold is then reused for:

- test-set mask binarization
- image-level decisions
- robustness evaluation

### Why this is important

This avoids test leakage. If I tune the threshold on the test set, the evaluation becomes optimistic and less trustworthy.

## Mixed-set vs tampered-only reporting

The project reports two useful views:

### Mixed-set

This includes all test images:

- tampered
- authentic

This gives a realistic full-pipeline view.

### Tampered-only

This looks only at tampered images.

This is useful because it isolates localization quality without letting authentic zero-mask cases dominate the summary.

## Authentic-image edge case handling

This is a good interview point because it shows metric maturity.

Authentic images have empty ground-truth masks. That creates a metric edge case:

- if prediction is also empty, that is a correct result
- but per-image precision and recall are not naturally defined in the same way as for positive masks

To avoid misleading numbers, the project reports:

- mixed-set precision and recall as global pixel metrics
- tampered-only precision and recall separately

That is a cleaner evaluation design than forcing every authentic image into a per-image precision-recall average.

## Alternatives that could have been used

- calibration metrics
- boundary IoU
- PR curves for pixel decisions
- separate threshold for image-level detection

## Why those were not selected for the MVP

They add useful depth, but the current metric set already gives a strong and defendable evaluation story for an interview and for the assignment.

## Future improvements

If I extended the evaluation pipeline, I would add:

- cross-dataset testing
- calibration analysis
- boundary-sensitive metrics
- classwise robustness summaries
- operating-point analysis for different deployment priorities

## How I would summarize the evaluation

"I evaluated the model both as a localizer and as an image-level detector. IoU and Pixel-F1 were the key localization metrics, and the threshold was tuned only on the validation set to avoid leakage. I also separated mixed-set and tampered-only reporting so the metrics stayed honest, especially for authentic zero-mask images."
