# 05 - Evaluation and Metric Trust Audit

## Bottom line

The evaluation code is the most dangerous part of the notebook because it looks thoughtful while still baking in several ways to flatter or confuse the results. On top of that, none of the reported outcomes are preserved in the notebook outputs, so the whole evaluation stack remains unverified.

## 1. No outputs means no trusted results

Start with the obvious. The notebook has no execution counts and no cell outputs. That means there is no evidence the threshold sweep, test evaluation, robustness charts, or artifact inventory ever ran. Any metric discussion beyond code inspection is conditional, not confirmed.

That alone is enough to sink trust.

## 2. Mixed-set F1 is inflated by design

Cell 26 returns Pixel-F1 of `1.0` when both prediction and ground truth are empty. That means every correctly predicted authentic image contributes a perfect score. When those authentic images are averaged together with tampered examples, the localization metric becomes padded by cases that contain no localization challenge at all.

This is the classic way segmentation reports get inflated:

- empty authentic images become free points,
- mixed-set average rises,
- the model looks less bad than it really is on tampered regions.

The notebook knows this problem exists. Cell 30 even announces tampered-only reporting. Then Cell 32 still tunes threshold on mixed validation F1 anyway.

## 3. Threshold tuning optimizes the wrong objective

Cell 32 sweeps thresholds on the validation set and maximizes mean Pixel-F1 over all validation samples. That means authentic empty-mask cases influence threshold selection. For a tamper localization assignment, that is the wrong target.

If the real goal is localizing tampered regions, threshold search should be driven by tampered-only localization quality or at least reported alongside it. Otherwise the model can gain threshold credit by being very good at predicting nothing on authentic images.

## 4. One threshold is reused for two different tasks

Cell 33 uses the same threshold for:

- turning pixel probabilities into a binary segmentation mask, and
- turning the image-level `max(prob_map)` score into a binary tamper decision.

That is technically lazy. Pixel-level localization and image-level detection are different operating problems. There is no reason the same threshold should be optimal for both unless the author just could not be bothered to calibrate them separately.

## 5. Image-level detection is especially weak

The image-level score in Cell 33 is `tamper_score = probs[i].view(-1).max().item()`. That means the whole image label depends on the single hottest pixel in the map. One spurious activation can flip the decision. That is not robust detection. It is a softmax-free landmine.

The AUC number might still look decent on paper, but the scoring rule is brittle and not grounded in a real classification head.

## 6. Edge-case metric handling is questionable

`compute_precision_recall()` in Cell 26 returns:

- `(1.0, 1.0)` if both ground truth and prediction are empty,
- `(0.0, 1.0)` if ground truth is empty and prediction is non-empty.

That second case is mathematically awkward. Recall is effectively undefined when there are no positives in the ground truth. Hard-coding it to `1.0` may be defensible for one niche convention, but if you average that value into global recall you are manufacturing interpretability that is not really there.

This is another example of the evaluation stack choosing convenience over clarity.

## 7. Training-time model selection is inconsistent with final evaluation

Cell 29 picks checkpoints based on validation F1 at threshold `0.5`. Cell 32 later finds a different threshold for final evaluation. So even the checkpointing logic and the final evaluation logic do not optimize the same operating point.

That creates an ugly ambiguity:

- best model at threshold 0.5,
- best threshold found later on the same validation split,
- reported test metrics using the post hoc threshold.

It is not outright invalid, but it is messy and easy to overstate.

## 8. The good part

The notebook at least tries to report:

- tampered-only metrics,
- mixed-set metrics,
- per-forgery breakdown,
- mask-size buckets,
- image-level detection,
- threshold sweeps.

That is better than a toy notebook that dumps one IoU number and calls it a day. The author knows what a serious evaluation section should contain. The problem is the trustworthiness of the metric design and the total absence of executed evidence.

## Verdict

The evaluation stack is not trustworthy enough for submission. The main reasons are:

1. zero execution evidence,
2. mixed-set inflation from empty authentic masks,
3. threshold tuning on the wrong objective,
4. reuse of one threshold for two tasks,
5. brittle image-level scoring based on one maximum pixel.

Right now the notebook has an evaluation section that sounds mature and behaves like a compromise.
