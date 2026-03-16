# Shortcut Learning And Robustness Audit

## The core risk

CASIA-style tamper datasets are dangerous because the model can cheat.

Common shortcut channels include:

- boundary artifacts from pasted regions
- compression mismatch between manipulated and untouched areas
- color or illumination discontinuities
- source-dataset quirks tied to manipulation type

A model can look decent on in-dataset metrics while mostly learning those artifacts instead of the actual semantics of tampering.

## The robustness section is useful, but the metric choice is flattering

Cell 47 reports robustness using mean Pixel-F1 over the whole test set:

- clean `0.5181`
- jpeg_qf70 `0.5338`
- jpeg_qf50 `0.5092`
- gaussian noise light `0.3878`
- gaussian noise heavy `0.3894`
- gaussian blur `0.4755`
- resize_0.75x `0.4650`
- resize_0.5x `0.4731`

The problem is that `0.5181` is the same mixed-set F1 from cell 33. That metric already benefits from authentic easy negatives. So the robustness section is partially judging robustness using an inflated baseline.

Senior expectation: report tampered-only robustness, and ideally stratify it by forgery type and mask size.

## The JPEG result does not prove the shortcut problem is solved

Cell 47 prints:

`JPEG robustness gap (0.009) is within acceptable range.`

That is too confident. A small JPEG gap can mean:

- augmentation helped
- the model is genuinely robust
- the metric is being cushioned by authentic easy cases
- the dataset's main learned shortcuts are not strongly JPEG-sensitive

The notebook jumps from "small gap" to "acceptable." That is not rigorous.

## The mask-randomization test is weak evidence

Cell 50 reports:

- Mask Randomization Test F1 `0.0772`

That is not useless, but do not oversell it. Low F1 against random masks only shows the model is not trivially matching arbitrary binary patterns. It does not show the model is not exploiting pasted boundaries, compression seams, or dataset-specific texture weirdness.

This is a weak sanity check, not a shortcut-learning exorcism.

## The boundary-sensitivity analysis cherry-picks the survivors

Cell 50 runs boundary sensitivity only on:

`tampered predictions if p['pixel_f1'] > 0.1`

That means the analysis excludes many failures before measuring sensitivity. Then it reports:

- original F1 `0.5818`
- eroded `0.5857`
- dilated `0.5649`

That tells you the already decent predictions are not hypersensitive to tiny morphological edits. Fine. It does not tell you whether boundary artifacts were driving the many bad predictions you filtered out.

## What the notebook should do to detect shortcuts more seriously

1. Report robustness on tampered-only subsets, not just the mixed test set.
2. Break robustness down by forgery type, especially copy-move versus splicing.
3. Visualize authentic false positives and inspect whether they cluster around compression or texture edges.
4. Run duplicate or near-duplicate checks across splits.
5. Compare performance on boundary-dilated or boundary-eroded ground truth, not just on filtered predictions.
6. Test whether probability maps collapse when obvious artifact cues are suppressed or cropped.

## Shortcut-learning verdict

The notebook at least knows shortcut learning is a real risk. That is better than most student work.

But the current checks are still more suggestive than conclusive:

- robustness uses a flattering aggregate metric
- JPEG stability is overinterpreted
- mask randomization is a weak sanity check
- boundary sensitivity filters away bad cases

So the right conclusion is not "shortcut risk addressed." The right conclusion is "shortcut risk acknowledged, lightly probed, and still very much open."
