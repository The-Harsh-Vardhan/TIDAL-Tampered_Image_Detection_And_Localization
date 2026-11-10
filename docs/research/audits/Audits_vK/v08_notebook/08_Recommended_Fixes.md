# 08 - Recommended Fixes

## Must fix before submission

1. Preserve execution evidence.
   Re-run the notebook end to end and submit it with execution counts, visible outputs, and the generated artifacts. Right now the notebook proves nothing.

2. Remove Kaggle lock-in.
   Replace hard-coded `/kaggle/input`, `/kaggle/working`, and `kaggle_secrets` usage with environment-aware path handling so the same notebook runs on Colab without edits.

3. Separate detection from localization.
   Add a proper image-level detection head or explicitly narrow the project claim to segmentation-first localization with heuristic detection. Stop pretending `max(prob_map)` is a finished detector.

4. Report tampered-only metrics first everywhere.
   Make tampered-only localization the primary validation, test, and robustness target. Mixed-set scores can stay as secondary context, not as the headline.

5. Harden leakage checks.
   Add source-aware splitting or at least perceptual-hash duplicate checks. Path disjointness is not enough for forensic datasets.

6. Justify or benchmark at least one alternative architecture.
   Keep U-Net plus ResNet34 if you want, but benchmark it against one obvious alternative such as DeepLabV3+ or a simple dual-head baseline so the architecture story is earned, not asserted.

7. Fix `pos_weight` computation.
   Compute class balance in one consistent pixel space. Either use native mask sizes for every sample or resized training masks for every sample. Do not mix them.

8. Stop using one threshold for two tasks.
   Calibrate segmentation and image-level detection separately. The current shared threshold is lazy and technically weak.

## Should fix for credibility

1. Tune threshold on the right objective.
   Optimize threshold on tampered-only validation F1 or another localization-centric target, not on mixed validation averages padded by authentic empties.

2. Replace cosmetic shortcut tests with real falsification tests.
   Drop the random-mask theater. Add tampered-only robustness reporting, artifact-suppression checks, and controls that actually pressure-test whether the model is learning manipulation evidence.

3. Validate authentic masks explicitly.
   If the dataset ships `Mask/Au`, inspect and verify them instead of silently zero-filling every authentic sample.

4. Analyze resize damage.
   Measure mask-area shift and failure rate by mask size after the `384 x 384` resize. Tiny regions are where this pipeline is most likely to break.

5. Use a better checkpoint-selection signal.
   Stop driving scheduler and early stopping from thresholded F1 at `0.5`. Use validation loss, a threshold-free score, or a tampered-only calibrated metric.

6. Tighten config discipline.
   Remove dead or misleading config knobs such as unused ratios, and make resume behavior explicit instead of automatic magic.

## Optional upgrades

1. Add a dual-head model.
   A shared encoder with a segmentation decoder plus image-level classification head would align far better with the assignment than the current heuristic detector.

2. Add forensic input streams.
   Residual-based channels, ELA-style features, or frequency-domain cues would make the project more defensible for tamper analysis than pure RGB.

3. Add boundary metrics.
   Boundary F1 or boundary IoU would tell you more about localization quality than region overlap alone.

4. Add cross-dataset evaluation.
   Even a small external sanity check would be more convincing than an expanded same-dataset robustness section.

5. Simplify the notebook until the baseline is proven.
   Keep W&B, Grad-CAM, and artifact upload if you want, but only after the notebook has a trusted core run. Right now the presentation layer is ahead of the science.
