# ASSIGNMENT COMPLIANCE VERDICT

Partial

This notebook proves a real Kaggle run happened, saved artifacts, and produced pixel-level masks, so it is not a fake submission artifact. Cells 29, 33, and 55 show a completed training run, test evaluation, and saved outputs. That said, it still does not fully satisfy the assignment as a serious engineering solution. Localization exists, but image-level detection is still a cheap `max(prob_map)` hack rather than a trained detector (cell 33). The architecture justification is thin baseline talk, not actual reasoning (cell 19). Runtime evidence is Kaggle-only, and the run used `DataParallel` across 2 GPUs with `batch_size=64`, `accumulation_steps=4`, and W&B plus `kaggle_secrets`, which is not the same thing as proving clean Colab portability (cells 5, 7, 20, 55).

Supporting audits:

- [01_Assignment_Compliance_and_Run01_Verdict.md](01_Assignment_Compliance_and_Run01_Verdict.md)
- [02_Data_Pipeline_Review.md](02_Data_Pipeline_Review.md)
- [03_Model_Architecture_and_Two_Head_Comparison.md](03_Model_Architecture_and_Two_Head_Comparison.md)
- [04_Training_Strategy_Audit.md](04_Training_Strategy_Audit.md)
- [05_Visualization_and_Interpretability_Audit.md](05_Visualization_and_Interpretability_Audit.md)
- [06_Run01_Result_Analysis.md](06_Run01_Result_Analysis.md)
- [07_Shortcut_Learning_and_Robustness_Audit.md](07_Shortcut_Learning_and_Robustness_Audit.md)
- [08_Engineering_Quality_and_Final_Fixes.md](08_Engineering_Quality_and_Final_Fixes.md)

# NOTEBOOK ROAST

1. The notebook pretends a segmentation spike is an image detector.

Why it is a problem: In cell 33, image-level detection is defined as `tamper_score = probs[i].view(-1).max().item()` and classified with the same threshold used for pixel masks. That is not a detection model. That is taking the hottest pixel and calling it a global decision. It is lazy, badly calibrated, and conceptually wrong for the assignment requirement that asks for both detection and localization.

What a senior ML engineer would expect instead: Train a real image-level head off the encoder bottleneck, supervise it with the existing labels, and report classification metrics from a learned detector instead of from a max-pooled segmentation map.

2. The threshold story is a calibration fire alarm, not a success story.

Why it is a problem: Cell 32 lands on a best threshold of `0.7500` and the notebook itself prints that `pos_weight may be too aggressive`. That is the notebook admitting the probability scale is distorted. Then the training loop, scheduler, and early stopping in cells 27 and 29 still optimize validation F1 at a fixed threshold of `0.5`. So training is optimized for one operating point and evaluated at another.

What a senior ML engineer would expect instead: Align model selection and threshold tuning to the same target metric, and treat a threshold that high as evidence of calibration trouble, not as a random post-processing tweak.

3. The mixed-set metrics are makeup on a weak tampered-only model.

Why it is a problem: Cell 33 reports mixed-set Pixel-F1 `0.5181`, but tampered-only Pixel-F1 is only `0.2949`. The notebook looks healthier the moment authentic empty-mask images are allowed to pad the averages. That is exactly the kind of metric presentation that hides a weak localizer behind a friendly class prior.

What a senior ML engineer would expect instead: Lead with tampered-only metrics, stratify by manipulation type and mask size, and explicitly treat mixed-set metrics as secondary bookkeeping.

4. The leakage check is path-overlap kindergarten.

Why it is a problem: Cell 12 checks only whether file paths overlap across train, val, and test. That proves almost nothing about content leakage. CASIA-style data can still leak near-duplicates, derivative manipulations, or sibling images while passing this toy check.

What a senior ML engineer would expect instead: Add duplicate or near-duplicate detection, source-group splitting where possible, and at minimum acknowledge that path disjointness is not a serious leakage audit.

5. You collected labels all the way through the pipeline and then refused to train on them.

Why it is a problem: The dataset returns `label` in cell 16, loaders propagate it in cell 17, evaluation uses it in cell 33, and training ignores it in cells 27 and 29. That is wasted supervision. The assignment explicitly asks for detection and localization, and the notebook had the labels in hand and still chose not to learn detection.

What a senior ML engineer would expect instead: Use the available supervision. A dual-head objective is the obvious fix, and the reference notebook already demonstrates the pattern.

6. The notebook spent effort on Grad-CAM garnish before the core segmentation story was convincing.

Why it is a problem: Cells 41 to 43 add Grad-CAM and TP/FP/FN overlays, which is fine in principle, but the model is still collapsing on copy-move (`0.1394` F1) and tiny masks (`0.1432` F1) in cell 33. Fancy explainability does not rescue weak fundamentals.

What a senior ML engineer would expect instead: First prove the model can localize the hard cases reliably. Then add explainability as a diagnostic, not as decoration.

# TOP 10 IMPLEMENTATION PROBLEMS

1. Image-level detection is not learned; it is a `max(prob_map)` heuristic in cell 33.
2. Validation, scheduler, and early stopping optimize F1 at threshold `0.5`, while final evaluation uses `0.75`.
3. Tampered-only localization is weak: Pixel-F1 `0.2949`, Pixel-IoU `0.2321` in cell 33.
4. Copy-move performance is terrible: F1 `0.1394` in cell 33.
5. Tiny-mask performance is terrible: F1 `0.1432` for masks under 2 percent in cell 33.
6. Mixed-set segmentation metrics are inflated by authentic empty-mask cases.
7. Leakage checking in cell 12 is only path disjointness, not content-level validation.
8. The pipeline sees `MASK/Au` in cell 9 and still fabricates zero masks for authentic images in cell 16 without validating those authentic masks.
9. `pos_weight` in cell 22 mixes raw mask pixel counts with a resized-image estimate for authentic images, which is an inconsistent calculation.
10. Runtime portability is overstated: the proven run used 2 GPUs, Kaggle paths, `kaggle_secrets`, and W&B sync.

# TOP 5 THINGS DONE WELL

1. There is a real executed run with checkpoints, plots, summaries, and artifact inventory, not just unexecuted notebook theater.
2. The config, checkpointing, and experiment bookkeeping are materially better than the average internship notebook.
3. The notebook reports tampered-only metrics first in cell 33, which is the right instinct even if the model is still weak.
4. Failure analysis and robustness sections exist and surface real weaknesses, especially small-mask and copy-move failure modes.
5. The pipeline is modular enough to support future loss and scheduler swaps without rewriting the whole notebook.

# WHAT THE AUTHOR CLEARLY UNDERSTANDS

1. How to build and run an end-to-end segmentation training pipeline on Kaggle.
2. That segmentation metrics can be misleading without tampered-only reporting and size/type breakdowns.
3. That imbalance matters and needs explicit treatment.
4. That saved artifacts, checkpoint resume logic, and tracked experiments matter for reproducibility.
5. That qualitative inspection and robustness checks belong in a serious notebook, even if the current implementation is not yet sharp enough.

# WHAT THE AUTHOR LIKELY DOES NOT UNDERSTAND YET

1. Detection and localization are related tasks, not interchangeable outputs from the same thresholded probability map.
2. A high threshold is usually a symptom of calibration trouble, not evidence of rigor.
3. Path-level split checks are not the same thing as leakage control.
4. Mixed-set segmentation metrics can flatter a bad model when authentic images dominate easy negatives.
5. Architecture justification means comparing task-fit tradeoffs, not saying "Docs said the bottleneck is elsewhere."

# SHOULD THE MODEL USE A TWO-HEAD ARCHITECTURE?

Yes. The current notebook already carries image-level labels through the whole pipeline and then wastes them. The reference notebook's `UNetWithClassifier` in `Pre-exisiting Notebooks/image-detection-with-mask.ipynb` proves the pattern is feasible: one encoder, one segmentation head, one classification head. Its implementation is crude, but the idea is still better than pretending the hottest segmentation pixel is a detector.

The upside is straightforward: a learned classifier head gives a real detection objective, decouples image-level thresholding from pixel thresholding, and lets the encoder learn both global and local tampering cues. The downside is extra loss balancing and some risk of negative transfer if classification starts dominating. That tradeoff is still worth it here because the current "detector" is not a detector at all.

# HOW TO IMPROVE NOTEBOOK V8 BEFORE FINAL SUBMISSION

1. Add a learned image-level classification head and stop using `max(prob_map)` as a fake detector.
2. Rework model selection so early stopping, scheduler decisions, and final threshold tuning target the same validation objective.
3. Audit content leakage properly with duplicate or near-duplicate checks instead of just path overlap.
4. Validate authentic masks explicitly because the dataset clearly contains `MASK/Au` files and the current pipeline ignores them.
5. Stop squashing all images to `384x384` without preserving aspect ratio if you care about small tampered regions.
6. Make the qualitative section brutally honest: show copy-move failures, tiny masks, false positives on authentic images, and probability maps, not just a curated grid.
7. Prove a single-GPU Colab-safe configuration, or stop implying the current config is cloud-portable by default.
8. Revisit `pos_weight`, calibration, and threshold behavior before claiming the current training strategy is principled.
