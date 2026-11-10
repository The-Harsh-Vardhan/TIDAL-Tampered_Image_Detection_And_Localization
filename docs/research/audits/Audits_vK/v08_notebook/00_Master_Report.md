# ASSIGNMENT COMPLIANCE VERDICT

Fail

This notebook does not satisfy the assignment in `Assignment.md`. The assignment asks for a single Google Colab notebook that demonstrates tampered image detection and localization, runs in a cloud notebook environment, explains architectural choices clearly, and presents trustworthy evaluation. `notebooks/v8-tampered-image-detection-localization-kaggle.ipynb` fails on all three of the hard requirements that matter most.

First, this is an unexecuted artifact, not evidence of a completed experiment. The notebook cells have no execution counts and no preserved outputs, so there is no trustworthy proof that the pipeline ran end to end. Second, the notebook is hard-wired to Kaggle, not Colab. Cell 5 writes to `/kaggle/working`, Cell 8 says the dataset is pre-mounted by Kaggle, Cell 9 crawls `/kaggle/input`, and Cell 7 imports `kaggle_secrets` for W&B login. That is not a portable Colab deliverable. Third, the notebook still does not implement a real image-level detector. Cell 33 derives detection from `max(prob_map)`. That is a thresholded segmentation side effect pretending to be a detector.

The supporting audits are here:

- [01_Assignment_Compliance_And_Deliverable_Gaps.md](01_Assignment_Compliance_And_Deliverable_Gaps.md)
- [02_Data_Pipeline_Audit.md](02_Data_Pipeline_Audit.md)
- [03_Model_Architecture_And_ML_Reasoning_Audit.md](03_Model_Architecture_And_ML_Reasoning_Audit.md)
- [04_Training_Strategy_Audit.md](04_Training_Strategy_Audit.md)
- [05_Evaluation_And_Metric_Trust_Audit.md](05_Evaluation_And_Metric_Trust_Audit.md)
- [06_Shortcut_Learning_And_Robustness_Audit.md](06_Shortcut_Learning_And_Robustness_Audit.md)
- [07_Engineering_Quality_And_Runtime_Audit.md](07_Engineering_Quality_And_Runtime_Audit.md)
- [08_Recommended_Fixes.md](08_Recommended_Fixes.md)

# PROJECT ROAST

1. This is not a finished notebook. It is a code dump wearing a notebook costume.
   Why it is a problem: The assignment is about demonstrated work, not fictional future tense. No execution counts, no outputs, no preserved metrics, no proof. A static notebook is not a result.
   What a senior ML engineer would expect instead: A fully executed notebook with visible outputs, saved artifacts, and enough evidence to reproduce the claims without guessing.

2. The submission is Kaggle-locked while pretending to satisfy a Colab deliverable.
   Why it is a problem: Cell 5, Cell 7, Cell 8, and Cell 9 hard-code Kaggle paths and Kaggle secrets. The assignment explicitly requires a single Google Colab notebook. This notebook does not even bother to hide that it only knows one environment.
   What a senior ML engineer would expect instead: Environment detection, portable path handling, optional secret usage, and a notebook that actually runs on the required platform without surgery.

3. The "detection" part is fake.
   Why it is a problem: Cell 33 computes `tamper_score = max(prob_map)`. That is not a modeled image-level detection head. That is a lazy heuristic strapped to a segmentation output and then reported as if the system solves both tasks cleanly.
   What a senior ML engineer would expect instead: Either a real joint detection-and-localization architecture or an explicit statement that image-level detection is only a provisional heuristic.

4. The architecture reasoning is outsourced to Docs8 instead of earned in the notebook.
   Why it is a problem: Cell 19 says the architecture stays because "Docs8 Section 03 determined" the bottleneck is elsewhere. That is not reasoning. That is citation laundering. The notebook never proves U-Net plus ResNet34 is the right answer for tamper localization.
   What a senior ML engineer would expect instead: A tight baseline justification plus at least one meaningful alternative comparison or a very narrow claim that this is only a convenience baseline.

5. The evaluation is built to flatter the model.
   Why it is a problem: Cell 26 awards Pixel-F1 of 1.0 when both prediction and ground truth are empty. Cell 32 tunes threshold on mixed validation F1. Cell 33 then reports mixed-set and tampered-only numbers together. Authentic images can hand the model free points while the threshold search optimizes the wrong target.
   What a senior ML engineer would expect instead: Threshold selection and primary reporting centered on tampered images, with image-level and pixel-level tasks calibrated separately.

6. The data leakage check is kindergarten-level.
   Why it is a problem: Cell 12 only checks that file paths do not overlap across splits. That proves almost nothing. It does not catch near-duplicates, derivatives from the same source image, or dataset-family leakage.
   What a senior ML engineer would expect instead: Source-aware splitting, duplicate detection, and at minimum a perceptual-hash sanity check.

7. The class-imbalance fix is half right and still sloppy.
   Why it is a problem: Cell 22 tries to compute `pos_weight`, but it mixes raw mask pixel counts for tampered images with `image_size ** 2` estimates for authentic images. That is inconsistent accounting. The notebook is pretending precision while mixing incompatible units.
   What a senior ML engineer would expect instead: Compute class balance on the actual training masks after the exact spatial preprocessing pipeline or use a consistent native-resolution accounting scheme.

8. The shortcut-learning section is theater.
   Why it is a problem: Cell 50 compares predictions against arbitrary random masks and expects F1 near zero as if that proves anything. Then it runs boundary sensitivity only on samples with F1 above 0.1, which quietly drops the embarrassing failures. That is not a stress test. That is cosmetic damage control.
   What a senior ML engineer would expect instead: Content-matched controls, tampered-only robustness reporting, boundary metrics, and failure analysis that includes the failures instead of filtering them out.

9. The notebook spends more time on ceremony than on proof.
   Why it is a problem: W&B, Grad-CAM, plots, artifact inventory, and resume logic are all present. The one thing missing is the boring part that actually matters: an executed, credible training and evaluation record.
   What a senior ML engineer would expect instead: First prove the baseline works. Then add tracking and explainability once the metrics are trustworthy.

# TOP 10 IMPLEMENTATION PROBLEMS

1. The notebook has no execution counts or outputs, so none of the claimed behavior is proven.
2. The submission is hard-coded to Kaggle and does not satisfy the single-Colab-notebook deliverable.
3. Image-level detection is not modeled; it is `max(prob_map)` in Cell 33.
4. Mixed-set segmentation metrics are inflated by authentic empty masks scoring perfectly in Cell 26.
5. Threshold search in Cell 32 optimizes mean mixed validation F1, not the real localization objective.
6. The same threshold is reused for segmentation and image-level detection in Cell 33.
7. `pos_weight` in Cell 22 mixes raw mask sizes with resized authentic-image estimates.
8. Data leakage checks in Cell 12 only prove path disjointness, not true split integrity.
9. Architecture justification in Cell 19 is convenience-driven and outsourced to Docs8.
10. Shortcut-learning checks in Cell 50 are weak, biased, and mostly performative.

# TOP 5 THINGS DONE WELL

1. The notebook has a centralized config in Cell 5 instead of scattering magic numbers everywhere.
2. Cells 10 through 13 at least attempt explicit dataset discovery, validation, and split manifesting.
3. Cells 28 and 29 implement checkpointing and resume support instead of pretending notebook state is a real experiment log.
4. Cell 30 at least acknowledges that tampered-only metrics should be reported prominently.
5. Cells 44 through 50 show the author knows failure analysis, robustness, and shortcut learning are the right places to look for model fraud.

# WHAT THE AUTHOR CLEARLY UNDERSTANDS

The author understands the basic anatomy of a modern segmentation baseline. The code shows familiarity with SMP, pretrained encoders, BCE plus Dice style losses, mixed precision, gradient accumulation, checkpointing, and standard augmentation tooling. The author also clearly understands that tamper localization is vulnerable to class imbalance, threshold sensitivity, copy-move difficulty, and shortcut learning. The existence of dedicated sections for robustness, failure cases, and artifact saving shows the right instincts about what a serious ML project should eventually contain.

The author also understands how to make a notebook look organized. There is a real attempt at configuration, reproducibility hooks, split manifesting, checkpoint management, and result packaging. That is better than the average internship notebook.

# WHAT THE AUTHOR LIKELY DOES NOT UNDERSTAND YET

The author does not yet understand the difference between a credible experiment and a plausible script. A notebook full of sections is not the same thing as a validated submission. The author also seems to confuse localization output with full task coverage. A segmentation model plus `max(prob_map)` is not a properly designed detection-and-localization system.

There is also a gap in metric discipline. The notebook knows empty masks inflate mixed-set scores and still tunes threshold on mixed validation F1. It knows calibration matters and still uses a single threshold for two different tasks. It knows leakage matters and still stops at path overlap checks. That combination signals partial understanding: the vocabulary is there, but the standards are not.

# HOW TO IMPROVE NOTEBOOK V8

1. Re-run the notebook end to end and preserve execution counts, outputs, and saved artifacts. Without that, everything else is noise.
2. Remove Kaggle lock-in. Add environment-aware path and secret handling so the same notebook runs on Colab without edits.
3. Separate detection from localization. Add a real image-level detection head or clearly downgrade the detection claim.
4. Recompute threshold selection on tampered-only localization metrics, not mixed validation F1.
5. Stop reusing the segmentation threshold for image-level detection. Calibrate detection separately.
6. Fix `pos_weight` accounting so class balance is computed in a consistent pixel space.
7. Replace path-overlap leakage checks with source-aware splitting and duplicate detection.
8. Benchmark at least one obvious alternative such as DeepLabV3+ or a lightweight dual-head baseline.
9. Report tampered-only metrics first everywhere, including robustness tests.
10. Keep the notebook simple until the core experiment is proven. Right now the ceremony is ahead of the science.
