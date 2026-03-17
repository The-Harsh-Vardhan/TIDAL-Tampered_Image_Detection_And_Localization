# Audit-vK.7.1

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

## 1. Executive Summary

This notebook implements a multitask computer-vision pipeline for tampered image detection and tampered-region localization using a U-Net-style encoder-decoder with an added image-level classification head. At a high level, that is a valid baseline for the assignment: one branch predicts whether an image is authentic or tampered, and the other predicts a binary manipulation mask (cells 59, 61, 69).

The central problem is not that the notebook lacks the right ideas. The problem is that the saved artifact does not prove what the markdown claims it proves. The notebook is heavily documented and much more readable than a typical student submission, but as an audited artifact it is closer to a plausible baseline notebook than a fully evidenced experimental submission. The saved file has 91 total cells, 42 code cells, 49 markdown cells, only 10 code cells with execution counts, and only 7 cells with outputs. The main effective training/evaluation path is unexecuted in the saved artifact (cells 71, 73, 75), and the final four-panel assignment visualization is also unexecuted (cell 90).

Overall technical quality is moderate. The implementation is coherent, the architecture is acceptable, and the notebook narrative is far better than average. But the evaluation evidence, execution consistency, and forensic rigor are not strong enough to call this a tight submission.

## 2. Assignment Alignment Audit

### Dataset Selection

The notebook uses a CASIA-style authentic/tampered dataset with masks, discovered through a Kaggle-first pipeline with optional Colab/Drive and Kaggle API fallbacks (cells 6, 9, 13, 15, 21, 23, 25). This satisfies the assignment's dataset requirement in principle.

### Mask Usage for Localization

Mask usage is implemented correctly at a basic level. The dataset class loads grayscale masks, binarizes them, and returns them alongside the input image and image-level label (cell 59). The segmentation head produces a one-channel output map and is trained with BCE-plus-Dice supervision (cells 61, 65, 69).

### Segmentation Model Correctness

The model is a legitimate dual-task CNN for this assignment. It uses a shared U-Net-like backbone with a bottleneck classifier head for image-level prediction and a decoder head for localization (cell 61). That satisfies the requirement that the system perform both image-level detection and pixel-level localization.

### Evaluation Metrics

Quantitative metrics are implemented in code. The notebook computes image-level accuracy plus Dice, IoU, and F1 for segmentation (cells 67, 69, 73). However, the effective training and final test evaluation cells are not executed in the saved notebook (cells 71, 73, 75), so those metrics are implemented but not demonstrated in the artifact.

### Required Visual Outputs

The required side-by-side visualization of original image, ground-truth mask, predicted mask, and overlay exists in code in the submission-ready panel function (cell 90). But that final cell is unexecuted and has no output in the saved artifact. Earlier qualitative visualization cells are executed and show overlays or predicted masks (cells 82, 84, 87), but they do not fully substitute for the exact final assignment panel.

### Verdict

**PARTIALLY ALIGNED**

Explanation: the notebook contains the required dataset logic, mask handling, multitask model, and evaluation/visualization code, but the saved notebook artifact does not actually show the key quantitative results or the final required four-panel visualization. As code it is aligned. As a submission artifact it is only partially aligned.

## 3. Dataset and Data Pipeline Review

The runtime setup is practical for Kaggle and reasonably thoughtful about fallbacks. The notebook standardizes around `/kaggle/input` and `/kaggle/working`, supports Drive search in Colab, and falls back to Kaggle API download only when necessary (cells 6, 9, 11, 13, 15). That is a sensible operational design.

The metadata builder is simple and readable, but technically brittle. It assumes that every image has a mask with the exact same filename in the mirrored `MASK/<class>` directory (cell 23). That assumption is common, but it is not validated beyond filesystem existence.

The more serious issue is that rows with missing masks are still included in the metadata CSV. The notebook explicitly records `mask_exists` and even prints missing-mask examples (cells 23, 25), but there is no explicit filtering step before the downstream dataset loader reads `mask_path` and attempts `cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)` unconditionally (cell 59). If missing masks are present, the pipeline does not fail early and cleanly; it fails later during training or evaluation.

The split logic is only label-stratified random splitting (cell 27). This preserves the authentic/tampered ratio, which is good, but it does nothing to control for near-duplicates, derivative manipulations from the same source image, or source-level leakage. In image forensics, that matters. If multiple manipulated variants or closely related images are spread across train/val/test, measured performance can be overly optimistic.

The augmentation pipeline is a mixed bag. Resizing, horizontal flipping, brightness/contrast perturbation, and mild affine jitter are standard and defensible (cells 37, 57). The effective block adds Gaussian noise and JPEG compression (cell 57), which are actually relevant to forensic robustness. That said, these augmentations can also blur or distort the very traces the model is supposed to localize, especially around fine boundaries. They may help coarse tamper detection more than precise mask quality.

## 4. Model Architecture Review

The architecture is a valid multitask baseline. The encoder-decoder backbone is structurally correct for localization, the skip connections are appropriate, and the classifier head attached to the bottleneck is a reasonable way to add image-level detection without building a separate network (cell 61).

Its strengths are straightforward:

- It is conceptually correct for joint detection and localization.
- The U-Net structure is a sensible starting point for binary mask prediction.
- The shared feature backbone keeps the model simple and assignment-friendly.
- The classification head is computationally cheap relative to the segmentation trunk.

Its weaknesses are also obvious:

- It is a generic vision architecture, not a forensic-specialized one.
- There is no pretrained encoder, so the model must learn everything from scratch.
- There is no explicit mechanism to emphasize subtle manipulation artifacts, boundaries, or compression inconsistencies.
- Shared features can favor coarse image-level discrimination over fine mask delineation.

The likely failure mode is that the model learns broad tamper cues and obvious splice regions but struggles on small, subtle, low-contrast, or boundary-sensitive manipulations. The notebook also saves the best checkpoint using validation classification accuracy rather than localization quality (cell 71), which further biases the final selected model toward the detection task instead of the localization task the assignment also requires.

## 5. Training Pipeline Audit

The effective training stack is reasonable at a baseline level. Adam with learning rate `1e-4`, CosineAnnealingLR, a focal-style classification loss, BCE-with-logits plus Dice for segmentation, and gradient clipping are all defensible choices (cells 65, 69). Batch size 8 and `256x256` inputs are realistic for Kaggle/Colab hardware (cell 63).

The strengths are:

- Losses are appropriate for a binary classification plus binary segmentation setup.
- Class weighting is used for the classifier branch (cell 65).
- Gradient clipping is included for some stability (cell 69).
- The reporting metrics cover more than just loss on the segmentation side (cells 67, 69).

The weaknesses are where the pipeline starts looking thin:

- No mixed precision or AMP is used, so efficiency is worse than it should be.
- No global seed is set for Python, NumPy, or PyTorch.
- No early stopping is used.
- Checkpoint selection is based only on validation accuracy (cell 71), not Dice/IoU/F1 or a multitask criterion.

That last point is the most important one. For a dual-task system, saving the model that maximizes classification accuracy is not the same thing as saving the model that best localizes tampered regions. The notebook's training logic can therefore produce a model that looks good on image-level detection while underperforming on the pixel-level task the assignment explicitly requires.

## 6. Evaluation and Metrics Review

The notebook separates detection and localization conceptually, but the evaluation design is incomplete.

### Detection Performance

Detection is effectively measured by accuracy only (cells 69, 71, 73). That is too weak for a binary forensic classifier. The notebook does not report precision, recall, F1, confusion matrix, ROC-AUC, or PR-AUC. Accuracy alone can hide class-asymmetric failure modes and does not tell the reviewer whether the classifier is conservative, over-sensitive, or badly calibrated.

### Localization Performance

Localization metrics are better in intent. Dice, IoU, and F1 are all implemented and reported from thresholded masks at `0.5` (cells 67, 69, 73). That is a reasonable starting point.

But there are two technical weaknesses:

1. Metrics are averaged at the batch level rather than aggregated over the full dataset, which introduces small weighting distortions when batch sizes differ.
2. Localization metrics are computed over all samples in the split, including authentic images with empty masks (cell 69).

That second issue is the major flaw. If the model predicts empty masks on authentic images, Dice/IoU/F1 can be inflated even if tampered-region localization is mediocre. The notebook does not report tampered-only localization metrics, which would be much more meaningful.

### Demonstrated Evidence

Even the existing metric design is not convincingly demonstrated in the saved artifact. The effective training loop, final test evaluation, and training-curves cells are unexecuted (cells 71, 73, 75), so the notebook does not actually display the final quantitative evidence it claims to provide.

## 7. Visualization and Results Review

The notebook includes several qualitative visualization routines, and some of them are executed. The collected sample pipeline is present (cell 81), overlay views are shown (cell 82), image-plus-predicted-mask panels are shown (cells 83, 84, 86, 87), and these give some evidence that the model is being used for inference.

However, the exact assignment-ready visualization is the four-panel comparison in cell 90, where each row contains:

- Original Image
- Ground Truth Mask
- Predicted Mask
- Overlay

That is the right final presentation. But it is not executed in the saved notebook. So the code quality of the visualization section is better than the delivered evidence. The artifact shows partial qualitative evidence, not the strongest and most assignment-aligned visual proof.

As a result, the qualitative results are suggestive but not fully convincing.

## 8. Notebook Engineering Quality

From a readability standpoint, the notebook is much improved. The markdown structure is clear, the sections are well labeled, and the flow is easier to audit than a monolithic notebook. That is a real strength.

From an engineering standpoint, it is still inconsistent.

The biggest issue is duplication. The notebook contains both a source-preserved earlier experiment block and a second “effective submission” training pipeline (cells 29-45 and 51-75). That duplication makes it harder to know which code path should be trusted as the real experiment.

The second issue is state incoherence. The main effective training/evaluation cells are unexecuted (cells 71, 73, 75), but downstream model-loading and visualization cells are executed (cells 77-87). That is not a clean top-to-bottom artifact. It is a stateful notebook with broken provenance.

There are also smaller engineering rough edges:

- hardcoded `/kaggle/working/...` paths remain in the effective training block (cell 55)
- W&B integration is useful but brittle and secrets-dependent (cell 50)
- notebook claims in markdown are stronger than what execution state supports (cells 3, 91)

So the notebook is polished in presentation, but not disciplined in execution hygiene.

## 9. Kaggle / Colab Feasibility

This notebook is realistically feasible on Kaggle GPU. A full-width U-Net with a `1024`-channel bottleneck at `256x256` and batch size `8` is not lightweight, but it is generally manageable on common Kaggle GPUs such as T4 or P100-class hardware (cells 61, 63).

It is also probably feasible on Google Colab GPU, but less elegantly. The notebook does not really become native Colab code. Instead, it emulates Kaggle paths inside Colab by creating `/kaggle/input` and `/kaggle/working`, then normalizes the dataset into that structure (cells 6, 13, 15). That is workable, but it is clearly fallback behavior, not a first-class Colab implementation.

The main inefficiency is the lack of AMP. Without mixed precision, training will be slower and memory headroom tighter than necessary. Still, the dataset size and I/O burden appear manageable, and the pipeline is practical enough for an assignment notebook.

## 10. Reproducibility Audit

Reproducibility is weak.

The split process uses fixed `random_state=42` (cell 27), but there is no global seeding for Python, NumPy, or PyTorch anywhere in the notebook. That means training remains non-deterministic across reruns.

The saved artifact is also not reproducible as a clean experiment trace. The effective training block is unexecuted, yet later cells assume the existence of `best_model_path`, `TRAINING_HISTORY`, and other runtime state when producing visualizations (cells 71, 73, 75, 77, 81). That suggests the notebook was not saved from a coherent single run.

Dependency control is only partial. Albumentations and OpenCV are pinned in the notebook (cells 31, 53), but core framework versions are not fully frozen. W&B online logging further depends on Kaggle secrets (cell 50), which adds another external prerequisite.

Another user could probably rerun the implementation with some effort. Another user could not confidently treat the saved notebook itself as a reproducible research artifact.

## 11. Brutal Technical Roast

This notebook is the ML equivalent of dressing a baseline up in a blazer and hoping nobody notices the missing results.

You clearly understand the assignment. You implemented the dual-task model. You wrote the metrics. You even wrote the exact four-panel submission visualization the reviewer wants to see. Then you stopped one step short of proving anything in the saved artifact.

That is the frustrating part. The technical core is not embarrassing. The experimental discipline is.

You left the notebook in a state where the markdown says “fulfilled,” but the execution trail says “trust me.” The effective training block is unexecuted. The final test metrics are not shown. The final assignment-style panel is not rendered. Yet the notebook still talks like the case is closed.

That is not how a strong submission works. A strong submission makes the reviewer’s job easy. This one makes the reviewer reconstruct your intent from unexecuted code and partially executed downstream cells.

The model choice is also safe to the point of being timid. Plain U-Net, plain classifier head, plain thresholded metrics, no serious forensic reasoning, no leakage control, no tampered-only localization analysis, no calibration view, no confusion matrix. This is a baseline, not a compelling forensic study.

As an internship submission, I would call it competent but not trustworthy. The code suggests you can build. The artifact suggests you do not yet think like a reviewer.

## 12. Improvement Recommendations

### High impact improvements

- Execute the effective training loop, final test evaluation, training-curves cell, and the final four-panel visualization cell, then save the notebook with those outputs present.
- Change checkpoint selection from validation accuracy to a criterion that reflects localization quality, such as validation Dice on tampered images or a multitask score.
- Report tampered-only localization metrics so empty authentic masks do not inflate the apparent segmentation quality.
- Add stronger image-level detection metrics: precision, recall, F1, confusion matrix, and ROC-AUC.

### Medium improvements

- Filter or repair `mask_exists == 0` rows before training, rather than recording them and failing later in the dataset loader.
- Add proper seeding for Python, NumPy, and PyTorch.
- Remove or isolate the duplicated earlier experiment block so only one authoritative training path remains.
- Replace remaining hardcoded `/kaggle/working/...` strings with the notebook's path variables for consistency.

### Nice-to-have improvements

- Add AMP for faster, more memory-efficient training on Kaggle and Colab GPUs.
- Add a short failure-analysis section with representative false positives and false negatives.
- Consider a pretrained encoder or a lightweight forensic-aware feature strategy if you want stronger performance without redesigning the project.
- Add a brief discussion of leakage risk and why the split strategy is acceptable or limited.

## Final Verdict

**Technical Quality Score (0–10): 6.0**

**Assignment Alignment Score (0–10): 5.0**

**Engineering Quality Score (0–10): 5.5**

**Final Overall Assessment**

A technically reasonable multitask baseline wrapped in a cleaner notebook structure, but still not a convincing submission artifact. The core detection/localization system is implemented, yet the saved notebook does not actually prove the required quantitative and qualitative results. As code, it is plausible. As an audited assignment submission, it is only partially aligned and not reviewer-tight.
