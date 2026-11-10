# Audit 2 Master Report

This audit evaluates the rewritten documentation set in `Docs 2/`, using `Audit 1/` as the baseline for issue-resolution tracking and `Assignment.md` as the assignment source of truth. It is documentation-first, but it also includes a lightweight notebook-alignment note because `tamper_detection.ipynb` now exists and materially supports the Colab-feasibility claim.

## 1. Overall Assessment

The rewritten documentation is substantially stronger than the prior round and is now mostly technically sound. The architecture is appropriate for the assignment: CASIA v2.0, preprocessing and mask binarization, a U-Net with a pretrained ResNet34 encoder, BCE + Dice loss, validation-based threshold selection, evaluation with IoU/Dice/precision/recall and image-level detection, visualization of binary masks and overlays, and optional robustness testing.

The biggest improvement is that the docs now read like one implementation path instead of several competing plans. Major issues from `Audit 1` were addressed: hard-coded dataset counts were removed, the split procedure was clarified, baseline versus optional phases were aligned, the SMP model API mismatch was fixed, mixed-set versus tampered-only evaluation was added, the main visualization now uses the binary predicted mask, and the final checklist clearly separates MVP from optional work.

The remaining problems are narrower and mostly operational:
- image-level scoring is still not fully locked
- the evaluation snippets still blur pixel-threshold and image-threshold handling
- resize degradation is described correctly but not fully wired into the robustness loop
- the default dependency install command omits `kaggle`
- `albumentations` parameter compatibility remains a runtime risk
- the checklist treats actual T4 detection as a requirement rather than T4 compatibility

- Technical soundness: Mostly yes.
- Assignment fit: Yes.
- Overall score: `8/10`

## 2. Document-by-Document Review

| Document | Purpose | Accuracy | Remaining issues | Detailed review |
|---|---|---:|---|---|
| `01_Assignment_Overview.md` | Scope, deliverables, and constraints | 9 | Minor scope wording around bonus items and ancillary scripts | [Review](Docs/01_Assignment_Overview.review.md) |
| `02_Dataset_and_Preprocessing.md` | Dataset selection, cleaning, and split policy | 8 | Group-aware split still unavailable; minor discovery-function edge cases | [Review](Docs/02_Dataset_and_Preprocessing.review.md) |
| `03_Data_Pipeline.md` | Dataset class, augmentation policy, and loaders | 8 | Phase 2 augmentation API compatibility still version-sensitive | [Review](Docs/03_Data_Pipeline.review.md) |
| `04_Model_Architecture.md` | U-Net baseline and encoder selection | 8 | Image-level score choice is still not frozen | [Review](Docs/04_Model_Architecture.review.md) |
| `05_Training_Strategy.md` | Loss, optimizer, loop, and checkpointing | 8 | Scheduler resume path is only described, not fully shown | [Review](Docs/05_Training_Strategy.review.md) |
| `06_Evaluation_Methodology.md` | Metrics and threshold protocol | 7 | Pixel/image threshold separation is still not fully encoded in the example | [Review](Docs/06_Evaluation_Methodology.review.md) |
| `07_Visualization_and_Results.md` | Required figures and layout | 9 | Small snippet hygiene issues only | [Review](Docs/07_Visualization_and_Results.review.md) |
| `08_Robustness_Testing.md` | Bonus degradation testing | 7 | Resize degradation is not fully integrated into the shown loop | [Review](Docs/08_Robustness_Testing.review.md) |
| `09_Engineering_Practices.md` | Colab setup and reproducibility | 7 | Default install path omits `kaggle` despite using the CLI | [Review](Docs/09_Engineering_Practices.review.md) |
| `10_Project_Timeline.md` | Phase-based execution order | 9 | No major remaining conflicts | [Review](Docs/10_Project_Timeline.review.md) |
| `11_Limitations_and_Future_Work.md` | Explicit limits and out-of-scope ideas | 9 | Leaves the final image-level score choice open | [Review](Docs/11_Limitations_and_Future_Work.review.md) |
| `12_Final_Submission_Checklist.md` | Pre-submission verification | 8 | T4 verification wording is stricter than necessary | [Review](Docs/12_Final_Submission_Checklist.review.md) |

## 3. Conflict Resolution Check

Most of the high-signal contradictions from `Audit 1` were resolved.

- Resolved: hard-coded dataset counts and hard-coded misalignment counts were removed in favor of dynamic validation and logging.
- Resolved: the split procedure is now explicit and the split manifest is persisted for reproducibility.
- Resolved: baseline versus optional augmentation is now aligned across `03_Data_Pipeline.md`, `10_Project_Timeline.md`, and `12_Final_Submission_Checklist.md`.
- Resolved: the `model.unet.*` mismatch is fixed; the docs now consistently refer to direct SMP attributes.
- Resolved: the LR scheduler moved out of the MVP baseline and into Phase 2.
- Resolved: the training loop now flushes the final partial accumulation window.
- Resolved: mixed-set and tampered-only reporting is explicitly documented.
- Resolved: the primary visualization now uses the binary predicted mask.
- Resolved: a separate limitations/future-work document exists.
- Resolved: the final checklist clearly separates MVP, optional Phase 2, and bonus Phase 3 work.

Remaining contradictions are small but real:
- `04_Model_Architecture.md`, `06_Evaluation_Methodology.md`, and `11_Limitations_and_Future_Work.md` acknowledge alternatives to `max(prob_map)` but do not lock one final image-level score.
- `06_Evaluation_Methodology.md` says the image-level threshold may differ from the pixel threshold, but the example evaluation function still uses one threshold argument for both.
- `08_Robustness_Testing.md` fixes the resize-mask issue conceptually, but the shown evaluation loop does not yet demonstrate the resize-degraded image path.

See [01_Conflict_Resolution_Check.md](01_Conflict_Resolution_Check.md) for the direct Audit 1 comparison.

## 4. Implementation Risk Assessment

The plan is now realistic for Google Colab, but a few implementation risks remain:

- The image-level detection rule is still under-specified. A single locked choice is needed before implementation is considered fully decision-complete.
- The evaluation examples should separate `pixel_threshold` and `image_threshold` if the docs intend them to differ.
- The robustness section needs one concrete resize-evaluation path, not only a helper function and prose guidance.
- The default setup instructions should install `kaggle` if they rely on the Kaggle CLI.
- `albumentations` arguments such as `quality_lower` and `var_limit` may still break on newer versions unless the notebook pins a tested version.

These are moderate risks, not structural blockers. The baseline pipeline itself is feasible on Colab T4 and is also reflected in the repository notebook.

## 5. Final Architecture Summary

Final recommended pipeline:

1. Download CASIA v2.0 in a single Colab notebook.
2. Discover image-mask pairs dynamically, validate alignment, binarize masks, generate zero masks for authentic images, and persist the split manifest.
3. Train an RGB `smp.Unet` with `encoder_name="resnet34"` using BCE + Dice, AdamW, AMP, gradient accumulation, checkpointing, and early stopping.
4. Select the operating threshold on the validation set only.
5. Evaluate on the test set with mixed-set and tampered-only localization metrics plus image-level accuracy and AUC.
6. Visualize original image, ground-truth mask, binary predicted mask, and overlay.
7. Run robustness testing only after the core pipeline is complete.

## 6. Final Recommendations

- Freeze one final image-level tamper score and document it everywhere the same way.
- Split the evaluation examples into `pixel_threshold` and `image_threshold` if those are truly separate operating points.
- Add `kaggle` to the default install command or make the setup explicitly assume it is preinstalled.
- Show one concrete implementation path for resize degradation in the robustness loop.
- Pin a tested `albumentations` version if the notebook will use legacy parameter names such as `quality_lower` and `var_limit`.
