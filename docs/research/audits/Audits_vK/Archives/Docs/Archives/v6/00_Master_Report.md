# Docs6 — Master Report

**Revision:** 6
**Date:** March 2026
**Primary notebook:** `tamper_detection_v5.1_kaggle.ipynb`
**Runtime:** Kaggle (T4 GPU)
**Overall status:** Final documentation revision. All Audit 5 issues resolved. Docs aligned with v5.1 notebook.

---

## 1. Summary of Changes from Docs5

| Area | Docs5 State | Docs6 Resolution |
|---|---|---|
| Runtime environment | Google Colab with Drive | Kaggle with `/kaggle/working/` output |
| Image resolution | 512 × 512 | 384 × 384 (T4 VRAM headroom) |
| Data split | 85 / 7.5 / 7.5 | 70 / 15 / 15 (more balanced evaluation) |
| Mask binarization | `> 128` | `> 0` (captures all annotated pixels) |
| Dataset discovery | Hardcoded `Image/` and `Mask/` paths | Case-insensitive `os.walk` discovery (handles `IMAGE/`, `MASK/`, `New folder/` nesting) |
| Artifact storage | Google Drive or local fallback | `/kaggle/working/` (checkpoints, results, plots) |
| W&B authentication | Interactive `wandb.login()` | Kaggle Secrets: `UserSecretsClient().get_secret("WANDB_API_KEY")` |
| Precision/recall edge case | Inconsistent true-negative handling | Returns `(1.0, 1.0)` when both prediction and GT are empty |
| Tampered image validation | Only masks checked for corruption | Both tampered images and masks validated with `is_valid_image()` |
| Data leakage check | Not implemented | Set-intersection assertions across train/val/test splits |
| Grad-CAM safety | No error handling | `try/except`, None-check on hook data, graceful fallback |
| Prediction grid | No empty guard | `n_rows == 0` guard prevents crash on empty subsets |
| Final print message | Implied Google Drive even on local fallback | Reports actual output directory |
| Notebook reference | `tamper_detection_v5.ipynb` | `tamper_detection_v5.1_kaggle.ipynb` |
| Artifact naming | `results_summary_v5.json` | `results_summary.json` |
| References document | Not present | New `13_References.md` added |

---

## 2. Architecture Summary

1. Run a Kaggle notebook with T4 GPU. Dataset is pre-mounted at `/kaggle/input/`.
2. Dynamically discover dataset root using case-insensitive directory matching for `IMAGE/` and `MASK/`.
3. Discover image-mask pairs with corruption guards, dimension validation, and forgery-type classification.
4. Persist the stratified train/val/test split (70/15/15) to `split_manifest.json` with data-leakage verification.
5. Train `smp.Unet` with `encoder_name="resnet34"` and `encoder_weights="imagenet"` using BCE + Dice loss, AdamW, AMP, gradient accumulation, gradient clipping, and early stopping.
6. Select the best threshold via validation-set sweep (50 thresholds, 0.1–0.9) and freeze it for all downstream evaluation.
7. Evaluate localization with mixed-set and tampered-only views, plus image-level accuracy and AUC-ROC.
8. Generate visualizations, Grad-CAM analysis, failure-case analysis, and robustness results.
9. Optionally log the full run to W&B behind `USE_WANDB`.

---

## 3. Document Index

| File | Purpose |
|---|---|
| 01_System_Architecture.md | End-to-end system overview and data flow |
| 02_Dataset_and_Preprocessing.md | Dataset selection, discovery, cleaning, splitting, and validation |
| 03_Model_Architecture.md | U-Net baseline and image-level detection strategy |
| 04_Training_Strategy.md | Loss, optimizer, training loop, checkpointing |
| 05_Evaluation_Methodology.md | Metrics, threshold protocol, reporting views |
| 06_Robustness_Testing.md | Degradation suite and evaluation protocol |
| 07_Visualization_and_Explainability.md | Required figures, Grad-CAM, overlays, failure analysis |
| 08_Engineering_Practices.md | Environment, dependencies, reproducibility |
| 09_Experiment_Tracking.md | Guarded W&B integration and fallback artifacts |
| 10_Project_Timeline.md | Execution roadmap |
| 11_Research_Alignment.md | Research grounding for design decisions |
| 12_Complete_Notebook_Structure.md | Actual notebook v5.1 section/cell map |
| 13_References.md | Dataset, research papers, and reference notebook citations |

---

## 4. Audit 5 Issue Resolution

| Audit 5 Finding | Resolution in Docs6/v5.1 |
|---|---|
| No critical blockers found | Confirmed — v5.1 maintains this status |
| Final print message implies Drive under local fallback | Fixed — v5.1 reports actual `/kaggle/working/` output path |
| Image-level detection remains heuristic | Documented as known limitation; dual-task head listed as future work |
| Grad-CAM is diagnostic, not rigorous explainability | Documentation clarified — labeled as lightweight diagnostic tool |
| CASIA lacks source-image grouping metadata | Acknowledged; leakage-check assertions added but cannot fully eliminate risk |
| Runtime validation still needed | Documented as operational requirement; docs describe design, not runtime proof |
| Colab operational risks (Kaggle auth, Drive mount) | Eliminated — v5.1 runs on Kaggle with pre-mounted dataset |

---

## 5. Known Limitations

1. **Source-image leakage risk** — CASIA does not expose source-image group metadata. Leakage-check assertions verify no file overlap across splits, but content-level leakage is not detectable.
2. **Image-level heuristic** — the top-k mean tamper score is more stable than `max(prob_map)`, but a dedicated classification head would be stronger.
3. **Dataset size** — CASIA is small by modern standards. Overfitting risk remains real despite augmentation and early stopping.
4. **Classical tampering scope only** — GAN edits, deepfakes, and AI-generated manipulations are out of scope.
5. **Baseline positioning** — this project is an assignment-scoped baseline, not a frontier forensics system.
6. **Annotation quality** — CASIA-derived masks may contain coarse or noisy boundaries that cap achievable localization quality.

---

## 6. Cross-Document Consistency

Current consistency guarantees:

- All notebook references point to `tamper_detection_v5.1_kaggle.ipynb`.
- All artifact references use the v5.1 naming and `/kaggle/working/` paths.
- Notebook-structure docs match the actual v5.1 layout: **61 cells / 17 sections**.
- Evaluation docs match the implemented precision/recall handling and image-level score.
- Engineering and W&B docs match the Kaggle-native artifact behavior.
- Dataset docs match the case-insensitive discovery and 70/15/15 split.
