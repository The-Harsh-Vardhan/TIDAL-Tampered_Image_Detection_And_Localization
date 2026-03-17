# Docs4 — Master Report

**Revision:** 4 (supersedes Docs 3)
**Date:** March 2026
**Overall status:** All Audit 3 issues resolved. Implementation-ready final documentation.

---

## 1. Summary of Changes from Docs 3

| # | Issue (from Audit 3) | Docs 3 state | Docs 4 fix |
|---|---|---|---|
| 1 | ELA channel-count contradiction | `01` said 6 channels; `03` and `10` said 4 channels | Standardized to **4th channel** everywhere. SRM is a separate path with its own channel config. Cross-reference section added to 01, 03, and 10. |
| 2 | Undefined `B` in `prob_map.view(B, -1)` | `B` not defined in code snippet in `03` | Fixed: `B = prob_map.size(0)` |
| 3 | Unquoted `albumentations` pip install | `08` used `albumentations>=1.3.1,<2.0` unquoted | Quoted: `"albumentations>=1.3.1,<2.0"` — matches notebook |
| 4 | W&B setup unconditional despite "optional" claim | `09` setup section showed bare install/login/init | Rewritten: entire setup section is inside `if USE_WANDB:` guard. No install, import, or login occurs when disabled. |
| 5 | Subtle-tampering overclaim | `00` mapped bonus "subtle tampering detection" to forgery-type breakdown | Reworded: forgery-type breakdown describes analysis of known CASIA categories, not a subtle-tampering capability |
| 6 | Missing precision/recall helper | `05` mentioned precision and recall but had no implementation | Added `compute_precision_recall()` helper function in `05` |
| 7 | VRAM estimate treated as verified | `01` stated VRAM budget as fact | Labeled as "estimated" — needs runtime confirmation |
| 8 | Explainability weak in visualization doc | `07` title said "Explainability" but had no attribution methods | Renamed to "Visualization and Results." Added feature-map inspection as optional Phase 2 interpretability tool. Added scope note explaining no formal explainability methods are used. |
| 9 | Uncited research claims in robustness | `06` stated forensic-context claims without citations | Reworded as general observations "consistent with the image forensics literature" rather than specific cited claims |
| 10 | Phase 2→3 ordering too strict | `10` required Phase 2 complete before Phase 3 | Softened: Phase 3 items can proceed independently once MVP is stable |
| 11 | Pre-installed package assumption | `08` listed packages as "Pre-installed in Colab (do not install)" | Reworded to "Typically available in Colab (verify before relying on)" with note about runtime image variability |
| 12 | W&B fallback not documented | `09` had no guidance for non-W&B artifacts | Added "Fallback Artifacts" section listing local Drive artifacts (JSON, checkpoints, manifests) |
| 13 | Master report overstated closure | `00` said "Addresses all critical and high-priority findings" | Removed blanket closure claim; this revision tracks each fix individually |

---

## 2. Architecture Summary

1. Download CASIA v2.0 via Kaggle API in a single Colab notebook.
2. Discover image–mask pairs dynamically; validate alignment; binarize masks; generate zero masks for authentic images; persist the split manifest.
3. Optionally compute Error Level Analysis (ELA) maps and concatenate as a 4th channel with RGB (Phase 2).
4. Train `smp.Unet` with `encoder_name="resnet34"`, `encoder_weights="imagenet"` using BCE + Dice loss, AdamW, AMP, gradient accumulation, checkpointing, and early stopping.
5. Select the operating threshold on the validation set only.
6. Evaluate on the test set with mixed-set and tampered-only localization metrics plus image-level accuracy and AUC.
7. Visualize original image, ground-truth mask, binary predicted mask, and overlay.
8. Run robustness testing (JPEG, noise, blur, resize) only after the core pipeline is complete.
9. Optionally log experiments to W&B — guarded behind `USE_WANDB` flag; notebook runs without it.

---

## 3. Document Index

| File | Purpose |
|---|---|
| [01_System_Architecture.md](01_System_Architecture.md) | End-to-end system overview and data flow |
| [02_Dataset_and_Preprocessing.md](02_Dataset_and_Preprocessing.md) | Dataset, cleaning, splitting, and validation |
| [03_Model_Architecture.md](03_Model_Architecture.md) | U-Net baseline, encoder, output format, optional features |
| [04_Training_Strategy.md](04_Training_Strategy.md) | Loss, optimizer, training loop, checkpointing |
| [05_Evaluation_Methodology.md](05_Evaluation_Methodology.md) | Metrics, threshold protocol, reporting views |
| [06_Robustness_Testing.md](06_Robustness_Testing.md) | Degradation suite with concrete implementation |
| [07_Visualization_and_Results.md](07_Visualization_and_Results.md) | Required figures, grid layout, diagnostic purpose of each visual |
| [08_Engineering_Practices.md](08_Engineering_Practices.md) | Environment, dependencies, reproducibility |
| [09_Experiment_Tracking.md](09_Experiment_Tracking.md) | Guarded W&B integration, fallback artifacts, sweep guidance |
| [10_Project_Timeline.md](10_Project_Timeline.md) | Three-phase roadmap with flexible Phase 2/3 ordering |

---

## 4. Remaining Known Limitations

1. **Data split integrity** — CASIA v2.0 has no source-image grouping; related images may leak across splits. Mitigated by stratified splitting and manifest persistence.
2. **Image-level fragility** — `max(prob_map)` is sensitive to single false positives. Acknowledged; top-k mean is a Phase 3 alternative.
3. **Dataset size** — ~5,000 images is small by modern standards. Augmentation and pretrained encoder partially compensate.
4. **Colab session limits** — Free tier has idle timeout. Checkpointing to Drive mitigates.
5. **No formal explainability** — Feature-map inspection is offered as an optional interpretability tool, but no gradient-based attribution or attention visualization is implemented.

---

## 5. Relationship to Assignment

Every section in `Assignment.md` is covered:

| Assignment section | Docs 4 coverage |
|---|---|
| 1. Dataset Selection & Preparation | 02, 08 |
| 2. Model Architecture & Learning | 03, 04 |
| 3. Testing & Evaluation | 05, 06 |
| 4. Deliverables & Documentation | 07, 08, 09 |
| Bonus: Robustness | 06 |
| Bonus: Forgery-type analysis | 05 (breakdown by known CASIA categories) |

---

## 6. Cross-Document Consistency

All cross-document conflicts identified in Audit 3 have been resolved:

| Conflict | Resolution |
|---|---|
| ELA channel count (01 vs 03 vs 10) | Standardized to 4th channel everywhere; SRM documented as separate path |
| pip install quoting (08 vs notebook) | Quoted in doc to match notebook |
| W&B optional vs unconditional (09 internal) | Entire setup block inside `if USE_WANDB:` |
| Subtle-tampering overclaim (00 vs 05) | Reworded to "forgery-type analysis of known CASIA categories" |
