# CASIA 2.0 Image Tampering Dataset — Project Audit

| Field | Value |
|-------|-------|
| **Project** | Tampered Image Detection & Localization |
| **Assignment** | Big Vision Internship Assignment |
| **Audit Date** | 2026-03-16 |
| **Dataset** | CASIA v2.0 (12,614 images) |
| **Framework** | PyTorch + Segmentation Models PyTorch (localization track); TensorFlow/Keras (classification track) |
| **Best Result** | Pixel F1 = 0.7329, IoU = 0.5785, Image Accuracy = 87.53% (vR.P.15) |

---

## Executive Summary

This project represents an ambitious and iterative attempt at image tampering detection and localization. The candidate explored **4 distinct research lineages** (v0x documentation-first, vK.x Kaggle reproduction, vR.x ETASR classification, vR.P.x pretrained localization), ultimately converging on a UNet + ResNet-34 + ELA pipeline that produces pixel-level tampered region masks. The project contains **232 notebooks**, **1,123 markdown documentation files**, and tracks **38 experiments** (21 completed, 16 pending) via Weights & Biases. The final system achieves a Pixel F1 of 0.7329, demonstrating competent localization performance on a challenging forensic dataset.

The project excels in experimental rigor, documentation depth, and scale of exploration. It falls short in final deliverable polish: the required single-notebook submission is not cleanly packaged, the research journey document is incomplete, and only one model weight file is provided.

**Overall Score: 74 / 100**

---

## Section-by-Section Evaluation

### 1. Dataset Selection & Preparation (12 / 15)

**What the assignment requires:**
- Use a publicly available tampering dataset with ground truth masks
- Build a complete data pipeline with cleaning, preprocessing, mask alignment
- Proper train/validation/test split
- Data augmentation for robustness

**What was delivered:**
- **Dataset:** CASIA v2.0 (7,491 authentic + 5,123 tampered = 12,614 images) — an appropriate and widely-used benchmark
- **Data pipeline:** ELA preprocessing at multiple quality levels (Q=75, 85, 95), proper mask loading and alignment, stratified 70/15/15 split with seed=42 for reproducibility
- **Mask handling:** Ground truth binary masks correctly aligned to 384×384 resolution; authentic images paired with all-black masks
- **Augmentation:** Tested in vR.P.12 (horizontal flip, vertical flip, random rotation, color jitter) — yielded best image-level accuracy (88.48%)
- **Data leakage awareness:** Identified and fixed data leakage in the earlier vK.x lineage — shows strong data hygiene awareness

**Gaps:**
- Only one dataset used (CASIA v2.0) — no cross-dataset evaluation on Columbia, Coverage, or NIST
- No systematic analysis of dataset quality issues (e.g., CASIA v2.0 has known mask alignment problems in some samples)

**Score rationale:** Strong pipeline with proper splits, reproducible seeding, and leakage awareness. Minor deduction for single-dataset evaluation.

---

### 2. Model Architecture & Learning (13 / 15)

**What the assignment requires:**
- Train a model to predict tampered regions
- Architecture and loss function choice is open
- Must be runnable on Google Colab (T4 GPU compatible)

**What was delivered:**
- **Architecture:** UNet with ResNet-34 encoder pretrained on ImageNet (from `segmentation_models_pytorch`) — a well-justified choice for pixel-level segmentation
- **Encoder strategy:** Frozen convolutional weights, unfrozen BatchNorm layers — adapts to forensic input while maintaining pretrained features (~3.17M trainable out of 24.4M total)
- **Attention:** CBAM (Channel + Spatial) in decoder blocks — adds only 11K parameters for +3.57pp Pixel F1 gain
- **Loss functions tested:** BCE + Dice (standard), Focal + Dice (P.9), Edge supervision (P.25), Dual-task (P.26)
- **Training config:** Adam (lr=1e-3, weight decay 1e-5), ReduceLROnPlateau, early stopping, batch size 16, mixed precision (AMP + TF32)
- **T4/P100 compatible:** All experiments run on Kaggle GPU instances

**Additionally explored (classification track):**
- ETASR custom CNN (vR.x series): 11 ablation experiments on a research paper architecture
- Best classification: 90.23% accuracy, Macro F1 0.9004 (vR.1.6)

**Gaps:**
- Did not explore transformer-based encoders (acknowledged in future work)
- Multi-Quality ELA + CBAM combination (vR.P.30 series) still pending/in-progress

**Score rationale:** Thoughtful architecture choice with strong ablation discipline. Multiple loss functions tested. Good use of pretrained models. Minor deduction for not fully combining best components.

---

### 3. Testing & Evaluation Metrics (11 / 15)

**What the assignment requires:**
- Thorough evaluation of localization performance and image-level detection accuracy
- Standard, industry-accepted metrics

**What was delivered:**

**Pixel-level metrics (localization):**
- Pixel F1 (Dice coefficient) — primary metric
- Pixel IoU (Intersection over Union)
- Pixel AUC
- Pixel Precision and Recall

**Image-level metrics (detection):**
- Image Accuracy
- Image Macro F1
- Image ROC-AUC
- Confusion matrix

**Evaluation infrastructure:**
- `experiment_results.csv` tracking 38 experiments with standardized columns
- W&B logging of per-epoch train/val metrics + final test metrics
- Centralized leaderboard notebook with 13-step analysis pipeline
- Reproducibility verified: vR.P.3 and vR.P.10 replicated with identical metrics

**Gaps:**
- No cross-dataset generalization testing (e.g., train on CASIA, test on Columbia)
- No robustness testing against JPEG compression, resizing, cropping, noise (this was explicitly listed as a bonus point opportunity)
- Per-class breakdown (splicing vs. copy-move) not reported
- No statistical significance testing (confidence intervals, multiple seeds beyond replication)

**Score rationale:** Comprehensive metric suite covering both pixel and image levels. Strong reproducibility. Loses points for missing robustness testing (a bonus criterion) and no cross-dataset evaluation.

---

### 4. Visual Results (10 / 10)

**What the assignment requires:**
- Clear visual results comparing Original Image, Ground Truth, Predicted output, and Overlay Visualization

**What was delivered:**
- **4-panel visualization grid:** Original → Ground Truth → Predicted Mask → Overlay for each sample
- Present in every vR.P.x experiment notebook (35+ versions)
- **Prediction examples:** 20-sample grids showing correct and incorrect predictions
- **Training curves:** Loss and accuracy plots per epoch
- **ROC curves and Precision-Recall curves** in evaluation sections
- **Confusion matrices** with heatmap visualization
- **W&B integration:** Evaluation plots and prediction examples logged as W&B images

**Score rationale:** Full marks. All four required visualization panels are present. Additional diagnostic visualizations (ROC, PR curves, confusion matrices, training curves) go beyond requirements.

---

### 5. Deliverables: Single Notebook (5 / 10)

**What the assignment requires:**
- The entire implementation must be in a **single Google Colab Notebook**
- Notebook must include dataset explanation, model architecture description, training strategy, hyperparameter choices, evaluation results, and clear visualizations

**What was delivered:**
- **Not a single notebook.** The project spans 232 notebooks across multiple tracks
- The "Best Notebooks" selection (5 curated notebooks) partially addresses this, with vR.P.10 recommended as the primary submission
- Individual notebooks are well-documented with markdown cells explaining each section
- Each notebook is self-contained (imports through evaluation in one file)
- Best notebooks include: dataset explanation, architecture description, training config, evaluation, and visualizations

**Gaps:**
- No unified "final submission" notebook that combines the best findings into a single, clean deliverable
- The candidate acknowledges this implicitly by having a "Submission Ready Checklist" folder — but the final merged notebook doesn't exist
- Running the submission notebook requires Kaggle, not Google Colab (though both have T4 GPUs)

**Score rationale:** Each individual experiment notebook is self-contained and well-structured, but the explicit assignment requirement of a **single** unified notebook was not met. The "Best Notebooks" curation partially compensates, but a reviewer would need to choose among 5 options rather than receiving one polished deliverable.

---

### 6. Deliverables: Assets (5 / 10)

**What the assignment requires:**
- Colab Notebook Link (with access permissions)
- Trained model weights
- Any additional scripts used

**What was delivered:**
- **Notebook access:** Notebooks are in a Git repository (local), with executed runs on Kaggle — but no direct Colab link provided
- **Model weights:** Only `best_model.pth` (1 file) found in `kaggle/working/`. This appears to be from a single experiment, not the best-performing one (vR.P.15)
- **Scripts:** Build scripts (notebook generators) are created and deleted per workflow — some remain (e.g., `upgrade_etasr_notebooks.py`, `gen_runners.py`, `fix_ela_quality.py`)
- **W&B project:** Full experiment tracking available via Weights & Biases (if shared)

**Gaps:**
- No explicit model weight for. the best experiment (vR.P.15 Multi-Q ELA)
- No shared Colab/Kaggle link documented in deliverables
- Model checkpoints from Kaggle runs are ephemeral (lost when Kaggle session ends) unless explicitly downloaded
- No requirements.txt or environment specification

**Score rationale:** Critical gap in model weights — only 1 `.pth` file exists, and it may not correspond to the best experiment. The assignment explicitly requires trained model weights as a deliverable.

---

### 7. Research Report Quality (9 / 10)

**What was delivered:**
- `Research_Report.md`: 363 lines, structured as a formal academic paper
  - Abstract, Introduction, Related Work, Methodology, Experimental Setup, Results, Ablation Study, Visual Results, Key Learnings, Future Improvements, Experiment Tracking
- Well-written with clear technical language
- Reports 23 experiments with quantitative results
- Includes ablation summary table with per-experiment deltas
- Honestly reports negative results (TTA -5.32pp, DCT -37.11pp)
- Impact hierarchy clearly articulated

- `Submission_Report.md`: 252 lines, polished executive summary with TL;DR, pipeline diagram, results table, and key findings

**Gaps:**
- No reference to published literature (no citations in Research_Report.md, only mentions "ETASR" paper)
- Could benefit from comparison to state-of-the-art results on CASIA v2.0

**Score rationale:** Excellent for an internship submission. Clear, honest, well-structured. Minor deduction for lack of literature comparison.

---

### 8. Research Journey & Experimentation Documentation (4 / 5)

**What was delivered:**
- `Research_Journey_and_Experimentation.md`: 14 lines — **incomplete**, ends with `[Then details about this lineage and it's results]` placeholder
- However, the journey is extensively documented elsewhere:
  - `Submission_Report.md` Section 4 covers the full research journey narrative
  - Per-version documentation in `Docs vR.P.x/` (29 version folders with experiment descriptions, implementation plans, expected outcomes)
  - `Audit_new_runs_vR_ETASR/` contains audit reports per experiment
  - Git history shows the iterative evolution clearly

**Gaps:**
- The primary research journey file is clearly unfinished
- The journey narrative is fragmented across many files instead of one cohesive document
- No timeline or effort allocation breakdown

**Score rationale:** The journey is well-documented in aggregate across the project, but the dedicated file is incomplete. The depth of per-experiment documentation is impressive.

---

### 9. Ablation Study Rigor (5 / 5)

**What was delivered:**
- **35+ single-variable experiments** across the vR.P.x lineage
- Strict one-change-per-experiment discipline maintained throughout
- **11 additional experiments** in the ETASR classification track (vR.x)
- Clear experimental genealogy: each version documents exactly what changed from its parent
- Controlled variables: same seed (42), same split ratios, same base architecture, same training config
- Both positive and negative results documented and analyzed
- Quantitative impact ranking with percentage-point deltas
- Reproducibility verified (P.3 and P.10 replicated)

**Impact hierarchy established:**
1. Input representation: +23.74pp (ELA)
2. Attention mechanisms: +3.57pp (CBAM)
3. Training configuration: +2.34pp (extended epochs)
4. Loss function: +0.03pp (Focal vs BCE — essentially neutral)

**Score rationale:** Full marks. This is the strongest aspect of the project. The ablation discipline is rigorous, well-documented, and produces actionable insights. The honest reporting of negative results (TTA, DCT standalone) adds significant scientific value.

---

### 10. Bonus Points (0 / 5)

**Bonus criteria from assignment:**
- Testing robustness against distortions (JPEG compression, resizing, cropping, noise)
- Successfully detecting subtle tampering (copy-move, splicing from similar textures)

**What was delivered:**
- No dedicated robustness testing against post-processing distortions
- No per-forgery-type analysis (splicing vs. copy-move broken down separately)
- The model handles both splicing and copy-move implicitly (CASIA v2.0 contains both), but no separate evaluation is provided

**Score rationale:** Neither bonus criterion was explicitly addressed. The project's depth lies in architectural exploration rather than robustness testing.

---

## Score Summary

| # | Dimension | Points Available | Score | Notes |
|---|-----------|-----------------|-------|-------|
| 1 | Dataset Selection & Preparation | 15 | **12** | Strong pipeline, single dataset |
| 2 | Model Architecture & Learning | 15 | **13** | Thoughtful choices, good ablation |
| 3 | Testing & Evaluation Metrics | 15 | **11** | Comprehensive metrics, no robustness testing |
| 4 | Visual Results | 10 | **10** | All 4 panels + extras |
| 5 | Single Notebook Deliverable | 10 | **5** | 232 notebooks, no unified submission |
| 6 | Assets (Weights, Links, Scripts) | 10 | **5** | Only 1 model weight, no Colab link |
| 7 | Research Report Quality | 10 | **9** | Excellent academic-style report |
| 8 | Research Journey Documentation | 5 | **4** | Rich but fragmented; main file incomplete |
| 9 | Ablation Study Rigor | 5 | **5** | Project's strongest dimension |
| 10 | Bonus Points | 5 | **0** | Not attempted |
| | **TOTAL** | **100** | **74** | |

---

## Key Strengths

1. **Exceptional ablation discipline.** 35+ single-variable experiments with rigorous control. Clear causal attribution of each modification's impact. This is graduate-level experimental methodology applied to an internship assignment.

2. **Honest and transparent reporting.** Negative results (TTA -5.32pp, DCT standalone -37.11pp) are documented just as thoroughly as positive ones. Failed lineages (v0x, vK.x, vR.x) are analyzed for lessons rather than hidden.

3. **Comprehensive experiment tracking.** Weights & Biases integration across 84 notebooks, centralized leaderboard, per-epoch metric logging, and prediction visualization logging. This is production-quality MLOps infrastructure.

4. **Deep domain understanding.** The evolution from naive RGB input → ELA preprocessing → Multi-Quality ELA demonstrates genuine learning about forensic image analysis, not just hyperparameter tuning.

5. **Massive documentation depth.** 1,123 markdown files including per-experiment documentation, audit reports, architecture analysis, and implementation plans. Every decision is traceable.

---

## Key Weaknesses

1. **No unified final deliverable.** The assignment explicitly requires "a single Google Colab Notebook." Despite 232 notebooks, no single polished submission notebook exists that combines the best findings.

2. **Model weights not preserved.** Only 1 `.pth` file found in the repository. For 35+ experiments, the checkpoints from the best-performing runs (P.15, P.10) are not available. Kaggle ephemeral storage means these are likely lost.

3. **No robustness testing.** The assignment offers bonus points for testing against JPEG compression, resizing, cropping, and noise. This was not attempted despite being a natural extension of the ELA-based approach.

4. **Incomplete research journey file.** `Research_Journey_and_Experimentation.md` is 14 lines and ends with a placeholder. While the journey is documented elsewhere, this primary file would be the first thing a reviewer reads.

5. **Quantity over polish.** The sheer scale (232 notebooks, 1,123 docs) suggests effort was spread across exploration rather than concentrated on polishing the final deliverable. An evaluator may be overwhelmed rather than impressed.

---

## Recommended Improvements (Priority Order)

### Critical (Would raise score by 10-15 points)

1. **Create a single submission notebook.** Combine the best pipeline (Multi-Q ELA + UNet + ResNet-34 + CBAM from P.15/P.10) into one clean, well-commented Google Colab notebook. Include all sections: dataset description, architecture explanation, training, evaluation, and visualization. Target: turn 232 notebooks into 1 polished deliverable.

2. **Save and provide model weights.** Re-run the best experiment (P.15 or P.10) and explicitly download and commit the `best_model.pth` to the repository. Provide a link or instruction for loading it.

3. **Complete the research journey file.** Expand `Research_Journey_and_Experimentation.md` from 14 lines to a proper narrative (the raw material already exists in `Submission_Report.md` Section 4). Cover all 4 lineages with dates, lessons, and transition rationale.

### Important (Would raise score by 5-10 points)

4. **Add robustness testing.** Create a evaluation cell that applies JPEG recompression (Q=50, 70, 90), Gaussian noise, resizing, and center-cropping to test images, then evaluates the model's performance under each distortion. This directly addresses the bonus criteria.

5. **Add per-forgery-type evaluation.** CASIA v2.0 filenames encode the tampering type (Sp = splicing, Cm = copy-move). Add a breakdown showing Pixel F1 separately for splicing and copy-move to address the second bonus criterion.

6. **Provide a Colab-compatible notebook.** Convert the Kaggle notebook to Google Colab by adjusting paths (`/kaggle/input/` → Google Drive mount) and ensuring all dependencies install correctly.

### Nice to Have

7. **Add cross-dataset evaluation.** Download the Columbia or Coverage dataset and evaluate the trained model without retraining. This tests generalization.

8. **Combine best components.** The vR.P.30 series (Multi-Q ELA + CBAM + extended training) is in progress — completing this and reporting results would strengthen the final deliverable.

9. **Add a requirements.txt.** Document the exact package versions used across all experiments for full reproducibility.

---

## Conclusion

This project demonstrates strong research skills, intellectual honesty, and impressive experimental discipline. The candidate learned from failures, maintained rigorous ablation methodology, and built production-quality experiment tracking infrastructure. The Pixel F1 of 0.7329 is a competent result for CASIA v2.0 tamper localization.

The primary gap is in **deliverable packaging**: the assignment asks for a clean, single-notebook submission with model weights, and this project instead delivers a comprehensive research exploration. The depth of work far exceeds typical internship expectations, but the final presentation needs consolidation. Addressing the "Critical" recommendations above would likely raise the score to 85-90/100.
