# 01 — Assignment Alignment Review

## Purpose

Re-evaluate the project against the original assignment requirements as stated in `Assignment.md`. Identify where the project remains aligned, where it has drifted, and what v9 must correct.

This review uses the **actual assignment text** as the ground truth, not any project-internal reinterpretation of it.

---

## The Assignment Requirements (Verbatim)

From `Assignment.md`:

> **Objective:** Develop a deep learning model to detect and localize tampered (edited or manipulated) regions in images. The model should not only classify whether an image is tampered, but also generate a pixel-level mask highlighting altered regions. We are looking for strong problem-solving skills, thoughtful architecture choices, and rigorous evaluation methodologies.

This explicitly requires:
1. **Image-level detection** — classify whether an image is tampered
2. **Pixel-level localization** — generate a mask of altered regions
3. **Thoughtful architecture choices** — not just "it works"
4. **Rigorous evaluation** — not inflated or misleading metrics

---

## Requirement-by-Requirement Assessment

### 1. Dataset Selection & Preparation

**Assignment says:**
> Use one or more publicly available datasets containing original (authentic) images, tampered images, and ground truth masks for localization. Examples include the CASIA Image Tampering Dataset, Coverage Dataset, CoMoFoD Dataset, or relevant Kaggle datasets.

**Current status:** CASIA v2.0 via Kaggle, 12,614 pairs.

**Alignment assessment:** ✅ Met. CASIA is explicitly listed as an example.

**Docs9 correction:** CASIA is a **chosen** baseline, not an assignment mandate. The assignment lists it as one of several options. Previous documentation versions overstated this as "the assignment's expected dataset." Docs9 corrects this framing.

**v9 action:** No dataset change. Add a clear statement in the notebook: "We selected CASIA v2.0 as our primary benchmark because it provides paired authentic/tampered images with ground truth masks, is publicly available on Kaggle, and enables direct comparison with prior work. The assignment also permits Coverage, CoMoFoD, and other datasets."

---

### 2. Data Pipeline

**Assignment says:**
> You are responsible for all dataset cleaning, preprocessing, and ensuring mask alignment. Properly split your data into train, validation, and test sets.

**Current status:**
- ✅ Mask binarization and alignment verified
- ✅ Stratified 70/15/15 split by forgery type
- ✅ 0 path overlaps across splits
- ⚠️ No content-based near-duplicate check
- ⚠️ No mask quality audit

**v9 action:**
- **Approved:** Run pHash near-duplicate check across splits. If duplicates found, group them into the same split.
- **Approved:** Document mask quality limitations as a known caveat rather than running a full manual review (impractical for scope).

---

### 3. Augmentation

**Assignment says:**
> Apply relevant data augmentation techniques to ensure model robustness.

**Current status (v8):**
- ✅ HFlip, VFlip, RandomRotate90 (geometric)
- ✅ ColorJitter, ImageCompression, GaussNoise, GaussianBlur (photometric, added in v8)

**Alignment assessment:** ✅ Met as of v8. The augmentation pipeline is now substantive.

**v9 action:** Maintain current augmentation. Add augmentation ablation experiment to validate each component's contribution. No further augmentation additions needed.

---

### 4. Model Architecture

**Assignment says:**
> Train a model to predict tampered regions. The choice of architecture and loss functions is entirely up to you.

**Current status:** U-Net with ResNet34 encoder. BCE + Dice loss with pos_weight and per-sample Dice.

**Alignment assessment:** ⚠️ Partial. The architecture works but the assignment asks for "thoughtful architecture choices." The current justification is stronger than Docs7 (baseline stability argument) but still lacks any comparison evidence.

**v9 actions:**
- **Approved:** Run one DeepLabV3+ comparison experiment to provide concrete evidence for the architecture choice.
- **Approved:** Add a learned image-level classification head to replace the heuristic `max(prob_map)` detector. This directly addresses the assignment's dual-task requirement ("not only classify... but also generate a pixel-level mask").
- **Approved:** Test ELA as an auxiliary input channel (4-channel RGB+ELA). This provides forensic signal beyond RGB and is lightweight enough for Colab.

---

### 5. Resource Constraints

**Assignment says:**
> Optimize for performance while keeping the solution runnable on Google Colab (T4 GPU compatible) or similar cloud platform.

**Current status:**
- ✅ Model fits on single T4 GPU (~24.4M params at 384×384)
- ✅ Kaggle variant tested on 2×T4
- ⚠️ Colab variant created but not verified end-to-end

**v9 action:**
- **Approved:** Verify Colab notebook runs end-to-end on Colab T4 before claiming compliance.
- All v9 improvements must be validated against T4 memory constraints before approval.

---

### 6. Performance Metrics

**Assignment says:**
> Thoroughly evaluate your model's localization performance and image-level detection accuracy using standard, industry-accepted metrics.

**Current status (v8):**
- ✅ Pixel-F1, Pixel-IoU for localization
- ✅ Image-level accuracy and AUC-ROC
- ✅ Per-forgery-type breakdown (splicing, copy-move)
- ✅ Tampered-only metrics reported as primary
- ⚠️ No boundary metrics
- ⚠️ Image-level detection is heuristic, not learned

**v9 actions:**
- **Approved:** Add Boundary F1 metric for localization quality at tampered edges.
- **Approved:** Replace heuristic detection with learned classification head (see §4).
- **Approved:** Add precision-recall curves for both pixel and image tasks.

---

### 7. Visual Results

**Assignment says:**
> Provide clear visual results comparing the Original Image, Ground Truth, Predicted output, and an Overlay Visualization.

**Current status:** ✅ Met. 4-panel visualizations with best/typical/worst examples, Grad-CAM heatmaps, failure case analysis.

**v9 action:** Maintain. Add ELA channel visualization for images where ELA input is used.

---

### 8. Deliverables

**Assignment says:**
> The entire implementation must be done in a single Google Colab Notebook.

**Current status:** ⚠️ Partial. Both Kaggle and Colab variants exist but Colab has not been verified end-to-end.

**v9 action:**
- **Critical:** The Colab notebook is the primary deliverable. v9 must verify it runs end-to-end.
- The Kaggle notebook is a development convenience, not the submission artifact.

---

### 9. Bonus: Robustness

**Assignment says:**
> Testing robustness against distortions such as JPEG compression, resizing, cropping, and noise.

**Current status (v8):** ✅ Met. Robustness suite tests JPEG (QF50, QF70), Gaussian noise, blur, and resize conditions.

**v9 action:** Maintain. Verify that v9 augmentation improvements reduce the robustness gap from Run01's −0.13.

---

### 10. Bonus: Subtle Tampering

**Assignment says:**
> Successfully detecting subtle tampering such as copy-move manipulation or splicing from similar textures.

**Current status:** ❌ Weak. Copy-move F1=0.31 in Run01. This is the assignment's explicit bonus area and the project's weakest point.

**v9 actions:**
- **Approved:** ELA auxiliary channel should improve copy-move detection by providing forensic signal invisible in RGB.
- **Approved:** Add per-forgery-type loss tracking to monitor copy-move convergence separately.
- Report copy-move F1 prominently and honestly. If it remains weak after v9, document why rather than hiding it.

---

## Overall Compliance Assessment

| Category | Run01 | v8 Status | v9 Target |
|---|---|---|---|
| Dataset & Pipeline | 8/10 | 8/10 | 9/10 (add pHash check) |
| Architecture & Learning | 6/10 | 7/10 | 8/10 (dual-task head, ELA, DeepLabV3+ comparison) |
| Testing & Evaluation | 7/10 | 8/10 | 9/10 (boundary F1, PR curves, learned detection) |
| Deliverables | 7/10 | 7/10 | 9/10 (verified Colab notebook) |
| Bonus: Robustness | 5/10 | 7/10 | 8/10 (tighter gap) |
| Bonus: Subtle Tampering | 3/10 | 4/10 | 6/10 (ELA + per-type tracking) |

**Estimated overall after v9: 8.2/10** — Solid assignment submission with honest limitations documented.

---

## Key Deviations to Correct

1. **Dataset framing.** Stop calling CASIA "the expected dataset." It is a chosen baseline.
2. **Detection component.** The assignment explicitly requires image-level classification. A heuristic is not enough. v9 must add a learned head.
3. **Colab deliverable.** This is non-negotiable. The primary artifact must be a verified Colab notebook.
4. **Architecture justification.** "Stable baseline" is acceptable but must be supported by at least one comparison experiment.
5. **Copy-move honesty.** Report the weakness clearly. Improvement is targeted but not guaranteed.
