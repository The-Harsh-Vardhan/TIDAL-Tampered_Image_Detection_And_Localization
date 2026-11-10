# Docs8 — Project Evolution Summary

## Purpose

Docs8 is the **design blueprint for Notebook v8** — the next training pipeline iteration for the Tampered Image Detection and Localization project. It bridges three phases of development:

1. **Docs7** — the original system design documentation
2. **Audit7 Pro** — the critical review of design assumptions and gaps
3. **Run01** (`v6-5-tampered-image-detection-localization-run-01.ipynb`) — the first experimental training run

This document set does not repeat the original documentation. It documents what changed, what broke, what held, and what must be improved.

---

## Run01 Identity

| Property | Value |
|---|---|
| Notebook | `v6-5-tampered-image-detection-localization-run-01.ipynb` |
| Platform | Kaggle, 2× Tesla T4 (DataParallel) |
| Dataset | CASIA v2.0 Splicing + Localization |
| Model | SMP U-Net, ResNet34 encoder, ImageNet pretrained |
| Loss | BCE + Dice (smooth=1.0) |
| Optimizer | AdamW (encoder 1e-4, decoder 1e-3) |
| Image size | 384×384 |
| Effective batch | 16 (batch=4 × accum=4) |
| Training | 25/50 epochs (early stopped at patience=10) |
| Best epoch | 15 (val Pixel-F1 = 0.7289) |
| Best threshold | 0.1327 (post-training sweep) |

---

## Run01 Key Metrics

### Test Results (threshold = 0.1327)

| Metric | Mixed-set (1893) | Tampered-only (769) |
|---|---|---|
| Pixel-F1 | 0.7208 ± 0.4158 | **0.4101 ± 0.4148** |
| Pixel-IoU | 0.6989 ± 0.4194 | 0.3563 ± 0.3798 |
| Precision | 0.7455 | — |
| Recall | 0.7634 | — |

| Metric | Value |
|---|---|
| Image Accuracy | 0.8246 |
| Image AUC-ROC | 0.8703 |

### Forgery Breakdown

| Type | Count | F1 |
|---|---|---|
| Splicing | 274 | 0.5901 ± 0.3850 |
| Copy-move | 495 | 0.3105 ± 0.3968 |

### Robustness (Pixel-F1)

| Condition | F1 | Δ from clean |
|---|---|---|
| Clean | 0.7208 | — |
| JPEG QF70 | 0.5912 | −0.1296 |
| JPEG QF50 | 0.5938 | −0.1269 |
| Gaussian noise (light) | 0.5938 | −0.1270 |
| Gaussian noise (heavy) | 0.5938 | −0.1270 |
| Gaussian blur | 0.5881 | −0.1326 |
| Resize 0.75× | 0.6631 | −0.0576 |
| Resize 0.5× | 0.6134 | −0.1073 |

---

## Critical Findings

### What Docs7 Got Right

1. **Pipeline structure is sound.** The end-to-end flow (dataset → preprocessing → U-Net → training → evaluation → visualization → robustness) works correctly.
2. **Engineering practices are strong.** CONFIG-driven pipeline, feature flags, checkpoint resumption, artifact inventory, W&B integration.
3. **Evaluation framework is comprehensive.** Threshold sweep, mixed + tampered-only + per-forgery reporting, robustness suite, Grad-CAM, failure analysis.
4. **No bugs or data leakage detected.** Stratified splits verified, metrics computed correctly, checkpoint strategy robust.

### What Audit7 Pro Correctly Predicted

1. **Mixed-set metric inflation.** Audit warned that authentic images scoring F1=1.0 inflates aggregates. Run01 confirmed: mixed F1=0.72 vs tampered-only F1=0.41.
2. **Loss design gaps.** Audit flagged missing `pos_weight` and per-sample Dice. Run01 confirmed: optimal threshold of 0.1327 indicates severe probability suppression from unweighted BCE.
3. **Weak augmentation.** Audit questioned the minimalist augmentation philosophy. Run01 confirmed: severe overfitting after epoch 15, 13% F1 drop under JPEG compression.
4. **BatchNorm instability.** Audit flagged batch-size-4 with BatchNorm encoder. Run01 used DataParallel across 2 GPUs (2 images/GPU), worsening this concern.
5. **Image-level heuristic weakness.** Audit flagged heuristic detector. Run01's max-probability approach achieved AUC=0.87 — functional but not robust.
6. **RGB-only limitation.** Audit flagged forensic blindness. Robustness results show ~13% F1 from compression artifacts (destroyed by any degradation), consistent with partial shortcut learning.

### What Run01 Revealed Beyond Predictions

1. **Copy-move near-failure.** F1=0.31 on copy-move is worse than expected. Copy-move represents 64% of tampered images, dragging overall performance.
2. **Complete failures exist.** Worst 10 predictions have F1=0.0. 8/10 are copy-move, 6/10 have mask area <2%.
3. **No learning rate scheduler.** The constant LR caused the model to plateau at epoch 15 and overfit for 10 more epochs before stopping. Train loss decreased from 0.63 to 0.51 while val loss increased from 0.81 to 1.20.
4. **Robustness plateau.** Four degradation conditions (JPEG QF50, QF70, noise light, noise heavy) produce nearly identical F1 (~0.593), suggesting the model collapses to a baseline when input distribution shifts.
5. **High metric variance.** Std dev of 0.41 on Pixel-F1 means some images are predicted perfectly and others fail completely — model performance is highly inconsistent.
6. **cudnn.benchmark contradiction.** `set_seed()` sets `benchmark=False`, immediately overridden by `setup_device()` setting `benchmark=True`.

---

## Docs8 Document Index

| Document | Focus |
|---|---|
| [01_Assignment_Requirement_Alignment.md](01_Assignment_Requirement_Alignment.md) | Gap analysis against assignment requirements |
| [02_Dataset_Evolution.md](02_Dataset_Evolution.md) | Dataset handling improvements |
| [03_Model_Architecture_Evolution.md](03_Model_Architecture_Evolution.md) | Architecture assessment and proposed changes |
| [04_Training_Strategy_Evolution.md](04_Training_Strategy_Evolution.md) | Loss, optimizer, scheduler, augmentation evolution |
| [05_Evaluation_Methodology_Evolution.md](05_Evaluation_Methodology_Evolution.md) | Metric reporting and validation improvements |
| [06_Run01_Results_Analysis.md](06_Run01_Results_Analysis.md) | Detailed analysis of Run01 training outputs |
| [07_Shortcut_Learning_Risk_Assessment.md](07_Shortcut_Learning_Risk_Assessment.md) | Artifact dependency and mitigation |
| [08_Notebook_V8_Implementation_Plan.md](08_Notebook_V8_Implementation_Plan.md) | Concrete implementation checklist |
| [09_Future_Experiments.md](09_Future_Experiments.md) | Experiment roadmap beyond v8 |
| [10_References.md](10_References.md) | Literature and resource references |
| [11_Training_Failure_Cases.md](11_Training_Failure_Cases.md) | Failure taxonomy and mitigation plan |

---

## Expected Outcome for Notebook v8

If the P0 and P1 fixes from this document set are implemented:

| Metric | Run01 | Expected v8 |
|---|---|---|
| Tampered-only Pixel-F1 | 0.41 | 0.50–0.60 |
| Splicing F1 | 0.59 | 0.65–0.72 |
| Copy-move F1 | 0.31 | 0.40–0.50 |
| Optimal threshold | 0.13 | 0.30–0.50 |
| Robustness Δ (JPEG) | −0.13 | −0.05 to −0.08 |
| Overfitting onset | Epoch 15 | Epoch 30+ |

These are estimates, not guarantees. The improvements are grounded in the specific failure modes observed in Run01 and standard remedies from the literature.
