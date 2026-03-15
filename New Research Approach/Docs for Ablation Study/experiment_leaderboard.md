# Experiment Leaderboard -- ETASR Ablation Study

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Ranked results and scoring for all ETASR ablation runs |
| **Paper** | ETASR_9593 -- "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Versions Covered** | ETASR: vR.1.0--vR.1.7 (8 runs) / Pretrained: vR.P.0--vR.P.18 (19 runs, 3 pending) / Standalone: 4 runs |

---

## 1. Overall Leaderboard

### Ranked by Test Accuracy

| Rank | Version | Change | Test Acc | Macro F1 | ROC-AUC | Verdict |
|------|---------|--------|----------|----------|---------|---------|
| 1 | **vR.1.6** | Deeper CNN | **90.23%** | **0.9004** | **0.9657** | POSITIVE |
| 2 | vR.1.3 | Class weights | 89.17% | 0.8889 | 0.9580 | POSITIVE |
| 2 | vR.1.7 | GAP | 89.17% | 0.8901 | 0.9495 | NEUTRAL |
| 4 | vR.1.5 | LR Scheduler | 88.96% | 0.8873 | 0.9560 | NEUTRAL |
| 5 | vR.1.4 | BatchNormalization | 88.75% | 0.8852 | 0.9536 | NEUTRAL |
| 6 | vR.1.1 | Eval fix (baseline) | 88.38% | 0.8805 | 0.9601 | Baseline |
| 7 | vR.1.2 | Data augmentation | 85.53% | 0.8505 | 0.9011 | REJECTED |
| -- | vR.1.0* | Paper reproduction | 89.89%* | 0.8972* | -- | *Val only* |

*vR.1.0 excluded from rankings (val-set metrics, no test split).*

### Ranked by Macro F1

| Rank | Version | Macro F1 | Test Acc |
|------|---------|----------|----------|
| 1 | **vR.1.6** | **0.9004** | 90.23% |
| 2 | vR.1.7 | 0.8901 | 89.17% |
| 3 | vR.1.3 | 0.8889 | 89.17% |
| 4 | vR.1.5 | 0.8873 | 88.96% |
| 5 | vR.1.4 | 0.8852 | 88.75% |
| 6 | vR.1.1 | 0.8805 | 88.38% |
| 7 | vR.1.2 | 0.8505 | 85.53% |

### Ranked by ROC-AUC

| Rank | Version | ROC-AUC | Test Acc |
|------|---------|---------|----------|
| 1 | **vR.1.6** | **0.9657** | 90.23% |
| 2 | vR.1.1 | 0.9601 | 88.38% |
| 3 | vR.1.3 | 0.9580 | 89.17% |
| 4 | vR.1.5 | 0.9560 | 88.96% |
| 5 | vR.1.4 | 0.9536 | 88.75% |
| 6 | vR.1.7 | 0.9495 | 89.17% |
| 7 | vR.1.2 | 0.9011 | 85.53% |

**Notable:** vR.1.1 (honest baseline) ranks **2nd** in ROC-AUC, ahead of all training-trick versions. Only vR.1.6's architectural change improved threshold-independent discriminatory power.

---

## 2. Per-Metric Champions

| Metric | Champion | Value | Why It Won |
|--------|----------|-------|------------|
| Test Accuracy | vR.1.6 | 90.23% | Deeper features + reduced dense bottleneck |
| Au Precision | vR.1.7 | 0.9590 | GAP makes model conservative on Au predictions |
| Au Recall | vR.1.6 | 0.8746 | Better feature extraction catches more authentic images |
| Au F1 | vR.1.6 | 0.9140 | Best balance of Au precision and recall |
| Tp Precision | vR.1.3 | 0.8431 | Class weights before BN shifted precision-recall balance |
| Tp Recall | vR.1.7 | 0.9467 | GAP forces aggressive tampered detection |
| Tp F1 | vR.1.6 | 0.8869 | Best balance of Tp precision and recall |
| Macro F1 | vR.1.6 | 0.9004 | Only version to cross 0.90 |
| ROC-AUC | vR.1.6 | 0.9657 | Only version to improve AUC from baseline |
| Param Efficiency | vR.1.7 | 64K params | 99.8% reduction from vR.1.0 |
| Lowest FN Rate | vR.1.7 | 5.3% | Misses fewest tampered images |
| Lowest FP Rate | vR.1.6 | 12.5% | Fewest false tampering accusations |

---

## 3. Parameter Efficiency Ranking

| Rank | Version | Test Acc | Params | Acc/M Params |
|------|---------|----------|--------|-------------|
| 1 | **vR.1.7** | 89.17% | 63,970 | **1,394.0** |
| 2 | vR.1.6 | 90.23% | 13,826,530 | 6.5 |
| 3 | vR.1.3 | 89.17% | 29,520,034 | 3.0 |
| 4 | vR.1.5 | 88.96% | 29,520,290 | 3.0 |
| 5 | vR.1.4 | 88.75% | 29,520,290 | 3.0 |
| 6 | vR.1.1 | 88.38% | 29,520,034 | 3.0 |
| 7 | vR.1.2 | 85.53% | 29,520,034 | 2.9 |

vR.1.7 is **214x more parameter-efficient** than vR.1.6.

---

## 4. Comprehensive Score Breakdown (/100)

### Scoring Rubric

| Category | Max | What It Measures |
|----------|-----|------------------|
| Architecture Implementation | /20 | Correct implementation, parameter efficiency, justified design |
| Dataset Handling | /15 | Proper split, stratification, class balance, ELA pipeline |
| Experimental Methodology | /20 | Single-variable fidelity, reproducibility, comparison rigor |
| Evaluation Quality | /20 | Per-class metrics, ROC-AUC, confusion matrix, visualizations |
| Documentation Quality | /15 | Version notes, architecture docs, change rationale |
| Assignment Alignment | /10 | Detection, localization, visual results, single-notebook |

### Score Table

| Version | Arch /20 | Data /15 | Method /20 | Eval /20 | Docs /15 | Assign /10 | **Total** |
|---------|----------|----------|------------|----------|----------|------------|-----------|
| vR.1.0 | 14 | 8 | 10 | 8 | 6 | 3 | **49** |
| vR.1.1 | 14 | 13 | 18 | 18 | 12 | 5 | **80** |
| vR.1.2 | 14 | 13 | 17 | 18 | 12 | 5 | **79** |
| vR.1.3 | 14 | 15 | 19 | 19 | 13 | 5 | **85** |
| vR.1.4 | 15 | 15 | 19 | 19 | 14 | 5 | **87** |
| vR.1.5 | 15 | 15 | 19 | 19 | 14 | 5 | **87** |
| vR.1.6 | 17 | 15 | 20 | 19 | 14 | 5 | **90** |
| vR.1.7 | 18 | 15 | 20 | 19 | 14 | 5 | **91** |

### Scoring Rationale

**vR.1.0 (49/100):** Low scores in Dataset (no test split, val-only metrics), Methodology (no single-variable framework), Evaluation (no per-class metrics, no AUC, no CM), Documentation (minimal), Assignment (no test eval, no localization). The paper reproduction itself is faithful (Architecture 14/20).

**vR.1.1 (80/100):** Massive jump from vR.1.0. Introduces proper test split, per-class metrics, ROC-AUC, confusion matrix, model saving. The most important single version in the series -- establishes honest evaluation. Still limited by no class balance handling (Data 13/15) and no localization (Assign 5/10).

**vR.1.2 (79/100):** Scores nearly as high as vR.1.1 despite REJECTED verdict. The score measures experimental quality, not result quality. The augmentation experiment was well-designed, properly evaluated, and correctly rejected. Methodology 17/20 (slight deduction for using ImageDataGenerator which changes effective batch size).

**vR.1.3 (85/100):** Class weights properly address the 1.46:1 class imbalance. Full methodology maturity. Dataset handling 15/15 (all pipeline aspects correct).

**vR.1.4 (87/100):** Architecture improves to 15/20 (BN is a reasonable addition). Full marks retained despite NEUTRAL result.

**vR.1.5 (87/100):** Same scores as vR.1.4. LR scheduler is infrastructure rather than architectural improvement.

**vR.1.6 (90/100):** Architecture jumps to 17/20 (53% param reduction, deeper features, correct design). Methodology reaches 20/20 (perfect single-variable execution with best POSITIVE result). Assignment still 5/10 (no localization).

**vR.1.7 (91/100):** Architecture peaks at 18/20 (99.5% param reduction, parameter-efficient design, GAP is standard modern practice). Despite lower accuracy than vR.1.6, the experimental quality and architectural significance earn the highest total score. Assignment still 5/10.

**Key observation:** Assignment Alignment is capped at 5/10 for ALL ETASR track runs because none produce localization masks. The pretrained track (vR.P.x) addresses this gap.

---

## 5. Verdict Distribution

| Verdict | Count | Versions | Net Impact |
|---------|-------|----------|------------|
| **POSITIVE** | 2 | vR.1.3, vR.1.6 | +0.79pp, +1.27pp |
| **NEUTRAL** | 3 | vR.1.4, vR.1.5, vR.1.7 | -0.42pp, +0.21pp, -1.06pp |
| **REJECTED** | 1 | vR.1.2 | -2.85pp |
| **Baseline** | 2 | vR.1.0, vR.1.1 | -- |

### Change Category Impact

| Category | Versions | Best Impact | Assessment |
|----------|----------|-------------|------------|
| Evaluation fix | vR.1.1 | -- | Critical (honest baseline) |
| Data pipeline | vR.1.2 | -2.85pp | Harmful for this architecture |
| Training config | vR.1.3, 1.4, 1.5 | +0.79pp cumulative | Marginal (only +0.58pp net) |
| Architecture | vR.1.6, 1.7 | +1.27pp from parent | Most impactful category |

---

## 6. Assignment Deliverable Alignment

| Deliverable | Best ETASR Version | Status | Notes |
|-------------|-------------------|--------|-------|
| Dataset with GT masks | All (vR.1.1+) | Partial | CASIA v2.0 used; GT masks exist but not used in ETASR track |
| Dataset preprocessing | All (vR.1.1+) | Met | ELA pipeline, cleaning, splitting |
| Train/val/test split | All (vR.1.1+) | Met | 70/15/15 stratified |
| Data augmentation | vR.1.2 | Missing | Attempted and rejected |
| Tampered region prediction | None | **Missing** | Classification only; no pixel-level output |
| Standard evaluation metrics | All (vR.1.1+) | Met | Accuracy, P/R/F1, AUC, CM |
| Visual results (Orig/GT/Pred/Overlay) | None | **Missing** | No localization = no overlay |
| Single notebook | Each version | Partial | Multiple notebooks; needs consolidation |
| Model weights saved | All (vR.1.1+) | Met | .keras format |
| Robustness testing (bonus) | None | Missing | Not attempted |

**Critical gap:** The ETASR track cannot satisfy the assignment's core requirement (pixel-level tampered region prediction). This is addressed by the pretrained localization track (vR.P.x).

---

## 7. Final Rankings Summary

### By Result Quality (what the model achieves)

| Rank | Version | Test Acc | Verdict |
|------|---------|----------|---------|
| 1 | vR.1.6 | 90.23% | POSITIVE |
| 2 | vR.1.7 | 89.17% | NEUTRAL |
| 2 | vR.1.3 | 89.17% | POSITIVE |

### By Experimental Quality (how well the experiment was conducted)

| Rank | Version | Score /100 | Key Distinction |
|------|---------|------------|-----------------|
| 1 | vR.1.7 | 91 | Best architecture design + methodology |
| 2 | vR.1.6 | 90 | Best results + methodology |
| 3 | vR.1.4/1.5 | 87 | Mature methodology, neutral results |
| 4 | vR.1.3 | 85 | First POSITIVE, full pipeline |
| 5 | vR.1.1 | 80 | Foundational (honest baseline) |
| 6 | vR.1.2 | 79 | Well-executed failure |
| 7 | vR.1.0 | 49 | Incomplete evaluation |

---

## 8. Pretrained Localization Track Leaderboard

### Ranked by Pixel F1

| Rank | Version | Change | Pixel F1 | IoU | Pixel AUC | Img Acc | Verdict |
|------|---------|--------|----------|-----|-----------|---------|---------|
| 1 | **vR.P.10** | **CBAM attention** | **0.7277** | **0.5719** | **0.9573** | 87.32% | **POSITIVE** |
| 2 | **vR.P.7** | Extended training | 0.7154 | 0.5569 | 0.9504 | 87.37% | **POSITIVE** |
| 3 | vR.P.4 | 4ch RGB+ELA | 0.7053 | 0.5447 | 0.9433 | 84.42% | NEUTRAL |
| 4 | vR.P.8 | Progressive unfreeze | 0.6985 | 0.5367 | 0.9541 | **87.59%** | NEUTRAL |
| 5 | **vR.P.12** | **Augmentation + Focal+Dice** | **0.6968** | **0.5347** | 0.9502 | **88.48%** | **NEUTRAL (NEW)** |
| 6 | vR.P.9 | Focal+Dice loss | 0.6923 | 0.5294 | 0.9323 | 87.16% | NEUTRAL |
| 7 | **vR.P.3** | ELA input | 0.6920 | 0.5291 | 0.9528 | 86.79% | STRONG POSITIVE |
| 8 | **vR.P.14** | **TTA (4 views)** | **0.6388*** | **0.4693*** | **0.9618*** | --** | **NEGATIVE (NEW)** |
| 9 | vR.P.6 | EfficientNet-B0 | 0.5217 | 0.3529 | 0.8708 | 70.68% | POSITIVE |
| 10 | vR.P.5 | ResNet-50 | 0.5137 | 0.3456 | 0.8828 | 72.00% | POSITIVE |
| 11 | vR.P.2 | Gradual unfreeze | 0.5117 | 0.3439 | 0.8688 | 69.04% | POSITIVE |
| 12 | vR.P.1 | Dataset fix (baseline) | 0.4546 | 0.2942 | 0.8509 | 70.15% | Baseline |
| 13 | vR.P.1.5 | Speed opts | 0.4227 | 0.2680 | 0.8560 | 71.05% | NEUTRAL |
| 14 | vR.P.0 | Initial (no GT masks) | 0.3749 | 0.2307 | 0.8486 | 70.63% | Baseline (no GT) |

*P.14 TTA metrics. Without TTA: Pixel F1=0.6919 (identical to P.3). **Image-level metrics not available (code bug).*

### Ranked by Image Accuracy

| Rank | Version | Input | Img Acc | Pixel F1 |
|------|---------|-------|---------|----------|
| 1 | **vR.P.12** | ELA | **88.48%** | 0.6968 |
| 2 | **vR.P.8** | ELA | 87.59% | 0.6985 |
| 3 | **vR.P.7** | ELA | 87.37% | 0.7154 |
| 4 | **vR.P.10** | ELA | 87.32% | 0.7277 |
| 5 | vR.P.9 | ELA | 87.16% | 0.6923 |
| 6 | **vR.P.3** | ELA | 86.79% | 0.6920 |
| 7 | vR.P.4 | RGB+ELA | 84.42% | 0.7053 |
| 8 | vR.P.5 | RGB | 72.00% | 0.5137 |
| 9 | vR.P.1.5 | RGB | 71.05% | 0.4227 |
| 10 | vR.P.6 | RGB | 70.68% | 0.5217 |
| 11 | vR.P.0 | RGB | 70.63% | 0.3749 |
| 12 | vR.P.1 | RGB | 70.15% | 0.4546 |
| 13 | vR.P.2 | RGB | 69.04% | 0.5117 |

**Key insight:** ELA-based inputs dominate all top spots (84-88%). P.12's augmentation leads Image Accuracy while P.10's CBAM attention leads Pixel F1.

---

## 9. Pretrained Track Score Breakdown (/100)

### Scoring Rubric (Pretrained Track)

| Category | Max | What It Measures |
|----------|-----|------------------|
| Architecture | /15 | Encoder choice, freeze strategy, input design |
| Dataset | /15 | GT masks, normalization, preprocessing |
| Methodology | /20 | Single-variable fidelity, execution quality, model saving |
| Evaluation | /20 | Pixel + image metrics, visualizations, per-image analysis |
| Documentation | /15 | Architecture diagrams, inline docs, change rationale |
| Assignment Alignment | /15 | Localization masks, model weights, visual results |

### Score Table

| Version | Arch /15 | Data /15 | Method /20 | Eval /20 | Docs /15 | Assign /15 | **Total** |
|---------|----------|----------|------------|----------|----------|------------|-----------|
| vR.P.0 | 10 | 8 | 12 | 14 | 8 | 8 | **60** |
| vR.P.1 | 11 | 13 | 15 | 16 | 10 | 10 | **75** |
| vR.P.1.5 | 11 | 13 | 16 | 16 | 11 | 10 | **77** |
| vR.P.2 | 12 | 13 | 13 | 15 | 10 | 8 | **71** |
| **vR.P.3** | **13** | **14** | **14** | **16** | **11** | **10** | **78** |
| **vR.P.4** | **14** | **14** | **17** | **19** | **12** | **10** | **86** |
| **vR.P.5** | 12 | 14 | 14 | 17 | 10 | 10 | **77** |
| **vR.P.6** | 13 | 14 | 13 | 18 | 12 | 8 | **78** |
| vR.P.3 r02 | 13 | 14 | 15 | 18 | 10 | 12 | **82** |
| **vR.P.8** | **14** | **14** | **14** | **18** | **12** | **12** | **84** |
| vR.P.9 | 13 | 14 | 12 | 17 | 11 | 11 | **78** |
| **vR.P.7** | 13 | 14 | 17 | 19 | 12 | 13 | **88** |
| **vR.P.10** | **15** | 14 | 14 | 19 | 11 | 14 | **87** |
| vR.P.10 r02 | 15 | 14 | 16 | 19 | 10 | 14 | **88** |
| **vR.P.12** | **13** | **14** | **13** | **19** | **12** | **13** | **84** |
| **vR.P.14** | **12** | **14** | **10** | **10** | **10** | **8** | **64** |

### Scoring Notes

**vR.P.3 (78/100):** Best results in series but CRITICAL bug (model not saved) heavily penalises Methodology and Assignment Alignment. Score would be ~88 without the crash.

**vR.P.4 (86/100):** Highest score despite NEUTRAL verdict. Best execution quality: all cells pass, model saved, comprehensive evaluation. Score reflects experiment quality, not result significance.

**vR.P.5 (77/100):** Copy-paste bugs (resnet34 in filename and comparison table) penalise Documentation and Methodology.

**vR.P.6 (78/100):** Methodology inconsistency (no AMP/TF32 unlike P.5) penalises from potential ~84.

**vR.P.3 Run-02 (82/100):** Reproducibility re-run confirms P.3 metrics. Higher score than Run-01 due to clean execution (all cells pass, model saved). Penalised for inconclusive training (best=last epoch, model still improving) and display bug (results table shows "RGB" instead of "ELA").

**vR.P.8 (84/100):** Progressive unfreeze experiment with 3-stage training. Best pixel precision (0.8857) and image accuracy (87.59%) in pretrained track. Stage 0 (frozen) produced best metrics; Stage 1 (layer4 unfreeze) was counterproductive. Methodology penalised for Stage 1 regression not diagnosed.

**vR.P.9 (78/100):** Clean single-variable ablation (Focal+Dice replacing BCE+Dice). Pixel F1 essentially unchanged (+0.03pp) but Pixel AUC regressed -0.0205 and Image ROC-AUC regressed -0.0426. Methodology penalised for untuned Focal hyperparameters and AUC regression not investigated.

**vR.P.10 Run-02 (88/100):** Perfect reproducibility verification — every metric, epoch log, and LR schedule change identical to Run-01. Validates SEED=42 determinism across independent Kaggle sessions. Methodology 16/20 for confirming determinism, but only same-hardware reproducibility (both P100). Documentation penalised for changelog not updated for P.10.

**vR.P.12 (84/100):** Data augmentation experiment (Albumentations + Focal+Dice loss). Well-designed joint image+mask augmentation pipeline with 6 safe transforms. Pixel F1 gain marginal (+0.48pp from P.3). Methodology penalised for confounding (2 variables changed: augmentation AND loss). Training instability (val loss spikes at ep19/21) not investigated. Best image accuracy (88.48%) in pretrained track.

**vR.P.14 (64/100):** TTA experiment with 4 geometric views. Clean experimental design isolating TTA's effect. **TTA degraded Pixel F1 by -5.32pp** — averaging pushes borderline pixels below 0.5 threshold. Critical code bug in cell 18 (`NameError: test_probs not defined`) crashed 40% of evaluation. Image-level metrics, confusion matrix, visualizations, and model save all MISSING. Interesting negative result poorly executed.

---

## 10. Cross-Track Comparison

### Best ETASR vs Best Pretrained

| Metric | ETASR Best (vR.1.6) | Pretrained Best | Winner |
|--------|---------------------|-----------------|--------|
| Image Accuracy | **90.23%** | 88.48% (P.12) | ETASR |
| Image Macro F1 | **0.9004** | 0.8756 (P.12) | ETASR |
| Image ROC-AUC | 0.9657 | **0.9633** (P.10) | ETASR (barely) |
| Pixel F1 | N/A | **0.7277** (P.10) | Pretrained |
| Pixel IoU | N/A | **0.5719** (P.10) | Pretrained |
| Pixel AUC | N/A | **0.9573** (P.10) | Pretrained |
| Localization masks | Not available | **Available** | Pretrained |
| Assignment alignment | Partial | **Full** | Pretrained |

**Conclusion:** ETASR wins classification, pretrained wins localization and ROC-AUC is nearly tied. The pretrained track is required for assignment submission.

---

## 11. Standalone Research Paper Architecture Runs

These runs implement the original paper's CNN architecture (or a deeper variant) as standalone experiments outside the ablation framework. They provide classification baselines but **cannot satisfy the assignment's localization requirement**.

### Results Summary

| Run | Architecture | Dataset | Params | Test Acc | Macro F1 | Localization | Score |
|-----|-------------|---------|--------|----------|----------|--------------|-------|
| Paper-divg07 | 2×Conv32(5×5) + Dense(150) | divg07 (12,614) | 24.2M | 90.33% | 0.9006 | NO | 56/100 |
| Paper-sagnik | 2×Conv32(5×5) + Dense(150) | sagnik (12,614) | 24.2M | ~~100%~~ | ~~1.0000~~ | NO | 28/100 |
| Deeper-divg07 | 3×Conv(64→128→256) + BN + Dense(512) | divg07 (12,614) | 38.3M | **90.76%** | **0.9082** | NO | 66/100 |
| **Deeper-sagnik** | **3×Conv(64→128→256) + BN + Dense(512)** | **sagnik (LEAKED)** | **38.3M** | **~~99.95%~~** | ~~0.9995~~ | NO | **34/100** |

### Key Findings

1. **Paper accuracy not reproduced:** 90.33% vs claimed 94.14% (-3.81pp). Likely cause: paper specifies JPEG-only (9,501 images) while reproduction uses all formats (12,614 images).
2. **Sagnik dataset has a DATA LEAK:** 100% accuracy is scientifically invalid — X range [0.0, 0.76] suggests mask images were loaded as input instead of photographs.
3. **Deeper CNN achieves best classification:** 90.76% accuracy with 96.27% tampered recall, but early stopping and BatchNorm contributed more than depth alone (+0.43pp over paper architecture).
4. **Classification ≠ Localization:** The +3.17pp classification advantage of the deeper CNN over the best UNet (P.8: 87.59%) is irrelevant when the assignment requires pixel-level masks.

### Paper Claims vs Reproduction

| Metric | Paper Claim | Reproduction (divg07) | Gap |
|--------|------------|----------------------|-----|
| Train Accuracy | 99.05% | 98.57% | -0.48pp |
| Test Accuracy | 94.14% | 90.33% | -3.81pp |
| Precision | 94.1% | 90.31% | -3.79pp |
| Recall | 94.07% | 90.10% | -3.97pp |

---

## 12. Future Experiment Proposals

### Completed Experiments (from P.3 lineage)

| ID | Experiment | Result | Actual Impact | Verdict |
|----|-----------|--------|---------------|---------|
| ~~vR.P.7~~ | ELA + extended training (50ep) | Pixel F1: 0.7154, best epoch 36 | **+2.34pp from P.3** | **POSITIVE** |
| ~~vR.P.8~~ | ELA + progressive unfreeze | Pixel F1: 0.6985 | +0.65pp from P.3 | **NEUTRAL** |
| ~~vR.P.9~~ | Focal+Dice loss | Pixel F1: 0.6923 | +0.03pp from P.3 | **NEUTRAL** |
| ~~vR.P.10~~ | ELA + CBAM attention | Pixel F1: **0.7277** (SERIES BEST) | **+3.57pp from P.3** | **POSITIVE** |
| ~~vR.P.12~~ | ELA + data augmentation + Focal+Dice | Pixel F1: 0.6968 | +0.48pp from P.3 | **NEUTRAL** |
| ~~vR.P.14~~ | Test-Time Augmentation (TTA) | Pixel F1: 0.6388 (TTA) / 0.6919 (no-TTA) | **-5.32pp (TTA)** | **NEGATIVE** |

### Remaining / Future Experiments

| ID | Experiment | Rationale | Expected Impact |
|----|-----------|-----------|-----------------|
| vR.P.13 | CBAM + augmentation + 50ep (combined best) | Combine P.10's CBAM + P.12's augmentation + P.7's training budget | +1-3pp Pixel F1 (est. 0.74-0.76) |
| vR.P.15 | Multi-quality ELA (Q=75/85/95) | 3 grayscale ELA channels capture different artifact signatures | +2-5pp Pixel F1 (input experiment) |
| vR.P.16 | EfficientNet-B0 + ELA | Test more efficient encoder on ELA input | +1-3pp Pixel F1 |
| vR.P.17 | Higher resolution (512×512) | More spatial detail for fine-grained localization | +2-4pp Pixel F1, +memory cost |
| vR.P.18 | Threshold optimization + CRF post-processing | Recalibrate threshold from 0.5; CRF smoothing | +1-3pp Pixel F1 (free gain) |
