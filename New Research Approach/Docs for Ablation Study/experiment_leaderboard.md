# Experiment Leaderboard -- ETASR Ablation Study

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Ranked results and scoring for all ETASR ablation runs |
| **Paper** | ETASR_9593 -- "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Versions Covered** | vR.1.0 through vR.1.7 (8 runs) |

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
