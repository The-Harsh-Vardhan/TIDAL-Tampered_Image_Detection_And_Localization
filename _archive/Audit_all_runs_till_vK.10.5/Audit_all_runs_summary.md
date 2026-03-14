# Master Audit Summary: All Experiment Runs (vK.1 → vK.10.5)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**Scope:** Every experiment run in `Notebooks/Runs/` evaluated against the Big Vision Internship Assignment requirements

---

## Executive Summary

This project has gone through **11 notebook versions** across two parallel architecture tracks:

- **v-series (v6.5, v8):** Uses `smp.Unet` with a **pretrained ResNet34 ImageNet encoder**, 384×384 input, AdamW with differential LR, AMP, DataParallel, early stopping, and comprehensive evaluation suites. Best tampered-only F1: **0.4101** (v6.5).

- **vK.x series (vK.1–vK.3, vK.7.x, vK.10.x):** Uses a custom `UNetWithClassifier` trained **from scratch** with no pretrained weights, 256×256 input. Best tampered-only F1: **~0.20** (estimated, vK.3). Most recent runs (vK.10.3b–vK.10.5) collapsed to **near-zero** segmentation despite excellent engineering.

The most striking finding is that the project **had the right architecture in v6.5** — pretrained encoder, proper evaluation, sound engineering — and then the vK.10.x series **abandoned it**, reverting to training from scratch and losing all evaluation features (threshold optimization, Grad-CAM, robustness testing, forgery-type breakdown). This architectural regression from v6.5 to vK.10.x is the single most consequential decision across all runs.

---

## Run Timeline

| # | Version | Architecture Track | Key Change | Outcome |
|---|---|---|---|---|
| 1 | vK.1 | vK.x (from scratch) | Baseline code | No run output |
| 2 | vK.2 | vK.x (from scratch) | +Markdown docs, +W&B | No run output |
| 3 | vK.3 | vK.x (from scratch) | +English docstrings | No run output |
| 4 | vK.3-run-01 | vK.x (from scratch) | Actual Kaggle execution | Acc=89.9%, Dice=0.58 (inflated) |
| 5 | **v6.5** | **v-series (pretrained)** | **SMP ResNet34, 384×384, AMP, early stop, Grad-CAM, robustness** | **Best: Tam-F1=0.41, Acc=0.82** |
| 6 | **v8** | **v-series (pretrained)** | **+ReduceLROnPlateau, +augmentations, +shortcut det., pos_weight=30** | **Regression: Tam-F1=0.29** |
| 7 | vK.7.1 | vK.x (from scratch) | Documentation refresh | Acc=89.9%, same as vK.3 |
| 8 | vK.7.5 | vK.x (from scratch) | Execution attempt | **Incomplete — no metrics** |
| 9 | vK.10.3b | vK.x (from scratch) | +CONFIG, +AMP, +seeding, +early stop | **Collapse: Tam-F1=0.0004** |
| 10 | vK.10.4 | vK.x (from scratch) | +Data visualization | **Collapse: Tam-F1=0.0000** |
| 11 | vK.10.5 | vK.x (from scratch) | +DataParallel | **Collapse: Tam-F1=0.0006** |

---

## Assignment Requirements Coverage

### 1. Dataset Selection & Preparation

| Requirement | Status | Notes |
|---|---|---|
| Publicly available dataset | **Met** | CASIA-2 from Kaggle |
| Dataset cleaning/preprocessing | **Partial** | Metadata caching in v6.5/v8 and vK.10.x |
| Mask alignment | **Met** | Masks properly loaded and aligned |
| Train/Val/Test split | **Met** | 70/15/15 stratified |
| Data augmentation | **Met** | v6.5: geometric; v8: geometric + color + noise + compression |
| Data leakage check | **Partial** | v6.5/v8 have explicit path-level verification (PASSED); vK.1–vK.7.x have Block 1 bug; vK.10.x fixed but no explicit check |

### 2. Model Architecture & Learning

| Requirement | Status | Notes |
|---|---|---|
| Train a model for tampered regions | **Met** | v6.5 achieves Tam-F1=0.41; vK.10.x achieves ~0 |
| Architecture choice justified | **Partial** | v6.5/v8 use pretrained ResNet34 (strong choice); vK.10.x trains from scratch (unjustified) |
| T4 GPU compatible | **Met** | All runs on Kaggle T4 |
| Performance optimization | **Partial** | v6.5/v8 use pretrained encoder + AMP + DataParallel; vK.10.x has AMP + DataParallel but no pretrained encoder |

### 3. Testing & Evaluation

| Requirement | Status | Notes |
|---|---|---|
| Localization performance metrics | **Met** | v6.5/v8 report tampered-only Dice/F1/IoU; vK.10.x reports but values are ~0 |
| Image-level detection accuracy | **Met** | All runs with output report accuracy |
| Standard industry metrics | **Partial** | v6.5/v8 have AUC-ROC, threshold optimization; PR curves missing from all runs |
| Visual results (4-panel) | **Met** | Prediction grids present in v6.5/v8 and vK.2+ |

### 4. Deliverables & Documentation

| Requirement | Status | Notes |
|---|---|---|
| Single Colab notebook | **Met** | All code in one notebook |
| Dataset explanation | **Met** | In v6.5/v8 and vK.2+ |
| Model architecture description | **Met** | Documented in all versions |
| Training strategy | **Met** | CONFIG dict in v6.5/v8/vK.10.x |
| Evaluation results | **Partial** | v6.5 has meaningful results; vK.10.x results are near-zero |
| Clear visualizations | **Met** | Prediction grids, training curves, Grad-CAM (v6.5/v8) |

### Bonus Points

| Requirement | Status | Notes |
|---|---|---|
| Robustness testing (JPEG, noise, etc.) | **Partial** | v6.5/v8 have 8-condition robustness suite; v6.5 results suspicious (identical F1); v8 shows JPEG gap fixed |
| Subtle tampering detection | **Partial** | v6.5/v8 have forgery-type breakdown (splicing vs copy-move); v8 has mask-size stratification |

---

## The Five Fatal Flaws

### 1. Architectural Regression: vK.10.x Abandoned the Pretrained Encoder

v6.5 demonstrated that `smp.Unet(encoder_name='resnet34', encoder_weights='imagenet')` produces Tam-F1=0.41 — 680× better than vK.10.5's 0.0006. The vK.10.x series reverted to a custom `UNetWithClassifier` trained from scratch with 31.6M parameters on 8,829 images. This is the project's most consequential regression. The **vK.x series** (vK.1–vK.3, vK.7.x, vK.10.x) has never used pretrained weights.

### 2. Metric Inflation (All Runs)

The Dice/IoU/F1 mixed-set averages include authentic images (59.4%) that score 1.0 by predicting all-zeros. v6.5's mixed F1=0.7208 vs tampered-only F1=0.4101 illustrates the inflation. Only v6.5, v8, and vK.10.x report tampered-only metrics — but vK.1–vK.3 and vK.7.x only show the inflated numbers.

### 3. Block 1 Data Leakage (vK.1–vK.3, vK.7.1, vK.7.5)

The dual-block structure in the vK.x series has Block 1 training on the test set. This affects vK.1, vK.2, vK.3, vK.7.1, and vK.7.5. **v6.5 and v8 are NOT affected** — they use a single-block structure with explicit path-level leakage verification. vK.10.x fixed this by removing Block 1.

### 4. Engineering Excellence Without Architectural Foundation (vK.10.x)

vK.10.3b–vK.10.5 built excellent engineering infrastructure (CONFIG, AMP, seeding, early stopping, DataParallel, three-file checkpoints, VRAM auto-scaling) but directed it at a model that cannot learn. The result: sophisticated infrastructure correctly reporting Tam-Dice = 0.0000. Meanwhile, v6.5 had both the right architecture AND solid engineering.

### 5. v8's Hyperparameter Catastrophe

v8 made the right improvements (scheduler, augmentations, per-sample Dice, shortcut detection) but set pos_weight=30.01 (computed from all pixels including authentic) and 16×'d the effective batch without LR rescaling. Result: Tam-F1 regressed 28% (0.41→0.29), copy-move F1 regressed 55% (0.31→0.14).

---

## Conclusions

1. **Best overall run: v6.5** — best metrics (Tam-F1=0.41), best engineering (CONFIG, AMP, DataParallel, early stopping, gradient accumulation, differential LR, W&B), most comprehensive evaluation (threshold opt, Grad-CAM, robustness, forgery breakdown, failure analysis, data leakage verification)

2. **Best evaluation methodology: v8** — added mask-size stratification, shortcut detection, per-sample Dice, expanded augmentations, ReduceLROnPlateau; but hyperparameter regression negated metric gains

3. **Best infrastructure for vK.x series: vK.10.5** — CONFIG, AMP, seeding, DataParallel, three-file checkpoints, `get_base_model()` unwrapper; but lacks pretrained encoder, evaluation features, and produces near-zero localization

4. **No run achieves competitive localization** — v6.5's 0.41 is well below published CASIA-2 benchmarks (0.65+ with similar architectures)

The path forward is clear: **combine v6.5's proven architecture with v8's evaluation improvements and vK.10.5's engineering infrastructure**, while fixing v8's pos_weight and batch size errors.

---

## Individual Audit Files

| File | Covers |
|---|---|
| `Audit_vK1_vK2_vK3.md` | vK.1, vK.2, vK.3, vK.3-run-01 |
| `Audit_v6.5.md` | v6.5 run-01 (SMP pretrained, best metrics) |
| `Audit_v8.md` | v8 run-01 (SMP pretrained, regression analysis) |
| `Audit_vK7.1.md` | vK.7.1 run-01 |
| `Audit_vK7.5.md` | vK.7.5 (incomplete run) |
| `Audit_vK10.3b_vK10.4_vK10.5.md` | vK.10.3b, vK.10.4, vK.10.5 |
| `Run_comparison_table.md` | Cross-version comparison tables |
| `Final_recommendations.md` | Prioritized fix recommendations |
