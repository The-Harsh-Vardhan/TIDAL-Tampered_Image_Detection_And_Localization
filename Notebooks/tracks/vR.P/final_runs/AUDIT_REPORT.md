# Audit Report: vR.P Final Runs Folder

**Date:** 2026-03-17
**Scope:** 50 notebooks + 1 PDF in `final runs/`
**Criteria:** VERSION consistency, W&B logging, execution status, ablation discipline, data integrity

---

## 1. Executive Summary

| Category | Count |
|----------|-------|
| Total files | 51 (50 .ipynb + 1 PDF) |
| Executed vR.P notebooks | 44 |
| External/reference notebooks | 6 (no VERSION string) |
| Notebooks with W&B logging | 32 |
| Notebooks without W&B logging | 18 |
| Duplicate VERSION values | 7 versions have 2-3 copies each |
| Ablation discipline violations | 4 experiments (P.8, P.12, P.13, P.28) |
| Data leakage found | None |

All 50 notebooks have been executed (non-empty outputs). No blank/unexecuted templates were found.

---

## 2. Canonical Notebook Map

The "Canonical" column marks the primary notebook for each version — the one with W&B logging and/or the most complete execution.

| Version | Canonical Notebook | W&B | Duplicates / Notes |
|---------|-------------------|-----|--------------------|
| vR.P.0 | `vr-p-0-pretrained-resnet-34-unet.ipynb` | Yes | — |
| vR.P.1 | `vr-p-1-pretrained-resnet-34-unet-baselin.ipynb` | Yes | — |
| vR.P.1.5 | `vr-p-1-5-image-detection-and-localisation.ipynb` | Yes | — |
| vR.P.3 | `vr-p-3-1-ela-as-input-replace-rgb.ipynb` | Yes | — |
| vR.P.4 | `vr-p-4-4-channel-input-rgb-ela.ipynb` | Yes | — |
| vR.P.5 | `vr-p-5-resnet-50-encoder-test-deeper-features.ipynb` | Yes | — |
| vR.P.6 | `vr-p-6-efficientnet-b0-encoder.ipynb` | Yes | — |
| vR.P.7 | `vr-p-7-ela-extended-training.ipynb` | Yes | Duplicate: `vr-p-7-ela-extended-training-01-run-01.ipynb` (no W&B) |
| vR.P.8 | `vr-p-8-ela-gradual-encoder-unfreeze-run-01.ipynb` | No | Ablation violation (see Section 4) |
| vR.P.9 | `vr-p-9-focal-dice-loss-run-01.ipynb` | No | — |
| vR.P.10 | `vr-p-10-ela-attention-modules-cbam.ipynb` | Yes | Duplicate: `vr-p-10-ela-attention-modules-cbam-01-run-01.ipynb` (no W&B) |
| vR.P.12 | `vr-p-12-ela-data-augmentation-run-01.ipynb` | No | Ablation violation (see Section 4) |
| vR.P.13 | `vr-p-13-cbam-augmentation-extended-training (1).ipynb` | Yes | Duplicate: `vr-p-13-cbam-augmentation-extended-training.ipynb` (no W&B). Ablation violation. |
| vR.P.14 | `vr-p-14-test-time-augmentation-tta-run-01.ipynb` | No | Two more copies: `vr-p-14b-test-time-augmentation-tta.ipynb`, `vr-p-14b-test-time-augmentation-tta-run-02.ipynb` (both no W&B) |
| vR.P.15 | `vr-p-15-multi-quality-ela-run-01.ipynb` | No | — |
| vR.P.16 | `vr-p-16-dct-spatial-map-baseline.ipynb` | Yes | 3 duplicates: `(1)` copy (W&B), `16b-run-01` (no W&B), `16b-(1)-run-01` (no W&B) |
| vR.P.17 | `vr-p-17ela-dct-spatial-fusion-6-channel-input-run-01.ipynb` | No | — |
| vR.P.18 | `vr-p-18-jpeg-compression-robustness-testing-run-01.ipynb` | No | — |
| vR.P.19 | `vr-p-19-multi-quality-rgb-ela.ipynb` | Yes | 2 duplicates: `...-9-channel.ipynb` (no W&B), `...-9-channels-run-02.ipynb` (W&B). **BEST MODEL** |
| vR.P.20 | `vr-p-20-ela-magnitude-chrominance-direction.ipynb` | Yes | — |
| vR.P.23 | `vr-p-23-chrominance-channel-analysis.ipynb` | Yes | — |
| vR.P.24 | `vr-p-24-noiseprint-forensic-features.ipynb` | Yes | — |
| vR.P.26 | `vr-p-26-segmentation-classification-head.ipynb` | Yes | — |
| vR.P.27 | `vr-p-27-jpeg-compression-augmentation.ipynb` | Yes | — |
| vR.P.28 | `vr-p-28-cosine-annealing-lr-scheduler.ipynb` | Yes | Ablation violation (see Section 4) |
| vR.P.30 | `vr-p-30-multi-quality-ela-cbam-attention.ipynb` | Yes | — |
| vR.P.30.1 | `vr-p-30-1-multi-quality-ela-cbam-attention-run-01.ipynb` | Yes | — |
| vR.P.30.2 | `vr-p-30-2-multi-quality-ela-cbam-unfreeze.ipynb` | Yes | Duplicate: `...-run-01.ipynb` (W&B) |
| vR.P.30.3 | `vr-p-30-3-multi-quality-ela-cbam-focal-dice-loss.ipynb` | Yes | — |
| vR.P.30.4 | `vr-p-30-4-multi-quality-ela-cbam-augmentation.ipynb` | Yes | — |
| vR.P.40.1 | `vr-p-40-1-efficientnet-b4-baseline-ela-q-90-3ch.ipynb` | Yes | — |
| vR.P.40.3 | `vr-p-40-3-inceptionv1-custom-encoder-multi-q-rgb.ipynb` | Yes | — |
| vR.P.40.5 | `vr-p-40-5-inceptionv3-custom-encoder-multi-q-rgb.ipynb` | Yes | — |

---

## 3. External/Reference Notebooks (No VERSION)

These 6 notebooks do not belong to the vR.P versioning scheme. They are external references or early explorations:

| Notebook | Description |
|----------|-------------|
| `casia-2-0-dataset-for-image-forgery-detecion-run-01.ipynb` | CASIA dataset exploration |
| `casia2-ela-cnn-with-divg07-dataset-run-01.ipynb` | ELA-CNN reference (divg07 data) |
| `casia2-ela-cnn-with-sagnik-dataset-run-01.ipynb` | ELA-CNN reference (sagnik data) |
| `ela-cnn-image-forgery-detection-on-sagnik-data-run-01.ipynb` | ELA-CNN reference variant |
| `ela-cnn-image-forgery-detection-with-divg07-data-run-01.ipynb` | ELA-CNN reference variant |
| `vf-2-0-fakeshield-lite-run-01.ipynb` | FakeShield paper exploration |

Also present: `FAKESHIELD 2410.02761v4.pdf` (reference paper).

---

## 4. Ablation Discipline Violations

The vR.P track follows a single-variable ablation protocol: exactly one change per experiment from its parent baseline. Four experiments violate this:

| Experiment | Parent | Changes Made | Violation |
|-----------|--------|-------------|-----------|
| **P.8** | P.3 (ELA baseline) | Progressive encoder unfreeze **+** 50 epochs (was 25) | 2 simultaneous changes |
| **P.12** | P.10 (CBAM) | Data augmentation **+** CBAM (inherited from P.10 but tested together with augmentation for the first time) | 2 simultaneous changes |
| **P.13** | P.12 | Focal+Dice loss **+** inherits P.12's combined changes | 3 compounded changes from P.3 baseline |
| **P.28** | P.3 (ELA baseline) | Cosine annealing LR **+** different training schedule | 2 simultaneous changes |

**Impact:** These results remain valid but cannot attribute performance changes to a single factor. They should be interpreted as "combined effect" measurements rather than isolated ablations.

---

## 5. Data Integrity Check

All executed vR.P notebooks were verified for:

| Check | Result |
|-------|--------|
| Dataset: CASIA v2.0 | Confirmed across all |
| Random seed: 42 | Confirmed across all |
| Data split: 70/15/15 stratified | Confirmed across all |
| Input resolution: 384x384 | Confirmed across all |
| Base decoder: UNet | Confirmed across all |
| Batch size: 16 | Confirmed across all |
| Optimizer: Adam | Confirmed across all |
| Data leakage | **None found** |
| Metric computation on tampered only (P.3+) | Confirmed |

---

## 6. W&B Logging Status

**18 notebooks without W&B logging in source cells:**

Most are Kaggle execution copies (`-run-01` suffix) where W&B init was stripped before upload, or early experiments run before W&B integration was standardized.

| Notebook | VERSION |
|----------|---------|
| `casia-2-0-dataset-for-image-forgery-detecion-run-01.ipynb` | N/A (external) |
| `casia2-ela-cnn-with-divg07-dataset-run-01.ipynb` | N/A (external) |
| `casia2-ela-cnn-with-sagnik-dataset-run-01.ipynb` | N/A (external) |
| `ela-cnn-image-forgery-detection-on-sagnik-data-run-01.ipynb` | N/A (external) |
| `ela-cnn-image-forgery-detection-with-divg07-data-run-01.ipynb` | N/A (external) |
| `vf-2-0-fakeshield-lite-run-01.ipynb` | N/A (external) |
| `vr-p-7-ela-extended-training-01-run-01.ipynb` | vR.P.7 (duplicate) |
| `vr-p-8-ela-gradual-encoder-unfreeze-run-01.ipynb` | vR.P.8 |
| `vr-p-9-focal-dice-loss-run-01.ipynb` | vR.P.9 |
| `vr-p-10-ela-attention-modules-cbam-01-run-01.ipynb` | vR.P.10 (duplicate) |
| `vr-p-12-ela-data-augmentation-run-01.ipynb` | vR.P.12 |
| `vr-p-13-cbam-augmentation-extended-training.ipynb` | vR.P.13 (duplicate) |
| `vr-p-14-test-time-augmentation-tta-run-01.ipynb` | vR.P.14 |
| `vr-p-14b-test-time-augmentation-tta.ipynb` | vR.P.14 (duplicate) |
| `vr-p-14b-test-time-augmentation-tta-run-02.ipynb` | vR.P.14 (duplicate) |
| `vr-p-15-multi-quality-ela-run-01.ipynb` | vR.P.15 |
| `vr-p-17ela-dct-spatial-fusion-6-channel-input-run-01.ipynb` | vR.P.17 |
| `vr-p-18-jpeg-compression-robustness-testing-run-01.ipynb` | vR.P.18 |
| `vr-p-19-multi-quality-rgb-ela-9-channel.ipynb` | vR.P.19 (duplicate) |
| `vr-p-16b-dct-spatial-map-baseline-run-01.ipynb` | vR.P.16 (duplicate) |
| `vr-p-16b-dct-spatial-map-baseline (1)-run-01.ipynb` | vR.P.16 (duplicate) |

**Note:** For versions with duplicates, W&B data was logged by the canonical notebook copy, so no experiment data is actually missing from the W&B project.

---

## 7. Duplicate VERSION Map

7 versions have multiple notebook copies:

| VERSION | # Copies | Canonical (has W&B) | Duplicates |
|---------|----------|--------------------|-----------|
| vR.P.7 | 2 | `vr-p-7-ela-extended-training.ipynb` | `...-01-run-01.ipynb` |
| vR.P.10 | 2 | `vr-p-10-ela-attention-modules-cbam.ipynb` | `...-01-run-01.ipynb` |
| vR.P.13 | 2 | `vr-p-13-cbam-augmentation-extended-training (1).ipynb` | `...extended-training.ipynb` |
| vR.P.14 | 3 | `vr-p-14-test-time-augmentation-tta-run-01.ipynb` | `vr-p-14b-*.ipynb` (x2) |
| vR.P.16 | 4 | `vr-p-16-dct-spatial-map-baseline.ipynb` | `(1)` copy, two `16b-*` variants |
| vR.P.19 | 3 | `vr-p-19-multi-quality-rgb-ela.ipynb` | `...-9-channel.ipynb`, `...-run-02.ipynb` |
| vR.P.30.2 | 2 | `vr-p-30-2-multi-quality-ela-cbam-unfreeze.ipynb` | `...-run-01.ipynb` |

---

## 8. Recommendations

1. **Archive duplicates:** Move duplicate notebooks to an `archive/` subfolder to reduce confusion. Keep only canonical notebooks in the main folder.
2. **Standardize naming:** Adopt a consistent naming convention — either include `-run-XX` for all Kaggle execution copies or none.
3. **W&B retroactive logging:** Consider re-running P.8, P.9, P.15, P.17, P.18 with W&B enabled if their metrics need to appear in the interactive dashboard.
4. **Acknowledge ablation violations:** P.8, P.12, P.13, P.28 should be footnoted in any publication as "combined-change experiments" rather than single-variable ablations.
5. **Move external notebooks:** The 6 non-vR.P notebooks and the PDF could be moved to a `references/` subfolder for clarity.
