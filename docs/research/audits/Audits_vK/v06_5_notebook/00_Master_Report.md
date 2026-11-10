# Audit 6.5 — Master Report

## Training Run: v6.5 Tampered Image Detection & Localization

**Notebook:** `v6-5-tampered-image-detection-localization-run-01.ipynb`  
**Run date:** March 11, 2026  
**Platform:** Kaggle (2× Tesla T4 GPUs, DataParallel)  
**Dataset:** CASIA Splicing Detection + Localization  
**Model:** SMP U-Net / ResNet34 encoder / ImageNet pretrained  
**Loss:** BCE + Dice  
**Optimizer:** AdamW (encoder LR=1e-4, decoder LR=1e-3)  
**Image size:** 384×384  
**Effective batch size:** 16 (batch=4 × accum=4)  
**Training:** 25/50 epochs (early stopping triggered at patience=10)  

---

## TRAINING RUN VERDICT

### Status: VALID — with documented weaknesses

The training run is **technically correct**. The model trained, converged in a normal pattern, triggered early stopping appropriately, and produced metrics consistent with expectations for a U-Net/ResNet34 on the CASIA dataset. No evidence of metric bugs, mask misalignment, or evaluation errors was found.

However, the run reveals several **genuine weaknesses** that limit practical utility:

1. Severe overfitting after epoch ~15 (val loss diverging while train loss continues to decrease)
2. Very low best threshold (0.1327) suggesting the model outputs poorly calibrated probabilities
3. Near-total failure on copy-move forgeries (F1=0.31)
4. Complete failure cases (worst 10 predictions have F1=0.0)
5. Concerning robustness drop under all degradation conditions (~13% F1 drop)

These are **real model limitations**, not bugs — which is actually a positive sign that the evaluation pipeline is working honestly.

---

## KEY METRICS SUMMARY

| Metric | Value | Assessment |
|---|---|---|
| Best Val Pixel-F1 | 0.7289 (epoch 15) | Realistic |
| Test Mixed Pixel-F1 | 0.7208 ± 0.4158 | Realistic but high variance |
| Test Tampered-only F1 | 0.4101 ± 0.4148 | Weak — reveals authentic inflation |
| Splicing F1 | 0.5901 ± 0.3850 | Moderate |
| Copy-move F1 | 0.3105 ± 0.3968 | Poor |
| Image-level Accuracy | 0.8246 | Reasonable |
| Image-level AUC-ROC | 0.8703 | Good |
| Best Threshold | 0.1327 | Suspiciously low |
| Training epochs | 25/50 (early stop at 15+10) | Normal |

---

## AUDIT DOCUMENTS

| Document | Focus |
|---|---|
| [01_Training_Dynamics.md](01_Training_Dynamics.md) | Loss curves, convergence, overfitting analysis |
| [02_Metric_Sanity_Check.md](02_Metric_Sanity_Check.md) | Whether metrics are realistic and trustworthy |
| [03_Dataset_Pipeline_Validation.md](03_Dataset_Pipeline_Validation.md) | Image-mask pairing, preprocessing, splits |
| [04_Class_Imbalance_Analysis.md](04_Class_Imbalance_Analysis.md) | Background dominance, metric inflation |
| [05_Checkpoint_And_Prediction_Quality.md](05_Checkpoint_And_Prediction_Quality.md) | Checkpoint strategy, prediction analysis |
| [06_Shortcut_Learning_And_Baselines.md](06_Shortcut_Learning_And_Baselines.md) | Artifact learning risks, literature comparison |
| [07_Engineering_Review.md](07_Engineering_Review.md) | Code quality, config system, reproducibility |
| [08_Recommended_Fixes.md](08_Recommended_Fixes.md) | Prioritized fixes with concrete suggestions |

---

## FINAL ASSESSMENT

The model is **suitable for further experimentation** but **not yet ready for assignment submission** in its current state. The core pipeline is sound, but the gap between mixed-set metrics (inflated by authentic images scoring perfect F1) and tampered-only metrics (F1=0.41) is the central weakness that must be addressed.

### Ready for:
- Further experimentation ✓
- Using as a baseline for architecture/hyperparameter search ✓

### Not yet ready for:
- Assignment submission (tampered-only F1 too low, copy-move failure needs analysis)
- Production use

### Priority actions:
1. Add a learning rate scheduler (ReduceLROnPlateau or CosineAnnealing) — the model is plateauing too early
2. Investigate the extremely low threshold (0.1327) — the model is outputting very low probabilities for tampered regions
3. Analyze copy-move failure mode — F1=0.31 suggests the model largely fails on this forgery type
4. Report tampered-only metrics prominently — mixed-set F1=0.72 is misleading due to authentic image inflation
5. Add stronger augmentations (ColorJitter, ElasticTransform, CoarseDropout) to combat overfitting
