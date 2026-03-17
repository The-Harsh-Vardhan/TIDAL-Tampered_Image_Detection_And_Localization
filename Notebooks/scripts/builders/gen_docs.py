#!/usr/bin/env python3
"""Generate per-experiment documentation for P.30.x series. DELETE after use."""
import os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs vR.P.x")

experiments = [
    {
        "key": "P.30", "version": "vR.P.30",
        "change": "Multi-quality ELA + CBAM attention (25ep, BCE+Dice)",
        "parent": "vR.P.15",
        "single_var": "Add CBAM attention (Channel + Spatial) to all 5 UNet decoder blocks",
        "epochs": 25, "loss": "BCE+Dice", "expected_f1": "0.76-0.78",
        "cells": "[0] Title, [1] Changelog, [2] Config, [12] Model (+CBAM), [25] Results, [26] Discussion",
        "code": "CBAM classes (SEBlock, ChannelAttention, SpatialAttention, CBAMBlock) from P.10 Cell [12]. Decoder injection loop from P.10.",
        "risk": "CBAM may not improve on decorrelated multi-Q ELA channels (vs correlated RGB ELA in P.10)",
    },
    {
        "key": "P.30.1", "version": "vR.P.30.1",
        "change": "Multi-quality ELA + CBAM attention (50ep, BCE+Dice)",
        "parent": "vR.P.30",
        "single_var": "Extend training from 25 to 50 epochs (PATIENCE=10)",
        "epochs": 50, "loss": "BCE+Dice", "expected_f1": "0.78-0.80",
        "cells": "[0] Title, [1] Changelog, [2] Config (EPOCHS=50, PATIENCE=10), [25] Results, [26] Discussion",
        "code": "Config changes only: EPOCHS=50, PATIENCE=10. All CBAM code inherited from P.30.",
        "risk": "Diminishing returns; overfitting possible with CBAM extra parameters and more epochs",
    },
    {
        "key": "P.30.2", "version": "vR.P.30.2",
        "change": "Multi-quality ELA + CBAM + Progressive Unfreeze (40ep, BCE+Dice)",
        "parent": "vR.P.30",
        "single_var": "Add progressive encoder unfreezing (3 stages) from P.8",
        "epochs": 40, "loss": "BCE+Dice", "expected_f1": "0.77-0.80",
        "cells": "[0,1,2] Header/Config, [12] Model (+freeze helpers from P.8), [14] Loss/Optimizer (progressive), [15] Training loop (stage-aware from P.8), [25,26] Results/Discussion",
        "code": "Merges P.15 (dataset) + P.10 (CBAM) + P.8 (progressive unfreeze). rebuild_optimizer() includes CBAM params in decoder group.",
        "risk": "Most complex notebook. Optimizer rebuild must capture CBAM params. Unfreezing may conflict with multi-Q ELA statistics.",
    },
    {
        "key": "P.30.3", "version": "vR.P.30.3",
        "change": "Multi-quality ELA + CBAM + Focal+Dice loss (25ep)",
        "parent": "vR.P.30",
        "single_var": "Replace BCE+Dice with Focal+Dice (alpha=0.25, gamma=2.0)",
        "epochs": 25, "loss": "Focal+Dice", "expected_f1": "0.76-0.78",
        "cells": "[0,1,2] Header/Config (+FOCAL_ALPHA/GAMMA), [14] Loss (FocalLoss replaces SoftBCE), [25,26] Results/Discussion",
        "code": "smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0). P.9 showed Focal was neutral alone; tests CBAM interaction.",
        "risk": "Focal may be redundant when CBAM already provides attention-based focus. P.9 showed only +0.03pp.",
    },
    {
        "key": "P.30.4", "version": "vR.P.30.4",
        "change": "Multi-quality ELA + CBAM + Geometric Augmentation (50ep, BCE+Dice)",
        "parent": "vR.P.30.1",
        "single_var": "Add geometric-only augmentation (HFlip, VFlip, Rotate90, ShiftScaleRotate)",
        "epochs": 50, "loss": "BCE+Dice", "expected_f1": "0.77-0.79",
        "cells": "[0,1,2] Header/Config (+albumentations), [8] Dataset (transform param), [9] Split (train_transform), [25,26] Results/Discussion",
        "code": "Albumentations geometric pipeline (no brightness/blur to preserve ELA signal). Applied to multi-Q ELA array.",
        "risk": "P.12 showed augmentation was neutral/negative with ELA. Hypothesis: CBAM spatial attention makes model robust to geometric transforms.",
    },
]

for exp in experiments:
    d = os.path.join(base, f"docs-{exp['version']}")
    os.makedirs(d, exist_ok=True)

    # experiment_description.md
    with open(os.path.join(d, "experiment_description.md"), "w", encoding="utf-8") as f:
        f.write(f"""# {exp['version']} -- Experiment Description

| Field | Value |
|-------|-------|
| **Version** | {exp['version']} |
| **Change** | {exp['change']} |
| **Parent** | {exp['parent']} |
| **Single Variable** | {exp['single_var']} |
| **Track** | Pretrained Localization (Track 2) |

---

## Hypothesis

Combining multi-quality ELA input (from P.15, +4.09pp F1) with CBAM attention (from P.10, +3.54pp F1 isolated) should produce an additive improvement because these techniques operate on different parts of the pipeline:
- **Multi-Q ELA** improves WHAT the model sees (input representation)
- **CBAM** improves WHERE the decoder focuses (attention mechanism)

## Motivation

P.15 and P.10 are the two most impactful single-variable improvements in the ablation study. Their combination has never been tested.

## Configuration

| Parameter | Value |
|-----------|-------|
| Input | Multi-Q ELA (Q=75/85/95) |
| Attention | CBAM (reduction=16, kernel=7) |
| Loss | {exp['loss']} |
| Epochs | {exp['epochs']} |
| Encoder | ResNet-34 (frozen+BN) |

## What DID NOT Change
- Dataset: CASIA v2.0 with GT masks
- Encoder: ResNet-34 (ImageNet pretrained)
- Data split: 70/15/15 stratified, seed=42
- Image size: 384x384
- Batch size: 16
""")

    # implementation_plan.md
    with open(os.path.join(d, "implementation_plan.md"), "w", encoding="utf-8") as f:
        f.write(f"""# {exp['version']} -- Implementation Plan

## Cells Modified from Parent ({exp['parent']})

{exp['cells']}

## Key Code Sources

{exp['code']}

## Verification Checklist

- [ ] VERSION = '{exp['version']}' in Cell [2]
- [ ] ATTENTION_TYPE = 'cbam' in Cell [2]
- [ ] DECODER_CHANNELS = (256,128,64,32,16) in Cell [2]
- [ ] CBAMBlock class defined in Cell [12]
- [ ] Decoder injection loop in Cell [12]
- [ ] W&B config includes attention_type, attention_reduction, cbam_kernel_size
- [ ] All 28 cells present
- [ ] Notebook runs without errors on Kaggle
""")

    # expected_outcomes.md
    with open(os.path.join(d, "expected_outcomes.md"), "w", encoding="utf-8") as f:
        f.write(f"""# {exp['version']} -- Expected Outcomes

## Metric Targets

| Metric | Expected Range | Success Threshold |
|--------|---------------|-------------------|
| Pixel F1 | {exp['expected_f1']} | > 0.7379 (+0.5pp over P.15) |
| Pixel IoU | TBD | > 0.5835 (+0.5pp over P.15) |
| Pixel AUC | > 0.96 | > 0.9608 (match P.15) |
| Image Accuracy | > 87% | > 87.53% (match P.15) |

## Success Criteria

- **POSITIVE:** Pixel F1 > 0.7379 (P.15 + 0.5pp)
- **STRONG POSITIVE:** Pixel F1 > 0.76 (additive combination)
- **NEUTRAL:** Pixel F1 within +/-0.5pp of P.15 (0.7329)
- **NEGATIVE:** Pixel F1 < 0.7279 (P.15 - 0.5pp)

## Failure Modes

1. {exp['risk']}
2. CBAM adds ~50K parameters -- may overfit on small CASIA dataset
3. Frozen encoder conv1 processes decorrelated multi-Q ELA channels suboptimally
""")

    print(f"Created 3 files in docs-{exp['version']}/")

print("Done! 15 documentation files created.")
