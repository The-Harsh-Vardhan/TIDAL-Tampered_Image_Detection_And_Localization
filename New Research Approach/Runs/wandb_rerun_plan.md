# W&B Rerun Deployment Plan

## Experiment Inventory (22 Completed Runs)

| # | Run File (Runs/) | Source | Epochs | Weight | Notes |
|---|-----------------|--------|--------|--------|-------|
| 1 | vr-p-0-dataset-with-no-gt-available-run-01 | vR.P.0 | 25 | Light | No GT masks, baseline |
| 2 | vr-p-1-detection-localisation-dataset-run-01 | vR.P.1 | 25 | Light | Proper baseline |
| 3 | vr-p-1-5-training-speed-optimizations-run-01 | vR.P.1.5 | 25 | Light | AMP/TF32 speed opts |
| 4 | vr-p-2-gradual-encoder-unfreeze-run-01 | vR.P.2 | 25 | Light | Differential LR |
| 5 | vr-p-3-ela-as-input-replace-rgb-run-01 | vR.P.3 | 25 | Light | ELA input (breakthrough) |
| 6 | vr-p-3-ela-as-input-replace-rgb-run-02 | vR.P.3 | 25 | Light | Reproducibility run |
| 7 | vr-p-4-4-channel-input-rgb-ela-run-01 | vR.P.4 | 25 | Light | 4ch RGB+ELA |
| 8 | vr-p-5-resnet-50-encoder-test-deeper-features-run-01 | vR.P.5 | 25 | Light | ResNet-50 |
| 9 | vr-p-6-efficientnet-b0-encoder-run-01 | vR.P.6 | 25 | Light | EfficientNet-B0 |
| 10 | vr-p-7-ela-extended-training-run-01 | vR.P.7 | 50 | **Heavy** | Extended training |
| 11 | vr-p-8-ela-gradual-encoder-unfreeze-run-01 | vR.P.8 | 40 | **Heavy** | Progressive 3-stage |
| 12 | vr-p-9-focal-dice-loss-run-01 | vR.P.9 | 25 | Light | Focal+Dice loss |
| 13 | vr-p-10-ela-attention-modules-cbam-run-01 | vR.P.10 | 25 | Medium | CBAM attention |
| 14 | vr-p-10-ela-attention-modules-cbam-run-02 | vR.P.10 | 25 | Medium | Reproducibility run |
| 15 | vr-p-12-ela-data-augmentation-run-01 | vR.P.12 | 50 | **Heavy** | Augmentation + Focal |
| 16 | vr-p-14-test-time-augmentation-tta-run-01 | vR.P.14 | 25 | Medium | TTA (cell 18 crashed) |
| 17 | vr-p-14b-test-time-augmentation-tta-run-02 | vR.P.14 | 25 | Medium | TTA complete (P.14b) |
| 18 | vr-p-15-multi-quality-ela-run-01 | vR.P.15 | 25 | Light | Multi-Q ELA (BEST) |
| 19 | vr-p-16b-dct-spatial-map-baseline-run-01 | vR.P.16 | 25 | Light | DCT baseline |
| 20 | vr-p-17ela-dct-spatial-fusion-6-channel-input-run-01 | vR.P.17 | 25 | Medium | ELA+DCT 6ch |
| 21 | vr-p-18-jpeg-compression-robustness-testing-run-01 | vR.P.18 | 0 | Light | Eval-only (INVALID) |
| 22 | casia-2-0-dataset-for-image-forgery-detecion-run-01 | vR.P.11 | 50 | **Heavy** | 512x512 hi-res |

**Weight distribution**: 4 Heavy (50ep), 4 Medium (25ep+extra compute), 14 Light (25ep standard)

---

## Distribution: 6 Kaggle Accounts

Each Kaggle account gets ~30h GPU quota. Each 25-epoch run takes ~25-35min, each 50-epoch run takes ~50-70min.

### Runner 1 (Account 1) — 4 experiments, ~2.5h
| Order | Source | Slug | Est. Time |
|-------|--------|------|-----------|
| 1 | vR.P.7 | vr-p-7-ela-extended-training-run-01 | 60min |
| 2 | vR.P.0 | vr-p-0-dataset-with-no-gt-available-run-01 | 30min |
| 3 | vR.P.1 | vr-p-1-detection-localisation-dataset-run-01 | 30min |
| 4 | vR.P.1.5 | vr-p-1-5-training-speed-optimizations-run-01 | 30min |

### Runner 2 (Account 2) — 4 experiments, ~2.5h
| Order | Source | Slug | Est. Time |
|-------|--------|------|-----------|
| 1 | vR.P.8 | vr-p-8-ela-gradual-encoder-unfreeze-run-01 | 50min |
| 2 | vR.P.2 | vr-p-2-gradual-encoder-unfreeze-run-01 | 30min |
| 3 | vR.P.3 r01 | vr-p-3-ela-as-input-replace-rgb-run-01 | 30min |
| 4 | vR.P.3 r02 | vr-p-3-ela-as-input-replace-rgb-run-02 | 30min |

### Runner 3 (Account 3) — 4 experiments, ~2.5h
| Order | Source | Slug | Est. Time |
|-------|--------|------|-----------|
| 1 | vR.P.12 | vr-p-12-ela-data-augmentation-run-01 | 60min |
| 2 | vR.P.4 | vr-p-4-4-channel-input-rgb-ela-run-01 | 30min |
| 3 | vR.P.5 | vr-p-5-resnet-50-encoder-test-deeper-features-run-01 | 30min |
| 4 | vR.P.6 | vr-p-6-efficientnet-b0-encoder-run-01 | 30min |

### Runner 4 (Account 4) — 4 experiments, ~2.5h
| Order | Source | Slug | Est. Time |
|-------|--------|------|-----------|
| 1 | vR.P.11 | casia-2-0-dataset-for-image-forgery-detecion-run-01 | 60min |
| 2 | vR.P.9 | vr-p-9-focal-dice-loss-run-01 | 30min |
| 3 | vR.P.16 | vr-p-16b-dct-spatial-map-baseline-run-01 | 30min |
| 4 | vR.P.17 | vr-p-17ela-dct-spatial-fusion-6-channel-input-run-01 | 35min |

### Runner 5 (Account 5) — 3 experiments, ~1.5h
| Order | Source | Slug | Est. Time |
|-------|--------|------|-----------|
| 1 | vR.P.10 r01 | vr-p-10-ela-attention-modules-cbam-run-01 | 30min |
| 2 | vR.P.10 r02 | vr-p-10-ela-attention-modules-cbam-run-02 | 30min |
| 3 | vR.P.15 | vr-p-15-multi-quality-ela-run-01 | 30min |

### Runner 6 (Account 6) — 3 experiments, ~1.5h
| Order | Source | Slug | Est. Time |
|-------|--------|------|-----------|
| 1 | vR.P.14 r01 | vr-p-14-test-time-augmentation-tta-run-01 | 35min |
| 2 | vR.P.14 r02 | vr-p-14b-test-time-augmentation-tta-run-02 | 35min |
| 3 | vR.P.18 | vr-p-18-jpeg-compression-robustness-testing-run-01 | 15min |

---

## Setup Instructions

### 1. W&B API Key
All 6 Kaggle accounts must have the same W&B API key set as a Kaggle Secret:
- Go to Kaggle > Account > Secrets
- Add secret: key=`WANDB_API_KEY`, value=`<your-wandb-api-key>`

### 2. Upload Source Notebooks
Each account uploads its assigned source notebooks (the W&B-patched versions from the main dir).

### 3. Run Overnight
Start all 6 runners. Each will execute its experiments sequentially.

### 4. Monitor
Check https://wandb.ai/<team>/tamper-detection-ablation for live metrics.

### 5. After Completion
Run the leaderboard notebook (wandb_leaderboard.ipynb) to generate comparison tables.
