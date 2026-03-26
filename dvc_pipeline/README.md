# TIDAL DVC Pipeline

Reproducible 4-stage training pipeline for Tampered Image Detection & Localization.

## Pipeline DAG

```
preprocess → train → evaluate
                  ↘ visualize
```

## Quick Start

```bash
cd dvc_pipeline

# Run full pipeline
dvc repro

# Run single stage
dvc repro train

# View metrics
dvc metrics show

# Compare params
dvc params diff
```

## Stages

| Stage | Script | Output |
|-------|--------|--------|
| preprocess | `src/preprocess.py` | `artifacts/ela_statistics.json` |
| train | `src/train.py` | `models/best_model.pt` |
| evaluate | `src/evaluate.py` | `evaluation_results/`, `metrics/eval_metrics.json` |
| visualize | `src/visualize.py` | `visualize_results/comparison_grid.png` |

## Configuration

All hyperparameters in `params.yaml`. Key settings match best run **vR.P.19**:
- 9-channel Multi-Q RGB ELA (Q=75/85/95)
- UNet + ResNet-34 (ImageNet, body frozen + BN unfrozen)
- BCE + Dice loss, Adam, ReduceLROnPlateau
- 384×384 input, batch_size=16, 25 epochs, patience=7
