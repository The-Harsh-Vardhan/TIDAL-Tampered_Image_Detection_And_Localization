"""Update Final Notebooks: W&B metadata + documentation completeness.

Changes:
1. W&B metadata: Add tags=['final','colab'], group='TIDAL-Final-Ablation',
   prefix run name with '[FINAL]', add 'notebook_type': 'final' to config
2. Cell 13: Add augmentation justification and hyperparameter rationale
3. Fix W&B reinit deprecation warning (reinit=True → finish_previous=True)
"""
import json
import os
import re

FINAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = ['vR.P.19', 'vR.P.30', 'vR.P.30.1', 'vR.P.30.2', 'vR.P.30.3', 'vR.P.30.4']

# ============================================================
# W&B init replacement blocks
# ============================================================
# P.19 has no attention config keys; P.30.x have CBAM keys
WANDB_INIT_P19 = """wandb.init(
            project=WANDB_PROJECT,
            name=f'[FINAL] {VERSION}',
            group='TIDAL-Final-Ablation',
            tags=['final', 'colab', 'multi-q-rgb-ela', '9ch'],
            config={
                'experiment': EXPERIMENT_ID, 'version': VERSION, 'change': CHANGE,
                'run': RUN_ID, 'dataset': DATASET_NAME, 'feature_set': FEATURE_SET,
                'notebook_type': 'final',
                'input_type': globals().get('INPUT_TYPE', FEATURE_SET),
                'tta': USE_TTA, 'jpeg_aug': JPEG_AUG,
                'edge_supervision': EDGE_SUPERVISION, 'noise_features': NOISE_FEATURES,
                'encoder': ENCODER, 'in_channels': IN_CHANNELS,
                'img_size': globals().get('IMAGE_SIZE', globals().get('IMG_SIZE', 'N/A')), 'batch_size': globals().get('BATCH_SIZE', 'N/A'),
                'epochs': globals().get('EPOCHS', 'N/A'), 'learning_rate': globals().get('LEARNING_RATE', 'N/A'), 'patience': globals().get('PATIENCE', 'N/A'),
            },
            finish_previous=True,
        )"""

WANDB_INIT_P30 = """wandb.init(
            project=WANDB_PROJECT,
            name=f'[FINAL] {VERSION}',
            group='TIDAL-Final-Ablation',
            tags=['final', 'colab', 'multi-q-ela', 'cbam'],
            config={
                'experiment': EXPERIMENT_ID, 'version': VERSION, 'change': CHANGE,
                'run': RUN_ID, 'dataset': DATASET_NAME, 'feature_set': FEATURE_SET,
                'notebook_type': 'final',
                'input_type': globals().get('INPUT_TYPE', FEATURE_SET),
                'tta': USE_TTA, 'jpeg_aug': JPEG_AUG,
                'edge_supervision': EDGE_SUPERVISION, 'noise_features': NOISE_FEATURES,
                'encoder': ENCODER, 'in_channels': IN_CHANNELS,
                'img_size': IMAGE_SIZE, 'batch_size': globals().get('BATCH_SIZE', 'N/A'),
                'epochs': globals().get('EPOCHS', 'N/A'), 'learning_rate': globals().get('LEARNING_RATE', 'N/A'), 'patience': globals().get('PATIENCE', 'N/A'),
                'attention_type': ATTENTION_TYPE, 'attention_reduction': ATTENTION_REDUCTION,
                'cbam_kernel_size': CBAM_KERNEL_SIZE,
            },
            finish_previous=True,
        )"""

WANDB_INIT_P302 = """wandb.init(
            project=WANDB_PROJECT,
            name=f'[FINAL] {VERSION}',
            group='TIDAL-Final-Ablation',
            tags=['final', 'colab', 'multi-q-ela', 'cbam', 'progressive-unfreeze'],
            config={
                'experiment': EXPERIMENT_ID, 'version': VERSION, 'change': CHANGE,
                'run': RUN_ID, 'dataset': DATASET_NAME, 'feature_set': FEATURE_SET,
                'notebook_type': 'final',
                'input_type': globals().get('INPUT_TYPE', FEATURE_SET),
                'tta': USE_TTA, 'jpeg_aug': JPEG_AUG,
                'edge_supervision': EDGE_SUPERVISION, 'noise_features': NOISE_FEATURES,
                'encoder': ENCODER, 'in_channels': IN_CHANNELS,
                'img_size': IMAGE_SIZE, 'batch_size': globals().get('BATCH_SIZE', 'N/A'),
                'epochs': globals().get('EPOCHS', 'N/A'), 'learning_rate': globals().get('LEARNING_RATE', 'N/A'), 'patience': globals().get('PATIENCE', 'N/A'),
                'attention_type': ATTENTION_TYPE, 'attention_reduction': ATTENTION_REDUCTION,
                'cbam_kernel_size': CBAM_KERNEL_SIZE,
            },
            finish_previous=True,
        )"""

WANDB_INIT_P303 = """wandb.init(
            project=WANDB_PROJECT,
            name=f'[FINAL] {VERSION}',
            group='TIDAL-Final-Ablation',
            tags=['final', 'colab', 'multi-q-ela', 'cbam', 'focal-loss'],
            config={
                'experiment': EXPERIMENT_ID, 'version': VERSION, 'change': CHANGE,
                'run': RUN_ID, 'dataset': DATASET_NAME, 'feature_set': FEATURE_SET,
                'notebook_type': 'final',
                'input_type': globals().get('INPUT_TYPE', FEATURE_SET),
                'tta': USE_TTA, 'jpeg_aug': JPEG_AUG,
                'edge_supervision': EDGE_SUPERVISION, 'noise_features': NOISE_FEATURES,
                'encoder': ENCODER, 'in_channels': IN_CHANNELS,
                'img_size': IMAGE_SIZE, 'batch_size': globals().get('BATCH_SIZE', 'N/A'),
                'epochs': globals().get('EPOCHS', 'N/A'), 'learning_rate': globals().get('LEARNING_RATE', 'N/A'), 'patience': globals().get('PATIENCE', 'N/A'),
                'attention_type': ATTENTION_TYPE, 'attention_reduction': ATTENTION_REDUCTION,
                'cbam_kernel_size': CBAM_KERNEL_SIZE,
            },
            finish_previous=True,
        )"""

WANDB_INIT_P304 = """wandb.init(
            project=WANDB_PROJECT,
            name=f'[FINAL] {VERSION}',
            group='TIDAL-Final-Ablation',
            tags=['final', 'colab', 'multi-q-ela', 'cbam', 'augmentation'],
            config={
                'experiment': EXPERIMENT_ID, 'version': VERSION, 'change': CHANGE,
                'run': RUN_ID, 'dataset': DATASET_NAME, 'feature_set': FEATURE_SET,
                'notebook_type': 'final',
                'input_type': globals().get('INPUT_TYPE', FEATURE_SET),
                'tta': USE_TTA, 'jpeg_aug': JPEG_AUG,
                'edge_supervision': EDGE_SUPERVISION, 'noise_features': NOISE_FEATURES,
                'encoder': ENCODER, 'in_channels': IN_CHANNELS,
                'img_size': IMAGE_SIZE, 'batch_size': globals().get('BATCH_SIZE', 'N/A'),
                'epochs': globals().get('EPOCHS', 'N/A'), 'learning_rate': globals().get('LEARNING_RATE', 'N/A'), 'patience': globals().get('PATIENCE', 'N/A'),
                'attention_type': ATTENTION_TYPE, 'attention_reduction': ATTENTION_REDUCTION,
                'cbam_kernel_size': CBAM_KERNEL_SIZE,
            },
            finish_previous=True,
        )"""

WANDB_MAP = {
    'vR.P.19': WANDB_INIT_P19,
    'vR.P.30': WANDB_INIT_P30,
    'vR.P.30.1': WANDB_INIT_P30,
    'vR.P.30.2': WANDB_INIT_P302,
    'vR.P.30.3': WANDB_INIT_P303,
    'vR.P.30.4': WANDB_INIT_P304,
}

# ============================================================
# Per-notebook Cell 13 (Training config) — updated with
# augmentation justification and hyperparameter rationale
# ============================================================
CELL13_FINAL = {
    'vR.P.19': '''---

## 5. Training

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Loss** | BCE + Dice (combined) | BCE for per-pixel classification; Dice for overlap maximization despite class imbalance |
| **Optimizer** | Adam (weight_decay=1e-5) | Adam adapts LR per-parameter; mild L2 regularization prevents overfitting |
| **LR** | 1e-3 | Standard for decoder-only training with frozen encoder; well-validated in prior runs |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) | Halves LR when val loss plateaus; patience=3 avoids premature reduction |
| **Early stopping** | patience=7, monitor=val_loss | Prevents overfitting while allowing recovery from LR reductions |
| **Epochs** | 25 max | Sufficient for convergence with frozen encoder (few trainable params) |
| **Batch size** | 16 | Largest stable batch for 384x384x9 input on T4 (16 GB VRAM) |
| **Image size** | 384x384 | Balances spatial detail for localization vs. GPU memory constraints |
| **Input** | 9-channel Multi-Quality RGB ELA (Q=75/85/95) | Preserves chrominance artifacts lost in grayscale ELA |

### Data Augmentation

**No augmentation is applied in this experiment.** This is a deliberate design choice:
- ELA forensic signals are sensitive to intensity transforms (brightness, blur, noise would corrupt the compression artifact signal)
- Geometric augmentation was tested separately in vR.P.30.4 and showed marginal benefit (+/- 1pp F1)
- The CASIA v2.0 dataset contains ~12,614 images with diverse tampering types, providing sufficient variation
- vR.1.2 (ETASR track) demonstrated that augmentation can be counterproductive for this task (-2.85pp)

### Why Single LR?

All encoder conv weights are frozen except conv1 (re-initialized for 9-channel input via weight tiling).
Only conv1, BatchNorm params, and decoder are trainable. These can safely share a single LR
as they are all adapting to the new 9-channel ELA domain.
''',

    'vR.P.30': '''---

## 5. Training

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Loss** | BCE + Dice (combined) | BCE for per-pixel classification; Dice for overlap maximization despite class imbalance |
| **Optimizer** | Adam (weight_decay=1e-5) | Adam adapts LR per-parameter; mild L2 regularization prevents overfitting |
| **LR** | 1e-3 | Standard for decoder-only training; validated in P.3--P.15 experiments |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) | Halves LR when val loss plateaus; patience=3 avoids premature reduction |
| **Early stopping** | patience=7, monitor=val_loss | Prevents overfitting while allowing recovery from LR reductions |
| **Epochs** | 25 max | Baseline run to measure component interaction (extended in P.30.1) |
| **Batch size** | 16 | Largest stable batch for 384x384x3 on T4 GPU (16 GB VRAM) |
| **Image size** | 384x384 | Balances spatial detail for localization vs. GPU memory |
| **Attention** | CBAM (reduction=16, kernel=7) | Standard CBAM hyperparameters from Woo et al. (2018); reduction=16 balances expressiveness and efficiency |

### Data Augmentation

**No augmentation is applied in this baseline experiment.** This is deliberate:
- This notebook establishes the Multi-Q ELA + CBAM baseline for the P.30.x ablation series
- Augmentation is tested separately in vR.P.30.4 to isolate its contribution
- Prior experiment vR.1.2 showed augmentation can hurt ELA-based models (-2.85pp accuracy)
- The CASIA v2.0 dataset (~12,614 images) provides sufficient variation for this architecture

### Why Single LR?

Encoder body is frozen; only BN params, CBAM attention modules, and decoder are trainable.
All trainable params share the same LR since they are all learning to adapt to multi-Q ELA input.
''',

    'vR.P.30.1': '''---

## 5. Training

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Loss** | BCE + Dice (combined) | BCE for per-pixel classification; Dice for overlap maximization despite class imbalance |
| **Optimizer** | Adam (weight_decay=1e-5) | Adam adapts LR per-parameter; mild L2 regularization prevents overfitting |
| **LR** | 1e-3 | Same as P.30 baseline; only change is training duration |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) | Halves LR when val loss plateaus |
| **Early stopping** | patience=10, monitor=val_loss | Higher patience (10 vs 7) to allow full convergence over 50 epochs |
| **Epochs** | 50 max | Extended from P.30's 25 epochs --- P.30 was still improving at epoch 23 |
| **Batch size** | 16 | Largest stable batch for 384x384x3 on T4 GPU (16 GB VRAM) |
| **Image size** | 384x384 | Balances spatial detail for localization vs. GPU memory |
| **Attention** | CBAM (reduction=16, kernel=7) | Standard CBAM hyperparameters from Woo et al. (2018) |

### Data Augmentation

**No augmentation is applied.** This is deliberate to isolate the effect of extended training:
- P.30 (25ep) vs P.30.1 (50ep) is a controlled comparison --- only training duration changes
- Augmentation is tested separately in vR.P.30.4
- ELA forensic signals are sensitive to intensity augmentations (brightness, blur, noise)

### Why Single LR?

Encoder body is frozen; only BN params, CBAM attention modules, and decoder are trainable.
Extended training (50 epochs) with higher patience (10) allows the model to fully converge.
''',

    'vR.P.30.2': '''---

## 5. Training

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Loss** | BCE + Dice (combined) | BCE for per-pixel classification; Dice for overlap maximization |
| **Optimizer** | Adam (differential LR) | Encoder and decoder use different LRs to prevent catastrophic forgetting |
| **LR (decoder)** | 1e-3 | Decoder learns new features from scratch |
| **LR (encoder)** | 1e-5 | 100x lower to preserve pretrained features while allowing fine-tuning |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) | Applied to both LR groups simultaneously |
| **Early stopping** | patience=7, monitor=val_loss | Standard patience for progressive unfreeze |
| **Epochs** | 40 max | 3 stages of ~13 epochs each |
| **Batch size** | 16 | Largest stable batch for 384x384x3 on T4 GPU (16 GB VRAM) |
| **Image size** | 384x384 | Balances spatial detail for localization vs. GPU memory |
| **Attention** | CBAM (reduction=16, kernel=7) | Standard CBAM hyperparameters from Woo et al. (2018) |

### Progressive Unfreeze Schedule

| Stage | Epochs | Unfrozen Layers | Trainable Params |
|-------|--------|----------------|-----------------|
| 0 | 1--10 | Decoder + CBAM + BN only | ~3.2M (13%) |
| 1 | 11--25 | + layer4 | ~16.3M (67%) |
| 2 | 26--40 | + layer3 + layer4 | ~23.1M (95%) |

Differential LR is used from Stage 1: encoder layers get 1e-5 while decoder keeps 1e-3
to prevent catastrophic forgetting of pretrained features.

### Data Augmentation

**No augmentation is applied.** This isolates the progressive unfreeze contribution:
- P.30.2 vs P.30.1 is a controlled comparison --- only encoder unfreezing changes
- Adding augmentation would confound the analysis of unfreezing effects
''',

    'vR.P.30.3': '''---

## 5. Training

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Loss** | **Focal (alpha=0.25, gamma=2.0) + Dice** | Focal focuses on hard pixels; Dice for overlap; tests if hard-example mining helps |
| **Optimizer** | Adam (weight_decay=1e-5) | Same as P.30 baseline; only loss function changes |
| **LR** | 1e-3 | Standard for decoder-only training |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) | Halves LR when val loss plateaus |
| **Early stopping** | patience=7, monitor=val_loss | Standard patience |
| **Epochs** | 25 max | Same as P.30 baseline for controlled comparison |
| **Batch size** | 16 | Largest stable batch for 384x384x3 on T4 GPU (16 GB VRAM) |
| **Image size** | 384x384 | Balances spatial detail for localization vs. GPU memory |
| **Attention** | CBAM (reduction=16, kernel=7) | Standard CBAM hyperparameters from Woo et al. (2018) |

### Why Focal + Dice?

Focal loss (alpha=0.25, gamma=2.0) focuses training on hard-to-classify pixels by
down-weighting easy examples. Combined with Dice loss for overlap maximization.
Tests whether hard-example mining improves over standard BCE+Dice from P.30/P.30.1.
- alpha=0.25: standard value balancing foreground/background
- gamma=2.0: standard focusing parameter from Lin et al. (2017)

### Data Augmentation

**No augmentation is applied.** This isolates the loss function contribution:
- P.30.3 vs P.30 is a controlled comparison --- only loss function changes
- Adding augmentation would confound the analysis
''',

    'vR.P.30.4': '''---

## 5. Training

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Loss** | BCE + Dice (combined) | Same as P.30.1 for controlled comparison |
| **Optimizer** | Adam (weight_decay=1e-5) | Same as P.30.1; only augmentation changes |
| **LR** | 1e-3 | Standard for decoder-only training |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) | Halves LR when val loss plateaus |
| **Early stopping** | patience=10, monitor=val_loss | Higher patience to match P.30.1 for fair comparison |
| **Epochs** | 50 max | Matches P.30.1 for controlled comparison |
| **Batch size** | 16 | Largest stable batch for 384x384x3 on T4 GPU (16 GB VRAM) |
| **Image size** | 384x384 | Balances spatial detail for localization vs. GPU memory |
| **Attention** | CBAM (reduction=16, kernel=7) | Standard CBAM hyperparameters from Woo et al. (2018) |

### Data Augmentation

Geometric-only augmentation is applied to training data:

| Augmentation | Probability | Parameters |
|-------------|-------------|------------|
| HorizontalFlip | 0.5 | --- |
| VerticalFlip | 0.3 | --- |
| RandomRotate90 | 0.5 | --- |
| ShiftScaleRotate | 0.3 | shift=0.05, scale=0.05, rotate=10deg |

**Why geometric-only?** Intensity augmentations (brightness, contrast, blur, noise) are
deliberately excluded because they would corrupt the ELA forensic signal. ELA maps encode
compression artifact differences, and adding pixel-level noise or brightness shifts would
destroy the very signal the model is trained to detect.

Applied to both image and mask simultaneously using Albumentations to maintain alignment.
''',
}


def replace_wandb_init(source, new_init):
    """Replace the wandb.init(...) block in source with new_init."""
    start = source.find('wandb.init(')
    if start == -1:
        return source
    # Find the matching closing paren
    depth = 0
    end = start
    for i in range(start, len(source)):
        if source[i] == '(':
            depth += 1
        elif source[i] == ')':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return source[:start] + new_init + source[end:]


for version in NOTEBOOKS:
    fname = f'{version} Image Detection and Localisation.ipynb'
    path = os.path.join(FINAL_DIR, fname)

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # --- 1. Update W&B metadata in cell 2 ---
    c2 = ''.join(nb['cells'][2]['source'])
    new_wandb = WANDB_MAP.get(version)
    if new_wandb:
        c2 = replace_wandb_init(c2, new_wandb)
    nb['cells'][2]['source'] = [c2]

    # --- 2. Update Cell 13 (training config + augmentation + hyperparameter rationale) ---
    if version in CELL13_FINAL:
        nb['cells'][13]['source'] = [CELL13_FINAL[version]]

    # --- 3. Write output ---
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f'Updated: {fname}')

print('\nDone!')
