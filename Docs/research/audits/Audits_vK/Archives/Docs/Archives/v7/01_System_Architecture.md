# System Architecture

End-to-end pipeline for tampered image detection and localization using a U-Net segmentation model with a pretrained ResNet34 encoder.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│  1. Environment Setup                                   │
│     setup_device() → GPU detection, TF32, cuDNN bench   │
│     CONFIG dictionary → all hyperparameters + flags      │
├─────────────────────────────────────────────────────────┤
│  2. Dataset Loading                                     │
│     Kaggle /kaggle/input/ mount (Kaggle)                │
│     Drive mount + Kaggle API download (Colab)           │
├─────────────────────────────────────────────────────────┤
│  3. Dataset Discovery                                   │
│     Case-insensitive walk → IMAGE/ and MASK/ dirs       │
│     Readability + dimension checks → pair validation    │
│     Unknown forgery type exclusion                      │
├─────────────────────────────────────────────────────────┤
│  4. Dataset Validation                                  │
│     Counts, class balance, sample-load verification     │
│     Data leakage assertion (zero overlap across splits) │
├─────────────────────────────────────────────────────────┤
│  5. Preprocessing & Split                               │
│     Stratified 70/15/15 split                           │
│     split_manifest.json persistence                     │
├─────────────────────────────────────────────────────────┤
│  6. Model Architecture                                  │
│     setup_model() → smp.Unet(resnet34, imagenet)        │
│     Optional DataParallel wrapping                      │
│     Shape verification (dummy forward pass)             │
├─────────────────────────────────────────────────────────┤
│  7. Training Pipeline                                   │
│     BCEDiceLoss + AdamW (differential LR)               │
│     train_one_epoch() + validate_model()                │
│     AMP via GradScaler(enabled=flag)                    │
│     Gradient accumulation (4 steps → effective batch 16)│
│     Early stopping on val Pixel-F1 (patience=10)        │
├─────────────────────────────────────────────────────────┤
│  8. Evaluation                                          │
│     Threshold sweep on validation (0.1–0.9, step 0.02) │
│     Test metrics: pixel-level + image-level             │
│     Mixed / tampered-only / forgery-type views          │
├─────────────────────────────────────────────────────────┤
│  9. Visualization & Explainable AI                      │
│     Training curves, prediction grid, Grad-CAM          │
│     Diagnostic overlays (TP/FP/FN), failure analysis    │
├─────────────────────────────────────────────────────────┤
│  10. Robustness Testing                                 │
│     8 degradation conditions (JPEG, noise, blur, resize)│
│     Same threshold, no retraining                       │
├─────────────────────────────────────────────────────────┤
│  11. Save Artifacts                                     │
│     Checkpoints, results_summary.json, plots            │
│     Optional W&B upload                                 │
└─────────────────────────────────────────────────────────┘
```

---

## Hardware Abstraction Layer

The v6.5 notebooks introduce a hardware abstraction layer that decouples the pipeline from specific GPU configurations.

### `setup_device(config)`

Detects available hardware, enables GPU-specific optimizations, and returns the training device. This function:

- Detects CUDA availability and GPU count
- Enables cuDNN benchmark mode for optimized convolution kernels
- Enables TF32 for matmul and cuDNN on Ampere+ GPUs
- Reports AMP and multi-GPU status based on CONFIG flags
- Falls back gracefully to CPU with a warning

**Why a setup function?** Centralizing hardware detection ensures consistent behavior across Kaggle (T4) and Colab (T4/V100/A100) environments. Without this, GPU-specific flags would be scattered through the notebook.

### `setup_model(config, device)`

Creates the model, optionally wraps in DataParallel, and verifies output shape. This function:

- Instantiates `smp.Unet` using config values (encoder, weights, channels, classes)
- Wraps in `torch.nn.DataParallel` when `config['use_multi_gpu']` is True and multiple GPUs are available
- Reports parameter counts (total and trainable)
- Runs a dummy forward pass to verify output shape matches `(1, 1, H, W)`

**Why DataParallel?** Some Colab instances provide 2× T4 GPUs. Wrapping in DataParallel lets the same notebook scale without code changes. The flag control means single-GPU environments (typical Kaggle) are unaffected.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | U-Net + ResNet34 (SMP library) | Proven encoder-decoder for dense prediction; ImageNet pretrained encoder provides strong feature extraction; assignment-appropriate complexity |
| Input resolution | 384 × 384 | Balances detail preservation with T4 VRAM constraints; 512 would require batch ≤ 2 |
| Loss | BCEDiceLoss (BCE + Dice, smooth=1.0) | BCE provides per-pixel gradients; Dice directly optimizes overlap metric for imbalanced masks |
| Optimizer | AdamW (differential LR) | Encoder (pretrained) uses lower LR (1e-4); decoder (random init) uses higher LR (1e-3) |
| Image-level detection | Top-k mean of pixel probabilities (top 1%) | Lightweight heuristic; avoids needing a separate classification head |
| Split | 70/15/15 stratified | Standard split; stratification preserves authentic/tampered ratio |
| Hardware flags | `use_amp`, `use_multi_gpu`, `use_wandb` | Decouples optional features from core pipeline; notebook runs correctly with all flags False |
| Environments | Kaggle + Colab variants | Kaggle version uses `/kaggle/input/` mount; Colab version uses Drive + Kaggle API download |

---

## Environment Specifications

| Property | Kaggle | Colab |
|---|---|---|
| GPU | T4 (15 GB VRAM) | T4 / V100 / A100 (varies) |
| Dataset access | Pre-mounted at `/kaggle/input/` | Kaggle API download to Drive |
| Output storage | `/kaggle/working/` | Google Drive |
| Python | 3.10+ | 3.10+ |
| Secrets | Kaggle Secrets API | Colab Secrets (`userdata.get`) |
| Notebook | `tamper_detection_v6.5_kaggle.ipynb` | `tamper_detection_v6.5_colab.ipynb` |

---

## CONFIG Dictionary

All hyperparameters, model settings, and feature flags are centralized in a single CONFIG dictionary at the top of the notebook:

```python
CONFIG = {
    # ── Data ──
    'image_size': 384,
    'batch_size': 4,
    'num_workers': 2,
    'train_ratio': 0.70,

    # ── Model ──
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,

    # ── Optimizer ──
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,

    # ── Training ──
    'max_epochs': 50,
    'patience': 10,
    'accumulation_steps': 4,
    'max_grad_norm': 1.0,

    # ── Feature Flags ──
    'use_amp': True,
    'use_multi_gpu': True,
    'use_wandb': False,

    # ── Reproducibility ──
    'seed': 42,
}
```

**Why a centralized config?** Prevents magic numbers scattered through the notebook. Every tunable parameter has a single source of truth. Changing a value in CONFIG propagates throughout training, evaluation, and robustness testing. If asked in an interview: "Where do I change the learning rate?" — the answer is always "the CONFIG dictionary."
