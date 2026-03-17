# Docs11: Strengths Assessment

An honest enumeration of what the current system does well. Improvements should build on these strengths, not replace them.

---

## 1. Engineering Foundation (Strongest Area)

### 1.1 CONFIG-Driven Pipeline

vK.10.5 centralizes every hyperparameter in a single CONFIG dictionary. This is the cleanest configuration system across all notebook versions:

- Every training parameter is changeable without modifying code
- Loss weights (α, β), scheduler settings, augmentation probabilities, and thresholds are all configurable
- CONFIG is serialized into checkpoints for perfect reproducibility
- CONFIG is logged to W&B for experiment tracking

**Why this matters:** Reviewers can instantly see all design decisions in one place. It also enables systematic hyperparameter sweeps without code edits.

### 1.2 Production-Grade Checkpoint System

The three-file checkpoint scheme (last, best, periodic) with full state persistence is more robust than any other notebook version:

- `last_checkpoint.pt` — every epoch, enables crash recovery
- `best_model.pt` — triggered by tampered-only Dice improvement, prevents metric regression
- `checkpoint_epoch_N.pt` — periodic snapshots for analysis

Each checkpoint stores: model state_dict, optimizer state, scaler state, scheduler state, full training history dict, CONFIG, best_metric, best_epoch. This means training can resume from any point with no loss of information.

The `get_base_model()` unwrapper ensures checkpoints are always saved without `module.` prefixes, making them portable between single-GPU and multi-GPU setups.

### 1.3 Correct Early Stopping Metric

Early stopping uses tampered-only Dice coefficient, not overall accuracy or mixed-set metrics. This directly addresses the metric inflation problem identified across all audits:

- **Mixed-set metrics are misleading:** Including authentic images with zero-area GT masks inflates Dice/IoU/F1 (vK.7.5's Dice=IoU=F1=0.5935 is the poster child for this failure)
- **Accuracy rewards trivial baselines:** Predicting "authentic" for everything gets ~50% accuracy on a balanced dataset
- **Tampered-only Dice** measures what actually matters — how well the model localizes tampered regions

### 1.4 Automatic Mixed Precision

AMP implementation follows PyTorch best practices:

```
with autocast('cuda', enabled=CONFIG['use_amp']):
    cls_logits, seg_logits = model(images)
    loss = ...
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])
scaler.step(optimizer)
scaler.update()
```

- `autocast` wraps the forward pass and loss computation
- `GradScaler` handles loss scaling for numerical stability
- Gradient clipping is applied between `unscale_` and `step` (the correct order)
- AMP is togglable via CONFIG without code changes

---

## 2. Dual-Head Architecture Design

### 2.1 Joint Detection + Localization

The `UNetWithClassifier` architecture produces both image-level classification and pixel-level segmentation from a shared encoder. This design is validated by:

- **Research literature:** Papers P2 (Dual-Task Classification + Segmentation) and Resource 14 (Kaggle notebook) demonstrate that joint training improves both tasks through shared feature learning
- **Practical evidence:** vK.3 achieved 89.9% image-level accuracy with this architecture, confirming the dual-head design is sound
- **Assignment requirement:** The spec explicitly requires both detection ("classify whether an image is tampered") and localization ("generate a pixel-level mask")

### 2.2 Correct Classification Head Placement

The classifier operates on the bottleneck (1024-channel, 16×16 feature map after AdaptiveAvgPool → 1×1). This is architecturally sound because:

- The bottleneck contains the most semantically rich features
- AdaptiveAvgPool eliminates spatial dimensions cleanly
- The classification signal provides implicit regularization for the encoder, which benefits segmentation

This is a significant advantage over v8, which used a heuristic `max(prob_map)` for image-level detection instead of a learned head.

---

## 3. Loss Function Design

### 3.1 Multi-Component Loss

The combined loss `α×FocalLoss(cls) + β×(0.5×BCE + 0.5×Dice)(seg)` addresses multiple problems simultaneously:

| Component | Purpose |
|---|---|
| FocalLoss (classification) | Handles class imbalance between authentic and tampered images. γ=2.0 downweights easy negatives. |
| BCEWithLogitsLoss (segmentation) | Per-pixel supervision — every pixel contributes to the gradient. Numerically stable with logit input. |
| DiceLoss (segmentation) | Region-level overlap optimization — directly optimizes the Dice metric. Handles extreme foreground/background imbalance. |

### 3.2 Configurable Loss Weights

The α/β weights allow rebalancing classification vs segmentation emphasis without code changes. The current setting (α=1.5, β=1.0) slightly favors classification, which makes sense for a dataset where image-level detection is easier and provides a stronger learning signal early in training.

---

## 4. Augmentation Pipeline

### 4.1 Forensics-Relevant Transforms

The augmentation pipeline includes transforms specifically relevant to image forensics:

| Transform | Forensic Relevance |
|---|---|
| ImageCompression(QF=50-90, p=0.25) | Simulates JPEG recompression, which is the primary real-world degradation for tampered images |
| GaussNoise(var=10-50, p=0.25) | Simulates sensor noise and anti-forensic noise injection |
| RandomBrightnessContrast(0.3, p=0.3) | Models lighting variation in spliced regions |
| Affine(translate/scale/rotate, p=0.5) | Models geometric transforms applied to tampered regions |

This is a meaningful improvement over earlier notebooks (vK.3, vK.7.5) that had no forensics-relevant augmentation.

### 4.2 Mask-Aware Augmentation

Albumentations applies identical spatial transforms to both image and mask tensors, ensuring mask alignment is preserved through augmentation. This is critical for segmentation training and is correctly implemented.

---

## 5. Reproducibility Infrastructure

### 5.1 Full Seeding

Seeds are set across all random number generators:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 5.2 Worker Seeding

DataLoader workers are seeded with a `worker_init_fn` to ensure reproducible data loading order. This addresses a subtle source of non-determinism that most notebooks ignore.

### 5.3 Checkpoint-Based Reproducibility

Because CONFIG and full training history are serialized into every checkpoint, any training run can be analyzed post-hoc to understand exactly what happened and why.

---

## 6. Multi-GPU Support

vK.10.5 correctly implements `nn.DataParallel` for Kaggle's 2×T4 GPUs:

- Wrapping is conditional on `torch.cuda.device_count() > 1`
- GPU diagnostics enumerate all GPUs with per-device VRAM reporting
- Batch size auto-scales based on total VRAM across all GPUs
- `get_base_model()` unwrapper handles save/load correctly for both single and multi-GPU

---

## 7. Summary

The strengths of vK.10.5 form a solid foundation. The engineering infrastructure, dual-head architecture, loss design, augmentation pipeline, and reproducibility setup are all sound. The primary gaps (no pretrained encoder, no forensic preprocessing, no advanced evaluation) are additive improvements that build on top of this foundation rather than requiring rewrites. This is the right notebook to iteratively improve.
