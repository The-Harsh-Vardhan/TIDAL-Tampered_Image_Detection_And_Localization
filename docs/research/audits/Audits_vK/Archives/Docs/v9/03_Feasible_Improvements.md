# 03 — Feasible Improvements

## Purpose

Evaluate every candidate improvement that has been proposed across Docs8, Audit8 Pro, and external resources. For each improvement, provide:

1. Technical explanation
2. Implementation difficulty
3. Expected performance improvement
4. Feasibility within Colab/Kaggle constraints
5. Decision: Approved / Deferred / Rejected

---

## Category 1: Architecture Improvements

### 1.1 Learned Image-Level Classification Head

**Explanation:** Replace the heuristic `max(prob_map)` detector with a learned binary classification branch. Add a global average pooling layer on the encoder bottleneck features, followed by an FC layer and sigmoid activation. Train with multi-task loss: `total_loss = segmentation_loss + λ * classification_loss`.

**Implementation difficulty:** Easy. Requires adding ~3 lines to model definition, one additional loss term, and minor training loop modification.

```python
# In model definition:
self.cls_head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(encoder_out_channels, 1)
)

# In forward:
cls_logits = self.cls_head(encoder_features[-1])
return seg_logits, cls_logits
```

**Expected benefit:** Moderate-High. Should improve image-level AUC from ~0.87 to 0.90+. Directly addresses Audit8 Pro's strongest architectural criticism.

**Colab feasibility:** Trivial. Adds negligible parameters (~512 weights for the FC layer).

**Decision: ✅ APPROVED**

---

### 1.2 DeepLabV3+ Architecture Comparison

**Explanation:** Train the same pipeline with `smp.DeepLabV3Plus(encoder_name='resnet34')` instead of `smp.Unet(...)`. DeepLabV3+ uses atrous spatial pyramid pooling to capture multi-scale context, which may help with larger tampered regions.

**Implementation difficulty:** Easy. One-line change in SMP model instantiation. Same training pipeline.

**Expected benefit:** Low-Moderate. May improve large-region detection. Unlikely to solve copy-move (same RGB limitation). Primary value is providing comparison evidence for architecture justification.

**Colab feasibility:** Same memory footprint as U-Net with ResNet34.

**Decision: ✅ APPROVED** (as a comparison experiment, not necessarily as a replacement)

---

### 1.3 EfficientNet Encoder Swap

**Explanation:** Replace ResNet34 encoder with EfficientNet-B0. Slightly more parameter-efficient with compound scaling.

**Implementation difficulty:** Easy. One-line SMP change.

**Expected benefit:** Low. Marginal. ResNet34 vs EfficientNet-B0 differences are unlikely to be meaningful compared to loss, augmentation, or input channel changes.

**Colab feasibility:** EfficientNet-B0 uses fewer parameters than ResNet34.

**Decision: ⏸️ DEFERRED** — Low expected impact. Not worth the experiment budget when more impactful changes are available. Can be tested after v9 if time permits.

---

### 1.4 Transformer Encoder (SegFormer / MiT-B0)

**Explanation:** Replace CNN encoder with a vision transformer. Transformers capture global context through self-attention, potentially improving long-range forensic reasoning.

**Implementation difficulty:** Medium-Hard. May require custom SMP integration or HuggingFace model loading. Training dynamics may differ (different LR schedules, warmup requirements).

**Expected benefit:** Uncertain. Theoretical advantage for global context, but no evidence that this matters for CASIA-scale localization. Copy-move improvement uncertain — depends on whether the bottleneck is context or input signal.

**Colab feasibility:** MiT-B0 is T4-compatible but may require different training hyperparameters, increasing experiment cost.

**Decision: ❌ REJECTED for v9** — Too uncertain for the implementation cost. Deferred to future research.

---

### 1.5 Encoder Freezing / Warmup

**Explanation:** Freeze encoder for the first 2 epochs to protect pretrained BatchNorm statistics from large initial gradients.

**Implementation difficulty:** Easy. Add freeze/unfreeze logic around the epoch loop.

**Expected benefit:** Low-Moderate. May improve early training stability, especially with batch-size-4 and new pos_weight.

**Colab feasibility:** Zero overhead.

**Decision:** Already implemented in v8 as an optional CONFIG flag. **No additional action needed.**

---

## Category 2: Input & Preprocessing Improvements

### 2.1 ELA (Error Level Analysis) Auxiliary Channel

**Explanation:** Compute an ELA map for each image by re-saving at a fixed JPEG quality (e.g., QF=90), then taking the absolute difference. This highlights regions with different compression histories — exactly the signal that copy-move boundaries produce.

Feed the model 4 channels (RGB + ELA grayscale) instead of 3.

```python
def compute_ela(image, quality=90):
    # Encode to JPEG at specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    # ELA = absolute difference
    ela = cv2.absdiff(image, decoded)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    return ela_gray
```

**Implementation difficulty:** Easy-Medium. Requires ELA computation in the dataset `__getitem__`, changing `in_channels=4` in the model, and adjusting normalization. The ELA computation adds ~2ms per image, which is negligible in the data pipeline.

**Expected benefit:** Moderate-High for copy-move. ELA directly reveals re-compression artifacts at paste boundaries — the forensic signal that RGB alone cannot provide. Multiple external resources (Resource 07, Resource 08) confirm ELA's value for tamper detection.

**Colab feasibility:** Adds one channel. Memory increase is negligible (~1/4 of input tensor size increase). Compute overhead is ~2ms/image. The first conv layer needs `in_channels=4` but SMP supports arbitrary input channels.

**Key consideration:** ImageNet pretrained weights expect 3 channels. With 4 channels, the first conv layer for channel 4 is randomly initialized. This is standard practice and handled by SMP's `in_channels` parameter.

**Decision: ✅ APPROVED**

---

### 2.2 SRM (Spatial Rich Model) Noise Residuals

**Explanation:** Apply a set of high-pass filters (SRM kernels) to extract noise residual maps that reveal microscopic manipulation traces. SRM was originally designed for steganalysis and has been adapted for forgery detection.

**Implementation difficulty:** Medium-Hard. Requires implementing SRM filter bank (30 kernels), deciding on channel reduction strategy, and modifying the input pipeline. More complex than ELA.

**Expected benefit:** High for copy-move and subtle manipulations. SRM captures noise-level inconsistencies that are invisible in RGB and ELA. However, the benefit is partially redundant with ELA for boundary artifacts.

**Colab feasibility:** Feasible but adds computational overhead. The filter bank processing adds ~10-20ms per image once SRM kernels are implemented as conv layers.

**Decision: ⏸️ DEFERRED** — ELA is a simpler first step that covers much of the same benefit. If ELA proves insufficient after v9, SRM is the natural next step.

---

### 2.3 CbCr Chrominance Channels

**Explanation:** Convert RGB to YCbCr and use Cb and Cr chrominance channels as auxiliary inputs. Some forgeries introduce chrominance inconsistencies that are difficult to see in RGB.

**Implementation difficulty:** Easy. Simple color space conversion in preprocessing.

**Expected benefit:** Low-Moderate. CbCr discontinuities exist in some forgeries (especially splicing from different cameras) but CASIA images may not exhibit strong chrominance artifacts.

**Colab feasibility:** Trivial overhead.

**Decision: ⏸️ DEFERRED** — Interesting but lower priority than ELA. Could be tested as an ablation after v9 if needed.

---

### 2.4 Multi-Scale Training

**Explanation:** Train with random image resolutions (e.g., 256, 384, 512) across batches. This forces the model to learn scale-invariant features.

**Implementation difficulty:** Medium. Requires dynamic resizing in the data pipeline and handling variable-size batches (or resizing within each batch to the same random scale).

**Expected benefit:** Low-Moderate for CASIA. The dataset uses consistent image sizes and the model already sees 384×384. Multi-scale training helps more when test images vary significantly in resolution.

**Colab feasibility:** Training at 512×512 may hit T4 memory limits with batch-size-4. Would require reducing batch size or using gradient checkpointing.

**Decision: ⏸️ DEFERRED** — The expected benefit does not justify the implementation complexity or memory risk for Colab. Better suited for a system processing real-world images of varying resolutions.

---

### 2.5 Multi-Scale Inference (Test-Time Augmentation)

**Explanation:** At inference time, process the image at multiple scales (e.g., 0.75×, 1.0×, 1.25×), predict masks at each scale, resize back to original dimensions, and average.

**Implementation difficulty:** Easy-Medium. Add a TTA wrapper around the model inference.

**Expected benefit:** Moderate. Typically improves segmentation metrics by 1-3% by averaging out scale-sensitive predictions. However, it increases inference time proportionally to the number of scales.

**Colab feasibility:** Increases inference time by 3× for 3 scales. Acceptable for evaluation but not practical for real-time use.

**Decision: ⏸️ DEFERRED** — Nice-to-have but not assignment-critical. Can be added as a bonus if v9 base results are strong. The evaluation pipeline should remain clean and interpretable.

---

## Category 3: Loss Function Improvements

### 3.1 Focal Loss (Replace BCE in BCE+Dice)

**Explanation:** Replace BCEWithLogitsLoss with Focal Loss. Focal Loss down-weights easy examples and focuses gradient on hard-to-classify pixels, which is especially useful for small tampered regions.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
```

**Implementation difficulty:** Easy. FocalLoss is already defined in v8 notebooks as an alternative. Just swap in the combined loss.

**Expected benefit:** Moderate. May help with small-region detection (the model currently struggles with <2% mask area). However, v8 already uses pos_weight which addresses part of the same imbalance problem. The marginal benefit over pos_weight+Dice is uncertain.

**Colab feasibility:** Zero overhead.

**Decision: ⏸️ DEFERRED** — v8 already has pos_weight + per-sample Dice. Adding Focal Loss on top risks over-tuning the loss function. Better to evaluate v9 results with current loss first, then consider Focal Loss if small-region performance is still poor. The FocalDiceLoss implementation already exists in v8 for easy activation.

---

### 3.2 Tversky Loss

**Explanation:** Generalizes Dice Loss with separate weights for false positives and false negatives. Setting β > α increases recall.

**Implementation difficulty:** Easy. Already implemented in v8 notebooks as TverskyDiceLoss.

**Expected benefit:** Low-Moderate. Similar to Focal+Dice in intent. Most useful when recall is significantly lower than precision.

**Colab feasibility:** Zero overhead.

**Decision: ⏸️ DEFERRED** — Same rationale as Focal Loss. The current loss stack is already strong. Keep as an ablation option.

---

### 3.3 Auxiliary Edge Loss

**Explanation:** Add an auxiliary loss that specifically penalizes boundary prediction errors. Extract boundaries from ground truth masks using morphological operations or Canny edge detection, then add a boundary-focused BCE loss.

```python
def compute_edge_mask(mask, kernel_size=3):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=kernel_size//2).clamp(0, 1)
    eroded = 1 - F.conv2d(1 - mask, kernel, padding=kernel_size//2).clamp(0, 1)
    edge = dilated - eroded
    return edge.clamp(0, 1)

# In loss:
edge_gt = compute_edge_mask(target)
edge_loss = F.binary_cross_entropy_with_logits(pred * edge_gt, target * edge_gt)
total_loss = seg_loss + 0.5 * edge_loss
```

**Implementation difficulty:** Easy-Medium. Requires edge extraction (morphological dilation/erosion is simplest) and an additional loss term with a weighting hyperparameter.

**Expected benefit:** Moderate. Directly targets the boundary quality that Audit8 Pro and external resources (EMT-Net, ME-Net) emphasize. Should improve Boundary F1 and may help with copy-move boundaries specifically.

**Colab feasibility:** Negligible overhead. Edge computation is a single conv operation.

**Decision: ✅ APPROVED**

---

### 3.4 Classification Loss for Dual-Task Head

**Explanation:** BCEWithLogitsLoss for the image-level classification branch. The label is 1 if the image is tampered, 0 if authentic.

**Implementation difficulty:** Easy. Standard binary classification loss.

**Expected benefit:** Required for the learned classification head (see §1.1).

**Decision: ✅ APPROVED** (dependent on §1.1 approval)

---

## Category 4: Training Pipeline Optimization

### 4.1 GPU Utilization Improvements (DataLoader Optimization)

**Explanation:** Optimize the DataLoader with persistent workers, pin_memory, and prefetch_factor to reduce GPU idle time during data loading.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    persistent_workers=True,        # Keep workers alive between epochs
    prefetch_factor=2,              # Prefetch 2 batches per worker
    drop_last=True
)
```

**Implementation difficulty:** Easy. Configuration changes only.

**Expected benefit:** Low-Moderate. Reduces training time by 5-15% depending on data pipeline bottleneck. No effect on model quality.

**Colab feasibility:** Fully compatible. `persistent_workers` requires `num_workers > 0`.

**Decision: ✅ APPROVED** — Free performance improvement.

---

### 4.2 Gradient Accumulation Tuning

**Explanation:** Current effective batch size is 16 (batch=4 × accum=4). Consider increasing to 32 (batch=4 × accum=8) for more stable gradients, or decreasing to 8 for more frequent updates.

**Implementation difficulty:** Easy. Change one CONFIG value.

**Expected benefit:** Uncertain. Larger batches give smoother gradients but fewer updates per epoch. The current 16 is already reasonable.

**Decision: ⏸️ DEFERRED** — Current accumulation of 4 is standard. Not worth changing without evidence of gradient instability from v9 training.

---

### 4.3 CosineAnnealingWarmRestarts Scheduler

**Explanation:** Replace ReduceLROnPlateau with CosineAnnealingWarmRestarts. Provides periodic LR increases that can escape local minima.

**Implementation difficulty:** Easy. One-line scheduler swap.

**Expected benefit:** Uncertain. ReduceLROnPlateau is working. Cosine may help in longer training runs (50+ epochs) but v8/v9 training likely runs 30-40 useful epochs.

**Decision: ⏸️ DEFERRED** — ReduceLROnPlateau is producing good results per Docs8 analysis. Keep as an ablation option.

---

## Category 5: Evaluation Improvements

### 5.1 Boundary F1 Metric

**Explanation:** Measure prediction quality specifically at tampered region boundaries using dilated boundary extraction and boundary-specific precision/recall/F1.

**Implementation difficulty:** Easy-Medium. Requires boundary extraction (morphological operations or `skimage.segmentation.find_boundaries`) and boundary-masked metric computation.

**Expected benefit:** High informational value. Boundary F1 directly measures localization precision, which is what distinguishes a useful forensic tool from one that just detects "roughly the right region."

**Colab feasibility:** Negligible overhead (evaluation-time only).

**Decision: ✅ APPROVED**

---

### 5.2 Precision-Recall Curves

**Explanation:** Plot PR curves for both pixel-level and image-level tasks across all thresholds.

**Implementation difficulty:** Easy. Use sklearn's precision_recall_curve.

**Expected benefit:** High informational value. Shows the full operating characteristic rather than a single-threshold snapshot.

**Colab feasibility:** Negligible.

**Decision: ✅ APPROVED**

---

### 5.3 Multi-Seed Validation

**Explanation:** Train v9 with 3 different seeds and report mean ± std of key metrics. Establishes confidence intervals for all conclusions.

**Implementation difficulty:** High in terms of compute (3× training time). Low in terms of code changes.

**Expected benefit:** High for credibility. Single-seed results are anecdotal. Three seeds establish whether improvements are robust.

**Colab feasibility:** Requires 3× the GPU time. Feasible if each run is ~2 hours on T4 (6 hours total).

**Decision: ✅ APPROVED** — 3 seeds minimum. If compute is tight, run the primary experiment with 3 seeds and comparison experiments (DeepLabV3+) with 1 seed.

---

### 5.4 Mask Randomization Test

**Explanation:** Shuffle ground truth masks across test images and re-evaluate. If the model has learned genuine localization, performance should drop dramatically. If the model exploits shortcuts (e.g., always predicting the same generic region), performance stays similar.

**Implementation difficulty:** Easy. Shuffle mask assignments and re-run evaluation.

**Expected benefit:** High for validation credibility. Provides a concrete shortcut falsification test.

**Decision: ✅ APPROVED**

---

### 5.5 pHash Near-Duplicate Check

**Explanation:** Compute perceptual hashes for all images and check for near-duplicates across train/val/test splits. If found, group them into the same split to prevent content leakage.

**Implementation difficulty:** Easy. Requires `imagehash` library and O(n²) comparison (feasible for 12K images).

**Expected benefit:** High for data integrity credibility. Directly addresses Audit8 Pro's leakage concern.

**Colab feasibility:** Runs in minutes on CPU.

**Decision: ✅ APPROVED**

---

### 5.6 Cross-Dataset Evaluation

**Explanation:** Train on CASIA, evaluate zero-shot on Coverage or CoMoFoD.

**Implementation difficulty:** Medium. Requires downloading, preprocessing, and adapting evaluation code for a new dataset with different annotation formats.

**Expected benefit:** High for generalization credibility. But expectations should be low — significant performance drop is expected and normal.

**Colab feasibility:** Requires additional dataset download and storage.

**Decision: ⏸️ DEFERRED** — Valuable but not assignment-required. The primary submission should be strong on CASIA first. Cross-dataset is a bonus that adds complexity without directly improving the graded deliverable.

---

## Category 6: Augmentation Improvements

### 6.1 Augmentation Ablation Tracking

**Explanation:** Track which augmentations contribute most to performance by running controlled ablation experiments (remove one augmentation type at a time).

**Implementation difficulty:** Medium. Requires multiple training runs.

**Expected benefit:** High informational value. Validates whether each augmentation is helping or just adding noise to training.

**Colab feasibility:** Requires multiple runs but each is identical duration.

**Decision: ✅ APPROVED** — Run as part of multi-seed experiments when feasible. At minimum, compare: (1) v9 full augmentation vs (2) v9 without photometric augmentation (geometric only, like Run01).

---

### 6.2 Stronger Geometric Augmentation

**Explanation:** Add ShiftScaleRotate, ElasticTransform, or GridDistortion beyond the current HFlip/VFlip/Rotate90.

**Implementation difficulty:** Easy. Albumentations provides all of these.

**Expected benefit:** Low. The current geometric augmentation is already covering the important cases. Elastic/grid distortion can actually harm forensic features by creating artificial boundary artifacts.

**Decision: ❌ REJECTED** — Risk of destroying forensic signal outweighs marginal regularization benefit. The photometric augmentations already address the overfitting problem.

---

## Summary Table

| Improvement | Category | Difficulty | Expected Benefit | Decision |
|---|---|---|---|---|
| Learned classification head | Architecture | Easy | Moderate-High | ✅ Approved |
| DeepLabV3+ comparison | Architecture | Easy | Low-Moderate | ✅ Approved |
| EfficientNet encoder | Architecture | Easy | Low | ⏸️ Deferred |
| Transformer encoder | Architecture | Hard | Uncertain | ❌ Rejected |
| ELA auxiliary channel | Input | Easy-Medium | Moderate-High | ✅ Approved |
| SRM noise residuals | Input | Medium-Hard | High | ⏸️ Deferred |
| CbCr channels | Input | Easy | Low-Moderate | ⏸️ Deferred |
| Multi-scale training | Input | Medium | Low-Moderate | ⏸️ Deferred |
| Multi-scale inference | Input | Easy-Medium | Moderate | ⏸️ Deferred |
| Focal Loss | Loss | Easy | Moderate | ⏸️ Deferred |
| Tversky Loss | Loss | Easy | Low-Moderate | ⏸️ Deferred |
| Auxiliary edge loss | Loss | Easy-Medium | Moderate | ✅ Approved |
| Classification loss | Loss | Easy | Required | ✅ Approved |
| DataLoader optimization | Pipeline | Easy | Low-Moderate | ✅ Approved |
| Gradient accumulation tuning | Pipeline | Easy | Uncertain | ⏸️ Deferred |
| Cosine scheduler | Pipeline | Easy | Uncertain | ⏸️ Deferred |
| Boundary F1 | Evaluation | Easy-Medium | High (info) | ✅ Approved |
| PR curves | Evaluation | Easy | High (info) | ✅ Approved |
| Multi-seed validation | Evaluation | High (compute) | High | ✅ Approved |
| Mask randomization test | Evaluation | Easy | High | ✅ Approved |
| pHash leak check | Evaluation | Easy | High | ✅ Approved |
| Cross-dataset evaluation | Evaluation | Medium | High | ⏸️ Deferred |
| Augmentation ablation | Augmentation | Medium | High (info) | ✅ Approved |
| Stronger geometric augmentation | Augmentation | Easy | Low | ❌ Rejected |
