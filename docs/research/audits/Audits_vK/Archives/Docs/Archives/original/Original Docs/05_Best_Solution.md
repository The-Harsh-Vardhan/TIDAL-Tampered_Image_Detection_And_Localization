# 5. Best Solution: Architecture & Training Strategy

## 5.1 Recommended Architecture

### Primary Choice: U-Net with EfficientNet-B1 Encoder + SRM Forensic Preprocessing

After evaluating all viable solutions (see Document 02), the recommended architecture is a **dual-input U-Net** built with `segmentation_models_pytorch` (SMP), with forensic noise features from SRM filters concatenated with the RGB input.

**Why this specific combination**:

| Factor | Justification |
|--------|---------------|
| **U-Net via SMP** | Battle-tested, clean API, guaranteed to work, extensive pre-trained encoders |
| **EfficientNet-B1** | Best accuracy/efficiency tradeoff for T4 GPU; 7.8M params; pre-trained on ImageNet |
| **SRM Preprocessing** | Suppresses semantic content, reveals forensic noise artifacts — the single biggest performance boost for tampering detection |
| **1-week timeline** | SMP reduces architecture implementation from days to minutes; lets us focus on the forensic-specific components |

### Alternative (If Ahead of Schedule): SegFormer-B1 Dual-Stream
If the primary implementation is completed early, an upgrade path exists to SegFormer-B1 with dual RGB + noise streams. This is closer to SOTA but adds implementation complexity.

---

## 5.2 Architecture Design

### 5.2.1 High-Level Architecture Diagram

```
                    Input Image (H×W×3, RGB)
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
        ┌──────────────┐    ┌──────────────────┐
        │  RGB Stream   │    │  SRM Filter Bank  │
        │  (3 channels) │    │  (30 high-pass    │
        │               │    │   kernels, fixed)  │
        └──────┬───────┘    └────────┬───────────┘
               │                     │
               │              ┌──────▼──────┐
               │              │ Noise Residual│
               │              │ (H×W×30)     │
               │              └──────┬───────┘
               │                     │
               │              ┌──────▼──────┐
               │              │ 1×1 Conv     │
               │              │ 30 → 3 ch    │
               │              └──────┬───────┘
               │                     │
               └──────────┬──────────┘
                          ▼
                   Concatenation
                   (H×W×6 channels)
                          │
                ┌─────────▼─────────┐
                │   U-Net Encoder    │
                │  (EfficientNet-B1) │
                │  in_channels=6     │
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │   U-Net Decoder    │
                │  (Skip Connections)│
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │  Segmentation Head │
                │  1×1 Conv → Sigmoid│
                └─────────┬─────────┘
                          │
                          ▼
              Tampering Mask (H×W×1)
              (probability per pixel)
```

### 5.2.2 Component Details

#### SRM Filter Bank (Fixed, Non-Trainable)
The Spatial Rich Model provides 30 handcrafted high-pass kernels (5×5) that extract noise residuals by computing the difference between each pixel and its neighbors.

**Implementation**: A `nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)` layer with weights initialized to SRM kernels and `requires_grad=False`. This layer:
- Suppresses high-amplitude semantic content (shapes, colors, objects)
- Amplifies low-amplitude noise patterns
- Reveals local variance mismatches at splicing boundaries
- Exposes different noise floors between authentic and tampered regions

#### Channel Reduction (1×1 Conv)
Reduces the 30 SRM channels to 3 channels via a learnable `nn.Conv2d(30, 3, kernel_size=1)`. This:
- Creates a compact forensic feature representation
- Allows the network to learn which noise residuals are most discriminative
- Keeps the encoder input to 6 channels (3 RGB + 3 noise), which is manageable

#### U-Net with EfficientNet-B1 Encoder (via SMP)
```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="efficientnet-b1",
    encoder_weights="imagenet",
    in_channels=6,  # 3 RGB + 3 compressed SRM
    classes=1,       # single-channel binary mask
    activation=None  # raw logits; sigmoid applied in loss/inference
)
```

**Key**: Setting `in_channels=6` — SMP automatically adapts the first encoder layer to accept 6 channels while preserving the pre-trained weights for the remaining layers.

#### Segmentation Head
Outputs raw logits (no activation). Sigmoid is applied:
- During training: inside the loss function (BCE with logits is numerically stable)
- During inference: explicitly to get probability maps

---

## 5.3 Loss Function Strategy

### 5.3.1 The Class Imbalance Problem

In tampered images, typically **less than 5% of pixels** are actually manipulated. With standard Binary Cross-Entropy (BCE), the model can achieve >95% accuracy by simply predicting "authentic" for every pixel — producing empty masks with zero localization ability.

### 5.3.2 Hybrid Loss: BCE + Dice + Edge

$$L_{total} = \alpha \cdot L_{BCE} + \beta \cdot L_{Dice} + \gamma \cdot L_{Edge}$$

**Recommended weights**: $\alpha = 1.0$, $\beta = 1.0$, $\gamma = 0.5$

#### Binary Cross-Entropy (with logits)
$$L_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\sigma(p_i)) + (1 - y_i) \log(1 - \sigma(p_i))]$$

**Role**: Global pixel distribution matching. Ensures the model learns the overall statistics of tampering vs. authenticity.

#### Dice Loss
$$L_{Dice} = 1 - \frac{2 \sum P \cdot G + \epsilon}{\sum P + \sum G + \epsilon}$$

Where $P$ is the predicted mask (after sigmoid), $G$ is the ground truth, and $\epsilon$ is a smoothing factor (typically 1.0).

**Role**: Region overlap optimization. Inherently robust to class imbalance because background pixels don't contribute to the numerator. Forces the model to **prioritize the minority tampered class**.

#### Edge Loss
$$L_{Edge} = L_{BCE}(\text{pred\_edges}, \text{gt\_edges})$$

Ground truth edges are derived by computing the morphological gradient of the tampering mask (dilation - erosion).

**Role**: Boundary sharpness. Forces the model to produce crisp, well-defined boundaries around tampered regions instead of blurry probabilistic halos.

### 5.3.3 Why This Combination?

| Loss | What It Optimizes | Without It... |
|------|-------------------|---------------|
| BCE | Global distribution | Dice alone can produce noisy masks |
| Dice | Region overlap (F1) | Model predicts mostly zeros (empty masks) |
| Edge | Boundary precision | Mask boundaries are blurry and imprecise |

---

## 5.4 Training Configuration

### 5.4.1 Hyperparameters

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| **Optimizer** | AdamW | Standard for transformer/modern architectures; decoupled weight decay |
| **Learning Rate** | 1e-4 (encoder), 1e-3 (decoder) | Differential LR: frozen pre-trained features need smaller updates |
| **Weight Decay** | 1e-4 | Prevents overfitting on small dataset |
| **LR Scheduler** | CosineAnnealingWarmRestarts | Smooth decay with periodic warm restarts; avoids LR plateau issues |
| **Warmup** | 5% of total steps | Stabilizes early training when encoder weights are adapting to 6-channel input |
| **Batch Size** | 4 (micro-batch) × 4 (accumulation) = 16 effective | Fits in T4 16GB VRAM; accumulation simulates larger batch |
| **Input Resolution** | 512×512 | Preserves high-frequency forensic artifacts; standard for segmentation |
| **Epochs** | 50 (with early stopping, patience=10) | Typically converges in 25-35 epochs on CASIA v2.0 |
| **Checkpoint Criteria** | Best validation Pixel-F1 | F1 is the primary metric; saving by loss can overfit to background pixels |

### 5.4.2 GPU Optimization (T4 Specific)

| Technique | Implementation | Impact |
|-----------|---------------|--------|
| **Mixed Precision (AMP)** | `torch.cuda.amp.autocast()` + `GradScaler()` | ~2× faster training, ~50% VRAM reduction |
| **Gradient Accumulation** | Update every 4 micro-batches | Effective batch 16 fits in 16GB VRAM |
| **Pin Memory** | `DataLoader(pin_memory=True)` | Faster CPU→GPU data transfer |
| **Num Workers** | `DataLoader(num_workers=2)` | Parallel data loading (Colab optimal: 2) |
| **Deterministic Seeds** | PyTorch, NumPy, random, CUDA | Reproducible results |

### 5.4.3 Training Loop Pseudocode

```
for epoch in range(max_epochs):
    model.train()
    for i, (images, masks) in enumerate(train_loader):
        with autocast():
            # Forward pass
            srm_features = srm_layer(images)          # Extract noise residuals
            compressed = channel_reducer(srm_features)  # 30 → 3 channels
            combined = concat(images, compressed)       # 6-channel input
            predictions = model(combined)               # U-Net forward
            loss = hybrid_loss(predictions, masks)      # BCE + Dice + Edge
            loss = loss / accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Validation
    val_f1 = evaluate(model, val_loader)
    if val_f1 > best_f1:
        save_checkpoint(model)
        best_f1 = val_f1

    scheduler.step()
```

---

## 5.5 Inference Pipeline

```
Input Image → SRM Filter → Channel Reduction → Concat with RGB → U-Net → Sigmoid → Threshold (0.5) → Binary Mask
```

At inference time:
1. Apply the same preprocessing (resize to 512×512, normalize)
2. Pass through SRM + U-Net to get probability map
3. Apply sigmoid to get per-pixel probabilities [0, 1]
4. Threshold at 0.5 for binary mask (or use optimal threshold from validation)
5. **Image-level detection**: If any pixel exceeds threshold → image is tampered

---

## 5.6 Decision Rationale: Why Not the Alternatives?

| Alternative | Why Not |
|-------------|---------|
| **Plain U-Net (RGB only)** | Missing forensic features = ~15-20% lower F1; semantic bias problem |
| **MVSS-Net++** | No clean library implementation; custom dual-branch code is complex and error-prone in 1 week |
| **SegFormer-B1** | Not in SMP; requires custom HuggingFace integration + fusion logic; higher risk of implementation bugs |
| **TruFor** | Noiseprint++ requires separate pre-trained model; pipeline too complex for timeline |
| **DeepLabV3+ instead of U-Net** | Marginal gain; ASPP module uses more VRAM; U-Net skip connections are better for fine-grained localization |
| **ResNet-34 instead of EfficientNet-B1** | EfficientNet-B1 achieves better accuracy with fewer parameters; compound scaling is more VRAM-efficient |

---

## 5.7 Expected Performance Targets

Based on published benchmarks and the chosen architecture:

| Metric | Conservative Target | Optimistic Target | SOTA Reference |
|--------|--------------------|--------------------|----------------|
| **Pixel-F1 (CASIA v2.0)** | 55-65% | 70-75% | 94.1% (VAAS) |
| **Pixel-IoU (CASIA v2.0)** | 45-55% | 60-68% | 85.1% (VASLNet) |
| **Image-Level Accuracy** | 80-85% | 88-92% | ~95% (MVSS-Net++) |
| **AUC-ROC** | 0.80-0.85 | 0.88-0.93 | ~0.95 |

**Note**: Conservative targets are achievable with the basic setup. Optimistic targets require tuned augmentation, careful learning rate scheduling, and potentially longer training. Both are respectable results for a 1-week project.

---

## 5.8 Upgrade Path (If Time Permits)

If the primary implementation is completed ahead of schedule, these improvements can be applied in order of impact:

| Priority | Enhancement | Expected Gain | Effort |
|----------|------------|---------------|--------|
| 1 | **BayarConv** (learnable) alongside SRM (fixed) | +3-5% F1 | 2-3 hours |
| 2 | **Edge supervision** auxiliary branch | +2-4% F1 | 3-4 hours |
| 3 | **Test-time augmentation** (TTA: flip + rotate) | +1-3% F1 | 1 hour |
| 4 | **SegFormer-B1** encoder replacement | +3-7% F1 | 4-6 hours |
| 5 | **COVERAGE** evaluation for bonus points | Bonus credit | 1-2 hours |
| 6 | **Robustness table** (JPEG, noise, resize) | Bonus credit | 2-3 hours |
