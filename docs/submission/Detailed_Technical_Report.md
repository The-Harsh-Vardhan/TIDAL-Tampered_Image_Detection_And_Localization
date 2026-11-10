# Detailed Technical Report: Systematic Ablation Study for Image Tampering Detection and Localization

**Project:** Image Tampering Detection & Localization
**Dataset:** CASIA v2.0 (12,614 images: 7,491 Authentic + 5,123 Tampered)
**Total Experiments:** 26+ across 3 tracks (ETASR classification, legacy vK, pretrained localization)
**Best Localization Result:** Pixel F1 = 0.7329, Image Accuracy = 87.53% (vR.P.15 --- Multi-Quality ELA)
**Best Classification Result:** 90.23% accuracy (vR.1.6 --- Deeper ETASR CNN)

---

## Abstract

This report documents a systematic ablation study for pixel-level image forgery detection and localization on the CASIA v2.0 benchmark. The project explored 4 experimental lineages over 26+ controlled experiments, evolving from documentation-first planning through Kaggle notebook reproduction, research paper baseline recreation, and finally, a pretrained UNet segmentation pipeline.

The final system uses a UNet encoder-decoder with a frozen ResNet-34 encoder (ImageNet pretrained), Error Level Analysis (ELA) preprocessing, and CBAM attention modules. Through strict single-variable ablation (one change per experiment), we established that: **input representation dominates all other factors** --- switching from RGB to ELA was worth +23.74 percentage points (pp) in Pixel F1, more than all other improvements combined. Multi-quality ELA (Q=75/85/95) achieved the best overall Pixel F1 of 0.7329, while CBAM attention was the most parameter-efficient improvement (+3.57pp for 11K parameters).

---

## 1. Project Evolution: Four Lineages

This project did not follow a straight line. Each failed approach taught lessons that shaped the next attempt.

### 1.1 v0x --- Documentation-First (Failed)

The project began with an extensive literature review. Documentation was written before any code: architecture specs, dataset analyses, training plans --- all generated with AI assistance (Gemini Deep Research, ChatGPT, Codex). The philosophy was "get the docs right, then build notebooks from them."

This failed when too many ideas from research papers were added simultaneously, leading to untested complexity. There was little to no executable code until late versions.

**Lesson:** Documentation without experimentation leads to untested assumptions. Ideas must be validated incrementally.

### 1.2 vK.x.x --- Kaggle Notebook Reproduction (Failed)

A Kaggle notebook (`image-detection-with-mask`) was discovered that appeared to solve the assignment task: a dual-head UNet with classification and segmentation outputs, reporting ~89.75% accuracy and Dice ~0.57 on CASIA.

An audit revealed two fatal problems:
- **Data leakage:** The test CSV pointed to the validation set. Reported "test accuracy" was actually validation accuracy.
- **Segmentation failure:** On a dataset where ~40% of images are authentic (all-zero masks), a model predicting all-zero masks scores Dice ~0.58. The segmentation head was not learning.

After fixing the leakage, training from scratch on ~10K RGB images with 31.6M parameters proved insufficient. Best honest tampered F1 in this track: 0.4101 (v6.5).

**Lesson:** Always audit for data leakage. Training from scratch on small datasets with large models is insufficient --- the model only learns low-level features.

### 1.3 vR.x.x --- ETASR Classification Track (Archived)

The project pivoted to reproducing a published research paper: "Enhanced Image Tampering Detection using ELA and a CNN" (Nagm et al. 2024, ETASR 9593). This paper reported 96.21% accuracy using a custom 2-layer CNN on ELA images.

Seven honest ablation experiments (vR.1.1 through vR.1.7) were conducted:

| Version | Change | Test Acc | Macro F1 | ROC-AUC | Params | Verdict |
|---------|--------|----------|----------|---------|--------|---------|
| vR.1.1 | Honest baseline | 88.38% | 0.8805 | 0.9601 | 29.5M | Baseline |
| vR.1.2 | Data augmentation | 85.53% | 0.8505 | 0.9011 | 29.5M | REJECTED |
| vR.1.3 | Class weights | 89.17% | 0.8889 | 0.9580 | 29.5M | POSITIVE |
| vR.1.4 | BatchNorm | 88.75% | 0.8852 | 0.9536 | 29.5M | NEUTRAL |
| vR.1.5 | LR scheduler | 88.96% | 0.8873 | 0.9560 | 29.5M | NEUTRAL |
| **vR.1.6** | **Deeper CNN (3rd conv)** | **90.23%** | **0.9004** | **0.9657** | **13.8M** | **POSITIVE** |
| vR.1.7 | Global Avg Pooling | 89.17% | 0.8901 | 0.9495 | 64K | NEUTRAL |

Key findings from this track:
- **Paper gap:** 5.98pp below the claimed 96.21%, never closed
- **Architecture > training tricks:** The deeper CNN (vR.1.6, +1.85pp) outperformed class weights + BatchNorm + LR scheduler combined (+0.58pp) by 3.2x
- **The Flatten-Dense bottleneck is toxic:** It caused augmentation failure (vR.1.2), training instability, and 99%+ parameter concentration. Augmented images activate completely different neurons because Flatten memorizes exact pixel positions
- **Critical limitation:** Classification-only. No pixel-level localization capability.

The track was archived because the assignment requires localization.

### 1.4 vR.P.x.x --- Pretrained Ablation Study (Active)

Armed with lessons from all prior failures, the final lineage established:
- **Pretrained encoder** (ResNet-34 on ImageNet) instead of training from scratch
- **UNet segmentation** for pixel-level localization
- **ELA preprocessing** as input (from ETASR track insight)
- **Single-variable ablation:** exactly one change per experiment
- **Weights & Biases tracking** for all experiments
- **Reproducibility protocol:** seed 42, identical hardware, two verification runs (P.3 and P.10)

---

## 2. Dataset & Preprocessing

### CASIA v2.0

| Property | Value |
|----------|-------|
| Authentic images | 7,491 (59.4%) |
| Tampered images | 5,123 (40.6%) |
| **Total** | **12,614** |
| Tampering types | Splicing, copy-move forgeries |
| Ground truth | Binary masks for each tampered image |
| Source | sagnikkayalcse52 Kaggle dataset (includes proper GT masks) |

### Data Split

Stratified 70/15/15 split using a two-step process:
1. 70/30 split of full dataset (stratified by label)
2. 50/50 split of the 30% holdout into validation and test

| Split | Count | Purpose |
|-------|-------|---------|
| Training | ~8,830 | Model optimization |
| Validation | ~1,892 | Early stopping, LR scheduling |
| Test | ~1,892 | Final reported metrics |

All splits use random seed 42 for reproducibility.

### Preprocessing

- All images resized to **384 x 384** pixels
- Masks resized with nearest-neighbor interpolation, binarized at threshold 0.5
- Per-channel normalization statistics computed from 500 randomly sampled training images

### Data Leakage Lessons

The sagnik dataset variant produced 100% test accuracy --- investigation revealed GT mask images were mixed into the input data. The CNN trivially distinguished masks from photographs. This result was discarded and the finding motivated the strict pipeline audits applied to all subsequent experiments.

---

## 3. Architecture Deep Dive

### 3.1 UNet with ResNet-34 Encoder

The model uses the **Segmentation Models PyTorch (SMP)** library's UNet implementation:

- **Encoder:** ResNet-34 pretrained on ImageNet (21.3M parameters)
- **Decoder:** 5 upsampling blocks with channel dimensions (256, 128, 64, 32, 16), skip connections at 4 resolution levels
- **Output:** Single-channel sigmoid, 384 x 384 pixels

**Freeze strategy:** All encoder convolutional weights are frozen. Only BatchNorm layers are set to training mode (unfrozen scale/shift parameters). This allows the BN statistics to adapt from ImageNet's natural image distribution to the ELA forensic domain while keeping the learned feature hierarchies intact.

- Trainable parameters: ~3.17M (decoder + encoder BN)
- Total parameters: ~24.4M
- Trainable ratio: 13%

This was validated empirically: P.2's aggressive unfreeze (23M trainable) overfit with a 1:2,615 data:param ratio. P.8's progressive unfreeze confirmed the best epoch occurred before any unfreezing stage.

### 3.2 Error Level Analysis (ELA)

ELA is a forensic preprocessing technique that reveals compression inconsistencies:

1. Load the original image as RGB
2. Re-save to an in-memory JPEG buffer at quality level Q
3. Reload the recompressed image
4. Compute pixel-wise absolute difference between original and recompressed
5. Scale brightness so the maximum channel difference maps to 255
6. Convert to grayscale

Authentic regions compress uniformly (low ELA residuals). Tampered regions that were saved at different JPEG quality levels or composited from different sources show inconsistencies (high ELA residuals).

**Single-quality ELA (P.3):** Q=90, converted to RGB. Pixel F1 = 0.6920.

**Multi-Quality ELA (P.15):** Three independent grayscale ELA maps stacked as 3 channels:
- **Q=75** (aggressive): Large residuals, catches strong manipulations. Mean=0.0684, Std=0.0656
- **Q=85** (balanced): Medium residuals. Mean=0.0605, Std=0.0604
- **Q=95** (gentle): Small residuals, sensitive to subtle edits. Mean=0.0402, Std=0.0471

The three channels carry meaningfully different statistics --- lower Q produces larger residuals as expected. This confirms the channels provide distinct, non-redundant forensic information. Multi-Q ELA achieved Pixel F1 = 0.7329 (+4.09pp over single-Q).

### 3.3 CBAM Attention

CBAM (Convolutional Block Attention Module) was injected into all 5 UNet decoder blocks, applied after each decoder stage:

- **Channel attention:** Shared MLP (bottleneck with reduction ratio 16) processes both average-pooled and max-pooled features, combined with sigmoid gating. Teaches the decoder which feature channels are most relevant.
- **Spatial attention:** Concatenated channel-wise average and max pooling, processed by a 7x7 convolution with sigmoid. Teaches the decoder which spatial locations to focus on.

Parameter cost: **11,402 additional parameters** (0.36% of trainable). This produced +3.57pp Pixel F1 --- the highest improvement density of any change (0.313pp per thousand parameters).

### 3.4 Alternative Architectures Tested

| Encoder | Version | Trainable Params | Pixel F1 (RGB) | Result |
|---------|---------|-----------------|----------------|--------|
| ResNet-34 | P.1 | 3.17M | 0.4546 | Baseline |
| ResNet-50 | P.5 | 9.0M | 0.5137 | +0.0591 |
| EfficientNet-B0 | P.6 | 2.24M | 0.5217 | +0.0671 |
| ResNet-34 + ELA | P.3 | 3.17M | **0.6920** | **+0.2374** |

The hierarchy is clear: ELA input on ResNet-34 (3.17M params, F1=0.6920) vastly outperforms ResNet-50 (9M params, F1=0.5137) and EfficientNet-B0 (2.24M params, F1=0.5217) on RGB. Input representation dominates encoder architecture.

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (weight decay 1e-5) |
| Learning rate | 1e-3 (single rate for all parameters) |
| Loss (baseline) | SoftBCEWithLogitsLoss + Binary Dice |
| Loss (ablation) | Focal Loss (alpha=0.25, gamma=2.0) + Dice |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | Patience 7 on validation loss (10 for extended runs) |
| Batch size | 16 |
| Max epochs | 25 (default), 50 (extended training experiments) |
| Mixed precision | AMP with GradScaler |
| TF32 | Enabled on Ampere+ GPUs |
| Hardware | Kaggle T4/P100 GPUs |
| Seed | 42 |

**Loss function note:** Focal+Dice (P.9) was essentially neutral (+0.03pp) compared to BCE+Dice. The Dice component already handles the class imbalance between tampered pixels and background pixels. Focal Loss additionally concentrates predictions near 0/1 extremes, destroying probability calibration (Pixel AUC regressed by -2.05pp).

---

## 5. Complete Experiment Results

### 5.1 Full Results Table

| Version | Modification | Pixel F1 | IoU | Pixel AUC | Img Acc | Img F1 | Img AUC | Verdict |
|---------|-------------|----------|-----|-----------|---------|--------|---------|---------|
| P.0 | Dataset baseline (no proper GT) | 0.3749 | 0.2307 | 0.8486 | 70.63% | 0.6814 | 0.7860 | Baseline |
| P.1 | Proper GT masks | 0.4546 | 0.2942 | 0.8509 | 70.15% | 0.6867 | 0.7785 | Proper Baseline |
| P.1.5 | AMP + TF32 speed opts | 0.4227 | 0.2680 | 0.8560 | 71.05% | 0.7016 | 0.7980 | Neutral |
| P.2 | Aggressive encoder unfreeze | 0.5117 | 0.3439 | 0.8688 | 69.04% | 0.6673 | 0.7196 | Positive |
| **P.3** | **ELA input (replacing RGB)** | **0.6920** | **0.5291** | **0.9528** | **86.79%** | **0.8560** | **0.9502** | **Strong +** |
| P.4 | 4-channel RGB+ELA | 0.7053 | 0.5447 | 0.9433 | 84.42% | 0.8322 | 0.9229 | Neutral |
| P.5 | ResNet-50 encoder (RGB) | 0.5137 | 0.3456 | 0.8828 | 72.00% | 0.7143 | 0.8126 | Positive |
| P.6 | EfficientNet-B0 encoder (RGB) | 0.5217 | 0.3529 | 0.8708 | 70.68% | 0.6950 | 0.7801 | Positive |
| P.7 | Extended training (50 epochs) | 0.7154 | 0.5569 | 0.9504 | 87.37% | 0.8637 | 0.9433 | Positive |
| P.8 | Progressive encoder unfreeze | 0.6985 | 0.5367 | 0.9541 | 87.59% | 0.8650 | 0.9578 | Neutral |
| P.9 | Focal + Dice loss | 0.6923 | 0.5294 | 0.9323 | 87.16% | 0.8606 | 0.9076 | Neutral |
| **P.10** | **CBAM attention in decoder** | **0.7277** | **0.5719** | **0.9573** | 87.32% | 0.8615 | **0.9633** | **Positive** |
| P.12 | Data augmentation | 0.6968 | 0.5347 | 0.9502 | **88.48%** | **0.8756** | 0.9427 | Neutral |
| P.14b | Test-time augmentation | 0.6388 | 0.4693 | **0.9618** | 87.43% | 0.8619 | 0.9610 | Negative |
| **P.15** | **Multi-Quality ELA (Q=75/85/95)** | **0.7329** | **0.5785** | 0.9608 | 87.53% | 0.8660 | 0.9423 | **Positive** |
| P.16 | DCT spatial map (replacing ELA) | 0.3209 | 0.1911 | 0.7778 | 61.60% | 0.5678 | 0.6204 | Negative |
| P.17 | ELA + DCT 6-channel fusion | 0.7302 | 0.5751 | 0.9431 | 87.06% | 0.8589 | 0.9462 | Positive |

Reproducibility verified: P.3 run-01 and run-02 produced identical metrics. P.10 run-01 and run-02 produced identical metrics.

### 5.2 Per-Metric Champions

| Metric | Best Version | Value |
|--------|-------------|-------|
| **Pixel F1** | vR.P.15 (Multi-Q ELA) | **0.7329** |
| **Pixel IoU** | vR.P.15 (Multi-Q ELA) | **0.5785** |
| Pixel AUC | vR.P.14b (TTA) | 0.9618 |
| Image Accuracy | vR.P.12 (Augmentation) | 88.48% |
| Image Macro F1 | vR.P.12 (Augmentation) | 0.8756 |
| Image AUC | vR.P.10 (CBAM) | 0.9633 |

No single experiment dominates all metrics --- a consistent finding across the study.

---

## 6. Ablation Analysis

### 6.1 Input Representation (7 experiments)

The dominant factor across the entire study.

- **RGB to ELA (P.3):** +23.74pp Pixel F1. The single largest improvement. FP rate dropped from 22.6% to 2.7%.
- **Multi-Q ELA (P.15):** +4.09pp from P.3. Stacking grayscale ELA at Q=75/85/95 provides complementary forensic signals at different compression levels. Best overall Pixel F1 (0.7329).
- **ELA+DCT fusion (P.17):** +3.82pp. DCT block-level frequency features add complementary information when combined with ELA.
- **RGB+ELA fusion (P.4):** +1.33pp. Marginal gain; image accuracy actually dropped from 86.79% to 84.42%. RGB adds noise more than signal when ELA is present.
- **DCT-only (P.16):** -37.11pp. Catastrophic failure. Block-level frequency coefficients without ELA-style spatial context are insufficient for the model.

### 6.2 Encoder Architecture (4 experiments)

- **CBAM attention (P.10):** +3.57pp from P.3. Best architectural modification. Also achieved best Image AUC (0.9633) and lowest FP rate (2.0%).
- **ResNet-50 (P.5):** F1=0.5137 on RGB (2.86x more parameters than ResNet-34, lower F1).
- **EfficientNet-B0 (P.6):** F1=0.5217 on RGB (smallest model at 2.24M, but still far below ELA results).

### 6.3 Training Strategies (4 experiments)

- **Extended training (P.7):** +2.34pp. P.3's best epoch was 25/25 (the max). With 50-epoch budget, optimum found at epoch 36.
- **Progressive unfreeze (P.8):** +0.65pp. Best epoch was in the frozen stage (epoch 23), before any unfreezing. Confirmed frozen+BN is the sweet spot.
- **Data augmentation (P.12):** +0.48pp pixel F1 but achieved the best image-level accuracy (88.48%) and image macro F1 (0.8756). Augmentation helps classification robustness more than pixel-level precision.
- **AMP/TF32 (P.1.5):** Neutral on metrics. Carried forward for speed.

### 6.4 Loss Function (1 experiment)

- **Focal+Dice (P.9):** +0.03pp (essentially zero). Pixel AUC regressed by -2.05pp because Focal Loss concentrates predictions at extremes. The baseline BCE+Dice already handles imbalance through the Dice component.

### 6.5 Evaluation Strategy (1 experiment)

- **Test-time augmentation (P.14b):** -5.32pp Pixel F1. Averaging predictions across 4 views (original, H-flip, V-flip, both) smooths sharp mask boundaries, destroying localization precision. However, it achieved the best Pixel AUC (0.9618) and lowest FP rate (1.2%) --- useful if soft probability ranking matters more than sharp masks.

### Impact Hierarchy

| Rank | Category | Modification | Delta from P.3 |
|------|----------|-------------|----------------|
| 1 | Input | Multi-Quality ELA (P.15) | +4.09pp |
| 2 | Input | ELA+DCT fusion (P.17) | +3.82pp |
| 3 | Architecture | CBAM attention (P.10) | +3.57pp |
| 4 | Training | Extended training (P.7) | +2.34pp |
| 5 | Input | RGB+ELA fusion (P.4) | +1.33pp |
| 6 | Training | Progressive unfreeze (P.8) | +0.65pp |
| 7 | Training | Augmentation (P.12) | +0.48pp |
| 8 | Loss | Focal+Dice (P.9) | +0.03pp |
| 9 | Evaluation | TTA (P.14b) | **-5.32pp** |
| 10 | Input | DCT-only (P.16) | **-37.11pp** |

**Central conclusion:** Input representation > Attention mechanisms > Training configuration > Loss function.

---

## 7. Error Analysis

### 7.1 False Positive / False Negative Rates

| Configuration | FP Rate | FN Rate |
|--------------|---------|---------|
| RGB baseline (P.1) | 22.6% | 40.4% |
| ELA baseline (P.3) | 2.7% | 28.6% |
| CBAM attention (P.10) | 2.0% | 28.3% |
| Data augmentation (P.12) | 2.6% | 24.6% |
| TTA (P.14b) | 1.2% | 29.3% |

ELA reduces the FP rate dramatically (22.6% to 2.7%). CBAM further reduces it to 2.0%. TTA achieves the lowest FP rate (1.2%) but at the cost of higher FN rate and destroyed pixel F1.

### 7.2 Pixel-Level vs Image-Level Trade-offs

The study reveals a tension between pixel-level and image-level metrics:

- **P.12** achieves the best image accuracy (88.48%) but only modest pixel F1 (0.6968). Augmentation improves classification robustness without sharpening masks.
- **P.15** achieves the best pixel F1 (0.7329) but slightly lower image accuracy (87.53%). Multi-Q ELA sharpens localization at the cost of marginal detection sensitivity.
- **P.8** achieves the best pixel precision (0.8857). Conservative predictions produce fewer false positives but miss more true positives.

### 7.3 Best Model (P.15) Detailed Metrics

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7329 |
| Pixel IoU | 0.5785 |
| Pixel AUC | 0.9608 |
| Pixel Precision | 0.8409 |
| Pixel Recall | 0.6496 |
| Image Accuracy | 87.53% |
| Image Macro F1 | 0.8660 |
| Image ROC-AUC | 0.9423 |

The model achieves high precision (0.8409) but moderate recall (0.6496). It correctly identifies tampered regions when it flags them, but misses some tampered areas --- particularly subtle copy-move forgeries and small tampered regions.

---

## 8. Classification Track (ETASR) Summary

The ETASR track reproduced and ablated the reference paper's CNN: 2x Conv2D(32, 5x5) + MaxPool + Flatten + Dense(256) + Softmax, trained on 128x128 ELA images.

### Paper Reproduction Gap

- Paper claim: 96.21% accuracy
- Best honest reproduction: 90.23% (vR.1.6)
- Persistent gap: **5.98pp**, never closed

The gap is likely due to dataset filtering: the original paper used JPEG-only images (9,501), while this study used all formats (12,614 images).

### The Flatten-Dense Bottleneck Diagnosis

The Flatten layer connecting the final conv feature map to Dense(256) created a 29.5M-parameter bottleneck (99%+ of total parameters). This caused:
- Data augmentation failure (vR.1.2): augmented images activate entirely different neurons because Flatten memorizes exact pixel positions
- Training instability across all runs
- 99%+ parameter concentration in a single layer

The breakthrough (vR.1.6) was adding a 3rd convolutional layer with MaxPool before Flatten, which **reduced spatial dimensions** and cut parameters from 29.5M to 13.8M (-53%) while improving accuracy by +1.85pp. This is counterintuitive --- adding a layer reduced parameters --- but it demonstrates that reducing the Flatten bottleneck is more important than parameter count.

---

## 9. Visual Results

Each experiment notebook generates a 4-panel visualization grid for sampled test images:
- **Original image:** The input photograph
- **Ground truth mask:** Binary mask showing actual tampered regions (white = tampered)
- **Predicted mask:** Model output after thresholding at 0.5
- **Overlay:** Predicted mask overlaid on the original image

Qualitative observations from the best models (P.10, P.15):
- Sharp localization boundaries on large spliced regions
- False positive rates of 2.0-4.1% mean authentic images very rarely get false detections
- The model performs best on splicing forgeries with different source compression levels
- It struggles with: subtle copy-move where source and target textures are similar, very small tampered regions, and images that have been re-saved multiple times (destroying ELA signal)

<!-- Visual comparison grids are available in the executed Kaggle notebook outputs for vR.P.10 and vR.P.15 -->
<!-- Future: export visualization grids from P.15 notebook cell outputs as standalone images -->

---

## 10. Reproducibility & Experiment Tracking

### Weights & Biases Integration

All experiments are logged to a centralized W&B project (`Tampered Image Detection & Localization`), tracking:
- **Per-epoch:** train_loss, val_loss, val_pixel_f1, val_pixel_iou, learning_rate
- **Final test:** pixel_f1, pixel_iou, pixel_auc, pixel_precision, pixel_recall, image_accuracy, image_macro_f1, image_roc_auc
- **Media:** prediction_examples (wandb.Image)

### Infrastructure

- **5 parallel Kaggle runner accounts** executing experiments via a centralized configuration system
- **Papermill** for programmatic notebook execution from YAML experiment queues
- **Centralized leaderboard notebook** that queries W&B API and generates comparison tables, bar charts, and ablation progression plots automatically

### Reproducibility Verification

- **vR.P.3:** Run-01 and Run-02 produced identical metrics
- **vR.P.10:** Run-01 and Run-02 produced identical metrics
- All experiments use seed 42 on Kaggle T4/P100 GPUs

---

## 11. Literature Context

A survey of 21 research papers was conducted to identify techniques relevant to U-Net-based tamper detection. Papers covered ELA-CNN hybrids, dual-task frameworks, multi-stream networks, Swin Transformer approaches (EMT-Net, AUC=0.987 on NIST), chrominance-based descriptors (96.52% on CASIA v2.0), and comprehensive forensics surveys.

25 specific techniques were extracted across 5 categories:
- **Forensic preprocessing:** ELA (implemented), SRM noise maps (planned), YCbCr chrominance (planned), CLAHE
- **Multi-domain fusion:** RGB+frequency, ELA+DCT (implemented in P.17), multi-scale feature fusion
- **Architectural enhancements:** CBAM (implemented in P.10), SE blocks, edge attention
- **Training strategies:** Data augmentation (implemented in P.12), JPEG quality augmentation (planned)
- **Evaluation enhancements:** TTA (implemented in P.14b), pixel AUC metrics (implemented)

Of the 25 techniques, 13 have been implemented, 5 are documented for future work, and 7 were evaluated as low-priority.

---

## 12. Limitations

1. **Single dataset:** All results are on CASIA v2.0 only. Cross-dataset generalization (Columbia, Coverage, NIST) is untested.
2. **Fixed resolution:** 384x384 input loses fine details. Higher resolution (512x512+) may improve boundary precision.
3. **ELA quality sensitivity:** Q=75/85/95 chosen by convention. Systematic quality sweep not performed.
4. **Frozen encoder:** Full fine-tuning with proper differential learning rates and warmup scheduling is untested.
5. **Binary segmentation only:** No multi-class labeling (splicing vs copy-move vs authentic). The model treats all tampering as one class.
6. **No SRM or noise features:** Spatial Rich Model noise extraction --- a key technique in state-of-the-art systems --- is not yet implemented.
7. **No cross-dataset evaluation:** Generalization to unseen datasets and tampering techniques is unknown.
8. **No robustness testing:** Behavior against post-processing attacks (JPEG recompression, blur, noise addition) is not evaluated.

---

## 13. Future Work

### Immediate (High Confidence)

- **Combine Multi-Q ELA + CBAM:** The two best individual improvements. The vR.P.30 series is testing this combination with 50-epoch training and geometric augmentation.
- **Extended training for all ELA variants:** P.3 and P.10 both hit the epoch ceiling (best epoch = last epoch), suggesting more training would help.

### Medium-Term

- **SRM noise filters:** Spatial Rich Model high-pass filters extract manipulation traces in the noise domain. Used by state-of-the-art systems (EMT-Net achieved AUC=0.987 on NIST).
- **YCbCr chrominance analysis:** Cb/Cr channels encode tamper traces better than luminance alone (96.52% on CASIA v2.0 with chrominance-only features).
- **Higher resolution:** 512x512 or 640x640 input for finer boundary detection.
- **JPEG quality augmentation:** Random re-compression during training to improve robustness.

### Long-Term

- **Cross-dataset evaluation:** Columbia, Coverage, NIST16 benchmarks
- **Ensemble methods:** Combine predictions from Multi-Q ELA, CBAM, and augmentation variants
- **Multi-class segmentation:** Distinguish splicing from copy-move at the pixel level
- **Transformer encoders:** Swin Transformer or ConvNeXt may capture long-range dependencies better than ResNet

---

## 14. Conclusion

This study established through 17 controlled ablation experiments that **input representation is the dominant factor** in deep learning-based image tamper localization. ELA preprocessing produced a +23.74pp Pixel F1 improvement over raw RGB --- more than all other improvements combined. Multi-quality ELA (Q=75/85/95) achieved the best overall Pixel F1 of 0.7329, while CBAM attention was the most parameter-efficient improvement (+3.57pp for 11K parameters).

The impact hierarchy is clear and consistent: **Input representation > Attention mechanisms > Training configuration > Loss function.**

Beyond the specific numbers, the methodology itself is a contribution. The strict single-variable ablation discipline --- one change per experiment, clear baselines, reproducibility verification --- transforms ad-hoc experimentation into systematic scientific investigation. Each failure (data leakage in vK, classification-only architecture in vR.x.x, catastrophic augmentation in vR.1.2) became a lesson that directly shaped the next approach.

---

## References

1. Nagm, A. et al. (2024). "Enhanced Image Tampering Detection using ELA and a CNN." Engineering, Technology & Applied Science Research (ETASR), 14(1), 12757-12762.
2. Dong, J., Wang, W., & Tan, T. (2013). CASIA Image Tampering Detection Evaluation Database v2.0. Institute of Automation, Chinese Academy of Sciences.
3. Yakubovskiy, P. (2019). Segmentation Models PyTorch. GitHub. https://github.com/qubvel/segmentation_models.pytorch
4. Woo, S. et al. (2018). "CBAM: Convolutional Block Attention Module." ECCV 2018.
5. He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016.
6. Ronneberger, O. et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
7. Bianchi, T. & Piva, A. (2012). "Image Forgery Localization via Block-Grained Analysis of JPEG Artifacts." IEEE Transactions on Information Forensics and Security.
