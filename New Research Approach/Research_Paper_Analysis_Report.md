# Research Paper Analysis Report
## Techniques to Improve the Tampered Image Detection Project

> **Purpose:** Extract actionable techniques from 16 research papers and identify improvements for the current U-Net + ResNet34 tamper detection project.  
> **Scope:** Improve the existing system — NOT rewrite the architecture entirely.

---

## Section 1: Research Paper Summaries

### 1.1 Papers from Markdown Research Document (IEEE)

| # | Paper | Key Contribution |
|---|-------|-----------------|
| P1 | **ELA-CNN Hybrid** (IEEE 10444440) | ELA preprocessing + 2-layer CNN achieves 87.75% accuracy on CASIA2 in 10 epochs. ELA amplifies compression artifacts invisible in RGB. |
| P2 | **Dual-task Classification + Segmentation** (IEEE 10052973) | Single end-to-end framework with shared backbone branching into classification and segmentation heads. Exploits Double Quantization (DQ) effect for JPEG forensics. |
| P3 | **Copy-Rotate-Move Detection** (IEEE 10168896) | Zernike moments (rotation-invariant) + Ant Colony Optimization. 98.44% accuracy on MICC-F220. Open Access. |
| P4 | **U-Net Mixed Tampering Localization** (IEEE 10652417) | U-Net encoder-decoder with spatial-frequency feature fusion (FENet). Adaptive feature fusion + edge attention mechanisms. ~95% accuracy on CASIA v2. |
| P5 | **MD5 + OpenCV Hybrid** (IEEE 10895348) | Dual-layered: cryptographic MD5 hashing for integrity + OpenCV texture/color/shape analysis for visual forensics. |

### 1.2 Main Folder PDFs

| # | Paper | Key Contribution |
|---|-------|-----------------|
| P6 | **Multistream Networks for ID Tampering** (043018_1.pdf) | Three parallel branches: texture, frequency-domain, and noise. Multiscale Feature Fusion Module (MSFF). F1=0.912 on custom ID dataset. Robust to JPEG compression. |
| P7 | **Enhanced ELA + CNN** (ETASR_9593.pdf) | Custom CNN with ELA preprocessing achieves **96.21% accuracy** on CASIA v2.0, outperforming VGG16 (90.32%), VGG19 (88.92%), ResNet101 (74.75%). |
| P8 | **CNN-Based Tampering Detection** (IJCRT24A5072.pdf) | Educational overview: CNN pipeline for preprocessing → feature extraction → classification on CASIA V1.0. |
| P9 | **Multi-Technique Review** (IMAGE_TAMPERING_DETECTION...pdf) | 33-page survey: 60+ methods from traditional (LBP, DCT, SIFT) to deep learning (CNNs, GANs, Transformers). CNN-SVM achieves 97.8% on CASIA/Columbia. |
| P10 | **QFT for Medical Image Auth** (Tamper_Localisation...pdf) | Quantum Fourier Transform phase-only watermarking. PSNR ~71 dB, SSIM ~0.99999. Specialized for medical imaging. |
| P11 | **DL-Based Tamper Detection Study** (IJERTV13IS020023.pdf) | Survey of CNN, SIFT, SURF, DCT, LBP, CFA forensics approaches. Identifies remaining challenges in universal detection. |
| P12 | **Video/Image Authentication Review** (A_Review_on_Video...pdf) | Early (2013) classification of digital signatures, watermarks, hash-based authentication, blind deconvolution. |

### 1.3 More Research Papers Subfolder

| # | Paper | Key Contribution |
|---|-------|-----------------|
| P13 | **EMT-Net: Multiple Tampering Traces + Edge Enhancement** (1-s2.0-S0031320322005064-main.pdf) | Swin Transformer for global noise + ResNet blocks for local noise (from SRM maps) + CNN for RGB artifacts + Edge Artifact Enhancement. AUC=0.987 on NIST. Robust to blur, JPEG, noise. **No pretraining required.** |
| P14 | **Traditional to DL Comprehensive Evaluation** (11042_2022_Article_13808.pdf) | 34-page survey comparing block-based (DCT, LBP, SIFT), keypoint-based, CNN-SVM (97.8%), FCN/VGG-16 across 10+ datasets. |
| P15 | **DL Methods for Image Forensics Review** (A_Comprehensive_Review...pdf) | 39-page review of 180+ methods. Covers constrained convolutional layers (SRM filters), Siamese networks, attention mechanisms, anti-forensics. Identifies that standard CV strategies don't directly apply to forensics. |
| P16 | **Multi-scale Weber Local Descriptors** (evaluation-of-image-forgery...pdf) | Chrominance-based (YCbCr Cb/Cr) multi-scale Weber descriptors + SVM. **96.52% accuracy** on CASIA v2.0. Demonstrates chrominance encodes tamper traces better than luminance. |
| P17 | **ME-Net: Multi-Task Edge-Enhanced** (ME - Multi-Task Edge-Enhanced...pdf) | Dual-branch: ConvNeXt for RGB edge features + ResNet-50 for noise. PSDA fusion + EEPA edge enhancement. F1=0.905 on NIST16, AUC=0.975. Outperforms ManTra-Net, MVSS-Net. |
| P18 | **Semi-Fragile Watermarking** (Optimal_Semi-Fragile...pdf) | QDTCWT + Maximum Entropy Random Walk + Swin Transformer watermark generation. PSNR=65 dB. Active method (watermark embedding). |
| P19 | **Copy-Move via Evolving Circular Domains** (s11042-022-12755-w.pdf) | SIFT + SURF in log-polar space + RANSAC + Evolving Circular Domains Coverage. F1=91.56% on FAU. Specifically for copy-move. |
| P20 | **Hybrid DCCAE + ADFC** (s11042-023-15475-x.pdf) | WE-CLAHE preprocessing + Hybrid DTT + VGGNet + Capsule Auto-Encoder + Fuzzy Clustering. **99.23% accuracy** on CASIA V1. |
| P21 | **TransU²-Net** (TransU_2_-Net...pdf) | U2-Net + self-attention (last encoder block) + cross-attention (skip connections). F-measure=0.735 on CASIA (14.2% improvement over base U2-Net). No large-dataset pretraining required. |

---

## Section 2: Extracted Techniques Relevant to the Project

From the 16 papers analyzed, the following techniques have direct relevance to improving the current U-Net + ResNet34 tamper detection system:

### 2.1 Forensic Feature Preprocessing

| Technique | Source Papers | Description | Relevance |
|-----------|--------------|-------------|-----------|
| **Error Level Analysis (ELA)** | P1, P7 | JPEG re-compression difference map highlights compression artifacts invisible in RGB | Amplifies forensic features pre-learning; 96.21% accuracy when combined with CNN (P7) |
| **SRM Noise Maps** | P13, P15, P17 | Spatial Rich Model high-pass filters extract noise residuals | Captures manipulation traces in noise domain; critical component of EMT-Net (AUC=0.987) |
| **Chrominance Analysis (YCbCr)** | P16 | Extract Cb/Cr channels which encode tamper traces better than luminance | 96.52% accuracy on CASIA v2.0 using chrominance alone |
| **CLAHE Preprocessing** | P20 | Contrast Limited Adaptive Histogram Equalization enhances subtle artifacts | Part of pipeline achieving 99.23% accuracy on CASIA V1 |

### 2.2 Multi-Domain Feature Fusion

| Technique | Source Papers | Description | Relevance |
|-----------|--------------|-------------|-----------|
| **Spatial + Frequency Fusion** | P4, P6, P13 | Combine RGB spatial features with frequency-domain representations | FENet (P4) achieves ~95% on CASIA v2; EMT-Net fuses noise + RGB + edge features |
| **Multi-Stream Architecture** | P6, P13, P17 | Parallel branches extracting different feature domains | P6: texture + frequency + noise branches; P17: ConvNeXt + ResNet-50 dual-branch |
| **Noise + RGB Dual-Branch** | P13, P17 | Dedicated noise extraction branch alongside RGB | ME-Net (P17): F1=0.905 on NIST16; EMT-Net (P13): AUC=0.987 on NIST |

### 2.3 Architectural Enhancements

| Technique | Source Papers | Description | Relevance |
|-----------|--------------|-------------|-----------|
| **Edge Attention / Enhancement** | P4, P13, P17 | Explicit edge supervision to preserve boundary artifacts | EMT-Net's EAE prevents loss of boundary clues under post-processing; EEPA integrates edge context into decoding |
| **Self-Attention in Encoder** | P21 | Transformer self-attention in last encoder block for global context | TransU²-Net: 14.2% F-measure improvement on CASIA; captures long-range dependencies |
| **Cross-Attention in Skip Connections** | P21 | High-level features guide low-level feature enhancement in skip connections | Filters non-semantic features in decoder; no large-dataset pretraining required |
| **Dual-Task (Classification + Segmentation)** | P2 | Shared backbone branching into classification + segmentation heads | Shared features enhance both tasks; exploits DQ effect |
| **Pyramid Split Double Attention (PSDA)** | P17 | Spatial + channel-wise attention for cross-domain feature fusion | ME-Net fuses hierarchical features from RGB and noise branches |

### 2.4 Training and Augmentation Strategies

| Technique | Source Papers | Description | Relevance |
|-----------|--------------|-------------|-----------|
| **JPEG Compression Augmentation** | P1, P4, P7, P13 | Train with JPEG-compressed variants to improve robustness | Multiple papers demonstrate JPEG robustness as key differentiator |
| **Gaussian Noise/Blur Augmentation** | P4, P13 | Add noise/blur during training to simulate anti-forensic attacks | EMT-Net robust to Gaussian blur (kernel up to 15) and JPEG quality 50+ |
| **Edge Supervision Loss** | P13, P17 | Auxiliary loss on edge prediction to sharpen boundary detection | Reinforces boundary artifacts that post-processing degrades |
| **Polynomial LR Decay** | P17 | SGD with polynomial learning rate decay schedule | ME-Net training strategy |

### 2.5 Evaluation Enhancements

| Technique | Source Papers | Description | Relevance |
|-----------|--------------|-------------|-----------|
| **AUC-ROC as Primary Metric** | P13, P17 | Pixel-level AUC-ROC alongside F1 for threshold-independent evaluation | EMT-Net reports both AUC and F1; more robust than threshold-dependent F1 alone |
| **Per-Forgery-Type Evaluation** | P4, P14 | Separate metrics for splicing vs. copy-move | Copy-move is harder to localize than splicing due to self-similarity |
| **Post-Processing Robustness Protocol** | P4, P13 | Systematic evaluation under JPEG, blur, noise degradations | Multiple severity levels with Δ from clean baseline |

---

## Section 3: Comparison with Current Project

### 3.1 Implementation Status Matrix

| Technique | Current Status | Details |
|-----------|---------------|---------|
| **U-Net Architecture** | ✅ Implemented | `smp.Unet(encoder_name="resnet34")` — solid baseline |
| **BCE + Dice Loss** | ✅ Implemented | Handles class imbalance well for small tampered regions |
| **AMP Training** | ✅ Implemented | Efficient memory usage on T4 |
| **Gradient Accumulation** | ✅ Implemented | Effective batch size 16 |
| **Early Stopping** | ✅ Implemented | Patience=10 on val Pixel-F1 |
| **Differential LR** | ✅ Implemented | Encoder 1e-4, decoder/head 1e-3 |
| **Basic Augmentation** | ✅ Implemented | Flips, rotation, resize |
| **Pixel-F1 / IoU Metrics** | ✅ Implemented | Primary evaluation metrics |
| **Image-level Detection** | ✅ Implemented | `max(prob_map)` approach |
| **Robustness Testing** | ✅ Implemented | JPEG, noise, blur, resize degradations |
| **W&B Tracking** | ✅ Implemented | Guarded behind flag (v4 notebook) |
| | | |
| **ELA Preprocessing** | 🟡 Documented (Phase 2) | Planned as 4th channel (RGB+ELA → `in_channels=4`); not yet implemented |
| **SRM Noise Maps** | 🟡 Documented (Phase 3) | Planned as separate experimental path (`in_channels=6`); not yet implemented |
| **Photometric Augmentation** | 🟡 Documented (Phase 2) | `RandomBrightnessContrast`, `HueSaturationValue`, `GaussNoise`, `ImageCompression` planned |
| **LR Scheduler** | 🟡 Documented (Phase 2) | `CosineAnnealingWarmRestarts` planned but not active in MVP |
| **Encoder Comparison** | 🟡 Documented (Phase 3) | EfficientNet-B0/B1 alternatives noted |
| | | |
| **Edge Attention/Supervision** | ❌ Not Implemented | No edge-specific loss or attention module |
| **Self-Attention in Encoder** | ❌ Not Implemented | No transformer blocks in encoder |
| **Cross-Attention in Skip Connections** | ❌ Not Implemented | Standard U-Net skip connections only |
| **Multi-Stream/Dual-Branch** | ❌ Not Implemented | Single RGB stream only |
| **Frequency-Domain Features** | ❌ Not Implemented | No DCT/FFT/frequency analysis |
| **Chrominance (YCbCr) Analysis** | ❌ Not Implemented | RGB-only input |
| **Dual-Task (Classification + Segmentation)** | ❌ Not Implemented | Segmentation only; image-level from `max(prob_map)` |
| **Edge Supervision Loss** | ❌ Not Implemented | No auxiliary edge loss |
| **CLAHE Preprocessing** | ❌ Not Implemented | No contrast enhancement |
| **Pixel-level AUC-ROC** | ❌ Not Implemented | Only image-level AUC-ROC |
| **JPEG Compression Augmentation** | ❌ Not in MVP | Listed for Phase 2 but not active |

### 3.2 Gap Analysis Summary

**Strengths of current implementation:**
- Solid U-Net + ResNet34 backbone — proven effective in P4 (U-Net Mixed Tampering, ~95% accuracy)
- BCE+Dice loss — well-suited for class-imbalanced binary segmentation
- Proper evaluation protocol with threshold sweep and robustness testing
- Good engineering: AMP, gradient accumulation, checkpointing, W&B

**Key gaps relative to SOTA (from papers):**
1. **No forensic feature extraction** — Relying on RGB only (no ELA, no SRM, no frequency features). Papers P4, P6, P7, P13, P17 all show that forensic preprocessing dramatically improves detection.
2. **No edge supervision** — Papers P13, P17 demonstrate that explicit edge attention/supervision is critical for maintaining boundary accuracy under post-processing attacks.
3. **No multi-domain fusion** — Current architecture is single-stream RGB. Best-performing methods (P6, P13, P17) use parallel branches extracting RGB + noise + frequency/edge features.
4. **No attention mechanisms** — No self-attention for global context or cross-attention for guided decoding. TransU²-Net (P21) shows 14.2% F-measure improvement from attention alone.
5. **Limited augmentation** — No JPEG compression or photometric augmentation in MVP, which are critical for robustness (P4, P13).

---

## Section 4: Suggested Improvements (Prioritized)

### 🔴 High Priority — Significant Impact, Feasible on Colab T4

#### H1. Add ELA as 4th Input Channel
- **Source:** P1, P7 (96.21% accuracy with ELA+CNN)
- **What:** Compute ELA map (JPEG re-save at QF=95 → absolute difference) and concatenate as 4th channel
- **Impact:** ELA amplifies compression artifacts invisible in RGB; proven to accelerate convergence and improve accuracy significantly
- **Trade-off:** Loses ImageNet pretrained weights (new first conv layer)
- **Already documented** in Docs4 Phase 2

#### H2. Enable JPEG Compression Augmentation
- **Source:** P4, P7, P13 (all show JPEG robustness is critical)
- **What:** Add `A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3)` to training augmentation
- **Impact:** Directly improves robustness to the most common real-world degradation; currently the biggest gap in the training pipeline
- **Already documented** in Docs4 Phase 2 (photometric augmentations)

#### H3. Add Edge Supervision Loss
- **Source:** P13 (EMT-Net), P17 (ME-Net)
- **What:** Add auxiliary binary cross-entropy loss on predicted edges vs. Canny/Sobel edges of ground-truth mask. Total loss = `BCE_Dice + λ * EdgeLoss` (λ ≈ 0.2–0.5)
- **Impact:** Preserves boundary accuracy under post-processing; EMT-Net and ME-Net both achieve SOTA results partly due to edge supervision
- **Complexity:** Low — requires Canny edge extraction from GT masks + auxiliary decoder head or simple conv layer

#### H4. Enable All Phase 2 Augmentations
- **Source:** P4, P13 (robustness testing)
- **What:** Activate `RandomBrightnessContrast`, `HueSaturationValue`, `GaussNoise`, `ImageCompression` in training pipeline
- **Impact:** Improves generalization and robustness to photometric variations
- **Already documented** in Docs4 Phase 2

### 🟡 Medium Priority — Meaningful Improvement, Moderate Complexity

#### M1. Add SRM Noise Maps as Additional Input Channels
- **Source:** P13 (EMT-Net uses SRM for noise extraction), P15 (recommends constrained convolutions)
- **What:** Apply 3 SRM high-pass filter kernels to input → concatenate with RGB → `in_channels=6`
- **Impact:** Captures manipulation traces in noise domain that are invisible in RGB; EMT-Net's noise branch is key to its AUC=0.987 on NIST
- **Trade-off:** Loses ImageNet pretrained weights; separate experimental path from ELA
- **Already documented** in Docs4 Phase 3

#### M2. Add Lightweight Attention to Decoder
- **Source:** P21 (TransU²-Net), P17 (ME-Net EEPA)
- **What:** Add channel attention (SE blocks) or CBAM modules to the U-Net decoder blocks. These are lightweight (few parameters) and don't require changing the encoder.
- **Impact:** TransU²-Net shows attention improves F-measure by 14.2%; attention helps the decoder focus on forensically relevant features
- **Complexity:** Moderate — requires modifying the SMP U-Net decoder or wrapping it

#### M3. Add Pixel-Level AUC-ROC Metric
- **Source:** P13, P17 (both report pixel-level AUC alongside F1)
- **What:** Compute AUC-ROC at pixel level using raw sigmoid outputs vs. GT mask
- **Impact:** Threshold-independent metric; better captures model discrimination ability. Current project only has image-level AUC-ROC.
- **Complexity:** Low — `sklearn.metrics.roc_auc_score(gt.flatten(), pred.flatten())`

#### M4. Add Classification Head (Dual-Task)
- **Source:** P2 (Dual-task Classification + Segmentation)
- **What:** Add a small classification branch off the encoder bottleneck: `GlobalAvgPool → FC → Sigmoid`. Train with auxiliary BCE loss for binary classification (authentic/tampered).
- **Impact:** Shared feature representation enhances both tasks; provides a more principled image-level detection score than `max(prob_map)`
- **Complexity:** Moderate — requires architectural modification and loss balancing

#### M5. Enable LR Scheduler
- **Source:** P17 (polynomial decay), Docs4 Phase 2 (CosineAnnealingWarmRestarts)
- **What:** Activate `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` as already documented
- **Impact:** Prevents learning rate stagnation; standard practice for training convergence
- **Already documented** in Docs4 Phase 2

### 🟢 Future Work — High Complexity or Architectural Change

#### F1. Multi-Stream Architecture (RGB + Noise + Edge)
- **Source:** P6, P13, P17
- **What:** Parallel encoder branches: (1) RGB → ResNet34, (2) SRM noise → lightweight encoder, (3) edge features. Fuse at decoder.
- **Impact:** Best-performing methods (P13: AUC=0.987, P17: F1=0.905 on NIST16) all use multi-stream architectures
- **Trade-off:** Significantly increases model complexity and VRAM; may not fit on T4 with large images; requires substantial rewrite

#### F2. Transformer Attention in Encoder
- **Source:** P21 (TransU²-Net), P13 (Swin Transformer in EMT-Net)
- **What:** Replace last encoder block with transformer self-attention for global context
- **Impact:** Captures long-range dependencies (large tampered regions spanning the image)
- **Trade-off:** Adds significant compute; may require reducing resolution or batch size on T4

#### F3. Cross-Attention in Skip Connections
- **Source:** P21 (TransU²-Net)
- **What:** Replace concatenation in skip connections with cross-attention (high-level queries, low-level keys/values)
- **Impact:** More selective feature propagation; filters non-semantic features in decoder
- **Trade-off:** Moderate complexity; novel architecture change

#### F4. Chrominance Feature Extraction
- **Source:** P16 (96.52% accuracy using Cb/Cr channels alone)
- **What:** Convert to YCbCr, extract Cb/Cr chrominance channels as additional inputs or parallel stream
- **Impact:** Chrominance encodes tamper traces better than luminance per P16
- **Trade-off:** Additional channels; optional preprocessing step

#### F5. Encoder Upgrade (EfficientNet)
- **Source:** P9 (EfficientNetV2B0 beats older backbones), Docs4 Phase 3
- **What:** Try `smp.Unet(encoder_name="efficientnet-b0")` or `efficientnet-b1`
- **Impact:** More parameter-efficient with potentially better feature extraction
- **Already documented** in Docs4 Phase 3

#### F6. CLAHE Preprocessing
- **Source:** P20 (part of 99.23% accuracy pipeline)
- **What:** Apply CLAHE to enhance local contrast before feeding to model
- **Impact:** Enhances subtle artifacts; low-complexity preprocessing
- **Trade-off:** May introduce artifacts of its own; needs careful integration with existing normalize pipeline

---

## Section 5: Concrete Implementation Suggestions

### For H1: ELA as 4th Input Channel

**Where in notebook:** Cell creating dataset/transforms (Section 4: Data Pipeline)

**Code changes:**
```python
# In the Dataset class __getitem__ method:
import cv2
import numpy as np
from PIL import Image
import io

def compute_ela(image_path, quality=95):
    """Compute Error Level Analysis map."""
    img = Image.open(image_path).convert('RGB')
    # Re-save at specified quality
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert('RGB')
    # Compute absolute difference
    ela = np.abs(np.array(img, dtype=np.float32) - np.array(resaved, dtype=np.float32))
    # Convert to single channel (mean across RGB)
    ela_gray = np.mean(ela, axis=2)
    # Normalize to [0, 1]
    ela_max = ela_gray.max()
    if ela_max > 0:
        ela_gray = ela_gray / ela_max
    return ela_gray  # Shape: (H, W)

# In __getitem__:
ela_map = compute_ela(self.image_paths[idx])
ela_map = cv2.resize(ela_map, (512, 512))
# Stack with RGB: shape becomes (H, W, 4)
image_4ch = np.concatenate([image, ela_map[..., np.newaxis]], axis=2)

# Model change:
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # Cannot use ImageNet with 4 channels
    in_channels=4,
    classes=1,
    activation=None,
)
```

**Expected benefit:** Based on P7, ELA preprocessing improved accuracy from ~74-90% (various pretrained models) to 96.21% on CASIA v2.0. Even without ImageNet pretraining, ELA provides strong forensic signal.

---

### For H2: JPEG Compression Augmentation

**Where in notebook:** Cell defining training transforms (Section 4)

**Code changes:**
```python
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # ADD: JPEG compression augmentation
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

**Expected benefit:** Directly addresses the most common real-world degradation. Per P4 and P13, models trained with JPEG augmentation maintain high F1 even at QF=50, while those without see significant drops.

---

### For H3: Edge Supervision Loss

**Where in notebook:** Cell defining the loss function (Section 5: Model Architecture) and training loop (Section 6)

**Code changes:**
```python
import torch.nn.functional as F

def compute_edge_gt(mask):
    """Extract edges from ground-truth mask using Sobel filters."""
    # mask: (B, 1, H, W)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(mask.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(mask.device)
    edge_x = F.conv2d(mask.float(), sobel_x, padding=1)
    edge_y = F.conv2d(mask.float(), sobel_y, padding=1)
    edge = (edge_x.abs() + edge_y.abs()).clamp(0, 1)
    return edge  # (B, 1, H, W)

class BCEDiceEdgeLoss(nn.Module):
    def __init__(self, smooth=1.0, edge_weight=0.3):
        super().__init__()
        self.bce_dice = BCEDiceLoss(smooth=smooth)
        self.edge_bce = nn.BCEWithLogitsLoss()
        self.edge_weight = edge_weight
    
    def forward(self, logits, targets):
        main_loss = self.bce_dice(logits, targets)
        edge_gt = compute_edge_gt(targets)
        edge_loss = self.edge_bce(logits, edge_gt)
        return main_loss + self.edge_weight * edge_loss
```

**Expected benefit:** Per P13 and P17, edge supervision prevents the model from producing blurry boundary predictions. This is especially important for robustness testing where post-processing (blur, JPEG) degrades edge information. Expected improvement: sharper masks, better F1 on tampered-only evaluation, and more robust performance under degradation.

---

### For M2: Lightweight Attention in Decoder (SE Blocks)

**Where in notebook:** After model initialization (Section 5)

**Code changes:**
```python
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

# Add SE blocks to decoder outputs:
# After model creation, wrap decoder blocks
for i, block in enumerate(model.decoder.blocks):
    out_channels = block.conv2[0].out_channels  # SMP internal structure
    setattr(model.decoder, f'se_{i}', SEBlock(out_channels))
```

**Expected benefit:** Lightweight channel attention helps the decoder focus on forensically relevant feature channels. Per P17, attention mechanisms in the decoder contribute to F1 improvements. SE blocks add minimal parameters (~0.1% increase) and VRAM overhead.

---

### For M3: Pixel-Level AUC-ROC

**Where in notebook:** Evaluation cell (Section 8)

**Code changes:**
```python
from sklearn.metrics import roc_auc_score

def compute_pixel_auc(pred_probs, gt_masks):
    """Compute pixel-level AUC-ROC across all test images."""
    all_pred = []
    all_gt = []
    for pred, gt in zip(pred_probs, gt_masks):
        all_pred.append(pred.flatten())
        all_gt.append(gt.flatten())
    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    # Only compute if both classes present
    if len(np.unique(all_gt)) < 2:
        return float('nan')
    return roc_auc_score(all_gt, all_pred)
```

**Expected benefit:** Threshold-independent metric that better measures the model's discrimination ability. Both EMT-Net (P13) and ME-Net (P17) use pixel-level AUC as a primary metric. Trivial to implement.

---

### For M4: Classification Head (Dual-Task)

**Where in notebook:** Model definition (Section 5) and training loop (Section 6)

**Code changes:**
```python
class DualTaskUNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        # Classification head from encoder bottleneck
        encoder_channels = base_model.encoder.out_channels[-1]  # 512 for ResNet34
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels, 1),  # Binary: tampered or not
        )
    
    def forward(self, x):
        features = self.base.encoder(x)
        seg_output = self.base.decoder(*features)
        seg_output = self.base.segmentation_head(seg_output)
        # Classification from deepest features
        cls_output = self.cls_head(features[-1])
        return seg_output, cls_output

# In training loop:
seg_logits, cls_logits = model(images)
seg_loss = criterion(seg_logits, masks)
# Image-level label: 1 if any tampered pixel, else 0
cls_target = (masks.sum(dim=[1,2,3]) > 0).float().unsqueeze(1)
cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_target)
total_loss = seg_loss + 0.5 * cls_loss
```

**Expected benefit:** Provides a learned image-level detection score instead of heuristic `max(prob_map)`. Per P2, shared backbone features enhance both classification and segmentation. Expected improvement in Image Accuracy and AUC-ROC metrics.

---

## Appendix: Priority Implementation Roadmap

| Phase | Items | Estimated Effort | Dependencies |
|-------|-------|-----------------|-------------|
| **Phase 2A** (Quick Wins) | H2 (JPEG augment), H4 (all augments), M5 (LR scheduler), M3 (pixel AUC) | Minimal — config changes only | None |
| **Phase 2B** (Moderate) | H1 (ELA), H3 (edge loss) | Moderate — new preprocessing + loss | None |
| **Phase 2C** (Architecture) | M2 (SE blocks), M4 (dual-task head) | Moderate — model modifications | None |
| **Phase 3** (Experimental) | M1 (SRM), F5 (EfficientNet), F4 (chrominance) | Significant — encoder changes | Phase 2B complete |
| **Future** | F1 (multi-stream), F2 (transformer), F3 (cross-attention) | Major — architectural redesign | Phase 3 insights |

---

## Key Insight from Literature

> **The single most impactful improvement** based on the research is adding **forensic preprocessing (ELA or SRM noise maps)** as additional input channels. Papers P1, P7, P13, P16, and P17 all demonstrate that the RGB domain alone contains insufficient information for robust tamper detection. The forensic features (compression artifacts, noise residuals, chrominance anomalies) provide complementary signals that dramatically improve both accuracy and robustness to post-processing attacks.

> **The second most impactful improvement** is **edge supervision** (P13, P17). Both top-performing methods (EMT-Net and ME-Net) use explicit edge attention/supervision to prevent boundary degradation, which is the primary failure mode under JPEG compression and Gaussian blur — exactly the scenarios tested in the current robustness protocol.
