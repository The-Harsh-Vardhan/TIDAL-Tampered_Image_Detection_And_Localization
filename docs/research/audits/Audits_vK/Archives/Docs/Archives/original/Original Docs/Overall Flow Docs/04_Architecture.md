# 4. Architecture — Implementation Guide

## 4.1 Assignment Requirement

> *"Train a model to predict tampered regions. The choice of architecture and loss functions is entirely up to you."*

This is where you demonstrate **thoughtful architecture choices** — the evaluators' explicit criteria.

---

## 4.2 Architecture Decision: U-Net + EfficientNet-B1 + SRM Forensic Preprocessing

### Why This Specific Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Meta-Architecture** | U-Net (encoder-decoder with skip connections) | Best for fine-grained pixel-level localization; skip connections preserve spatial detail through the network |
| **Encoder Backbone** | EfficientNet-B1 (pre-trained on ImageNet) | Compound scaling (resolution + depth + width) gives best accuracy-per-FLOP; 7.8M params fits T4 comfortably |
| **Forensic Preprocessing** | SRM Filter Bank (30 fixed high-pass kernels) | Suppresses semantic content, reveals noise-level artifacts invisible to RGB-only models — the single biggest forensic-specific improvement |
| **Library** | `segmentation_models_pytorch` (SMP) | Production-tested; 1-line model creation; handles encoder weight adaptation for custom `in_channels` |
| **Output** | Single-channel sigmoid (binary segmentation) | Pixel-level tampering probability map |

---

## 4.3 Component 1: SRM Filter Bank

### What It Does
The Spatial Rich Model (SRM) is a collection of 30 handcrafted 5×5 high-pass filters originally developed for steganalysis. These filters compute **noise residuals** — the difference between each pixel's actual value and the value predicted by its neighbors.

### Why It's Essential for Forensics
Standard CNNs trained on ImageNet learn to recognize **shapes, textures, and objects** (semantic features). In forensics, these are misleading — a forged image looks semantically correct. The manipulation evidence lives in the **noise floor**: subtle statistical patterns from the camera sensor, demosaicing algorithm, and compression that are disrupted by tampering.

SRM filters act as a **content suppressor**: they remove the high-amplitude semantic information and amplify the low-amplitude noise where forensic artifacts live.

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class SRMFilterLayer(nn.Module):
    """
    Applies 30 fixed SRM high-pass filters to extract noise residuals.
    Non-trainable — the filters are handcrafted forensic feature extractors.
    """
    def __init__(self):
        super().__init__()
        # Load the 30 SRM kernels (5x5 each)
        srm_kernels = self._get_srm_kernels()  # Shape: (30, 1, 5, 5)
        
        # Create conv layer: 3 input channels (RGB) → 30 output channels
        # Each SRM kernel is applied independently to each RGB channel
        # We replicate kernels across 3 input channels
        kernels_rgb = srm_kernels.repeat(1, 3, 1, 1) / 3.0  # Average across channels
        
        self.filter = nn.Conv2d(
            in_channels=3,
            out_channels=30,
            kernel_size=5,
            padding=2,
            bias=False
        )
        self.filter.weight = nn.Parameter(kernels_rgb, requires_grad=False)
    
    def forward(self, x):
        return self.filter(x)
    
    @staticmethod
    def _get_srm_kernels():
        """
        Returns the 30 SRM kernels as a tensor of shape (30, 1, 5, 5).
        
        These include:
        - 1st order edge filters (horizontal, vertical, diagonal)
        - 2nd order edge filters
        - 3rd order SQUARE filters
        - 3rd order EDGE filters
        - SQUARE 3x3 and 5x5 filters
        """
        # Simplified: key SRM kernels used in forensic literature
        # Full 30-kernel set from Fridrich & Kodovsky (2012)
        
        filter_1 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=np.float32)  # 2nd order horizontal
        
        filter_2 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  0, -2,  0,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=np.float32)  # 2nd order vertical
        
        filter_3 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  1,  0,  0,  0],
            [ 0,  0, -2,  0,  0],
            [ 0,  0,  0,  1,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=np.float32)  # 2nd order diagonal
        
        # ... (full 30 kernels implemented in final code)
        # For brevity, shown as placeholder — actual implementation 
        # uses the complete Fridrich SRM set
        
        # Stack into (30, 1, 5, 5)
        kernels = [filter_1, filter_2, filter_3]
        # Pad to 30 with rotations and higher-order variants
        for k in [filter_1, filter_2, filter_3]:
            for angle in [90, 180, 270]:
                kernels.append(np.rot90(k, k=angle // 90))
        # Extend with 3x3 embedded in 5x5
        square_3x3 = np.array([
            [0, 0, 0, 0, 0],
            [0,-1, 2,-1, 0],
            [0, 2,-4, 2, 0],
            [0,-1, 2,-1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)
        kernels.append(square_3x3)
        
        # Trim/pad to exactly 30
        while len(kernels) < 30:
            kernels.append(kernels[len(kernels) % len(kernels[:3])])
        kernels = kernels[:30]
        
        return torch.tensor(np.array(kernels)).unsqueeze(1)  # (30, 1, 5, 5)

    
class ChannelReducer(nn.Module):
    """
    Reduces 30 SRM channels to 3 channels via learnable 1x1 convolution.
    This allows the network to learn which noise residuals are most discriminative.
    """
    def __init__(self, in_channels=30, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

---

## 4.4 Component 2: U-Net with EfficientNet-B1 (via SMP)

### Model Creation

```python
import segmentation_models_pytorch as smp

# Create U-Net with 6-channel input (3 RGB + 3 compressed SRM)
model = smp.Unet(
    encoder_name="efficientnet-b1",
    encoder_weights="imagenet",
    in_channels=6,          # 3 RGB + 3 noise channels
    classes=1,               # Binary: tampered vs. authentic
    activation=None          # Raw logits → sigmoid applied in loss/inference
)
```

### How SMP Handles 6-Channel Input
When `in_channels=6` is specified:
1. SMP takes the pre-trained first convolution layer (originally 3-channel)
2. Creates a new 6-channel convolution layer
3. Copies the pre-trained 3-channel weights into the first 3 channels
4. Initializes the remaining 3 channels with the same weights (averaged)
5. All subsequent layers retain their pre-trained weights unchanged

This means the encoder immediately has a strong feature extraction capability for the RGB channels, while the noise channels start with reasonable initialization.

### Architecture Details

```
Input: (B, 6, 512, 512) — Batch × (3 RGB + 3 SRM noise) × Height × Width

EfficientNet-B1 Encoder (5 stages):
├── Stage 1: (B, 32, 256, 256)   — Low-level edges + noise patterns
├── Stage 2: (B, 24, 128, 128)   — Local texture features
├── Stage 3: (B, 40, 64, 64)     — Mid-level structural features
├── Stage 4: (B, 112, 32, 32)    — High-level semantic features
└── Stage 5: (B, 320, 16, 16)    — Bottleneck (most abstract representation)

U-Net Decoder (5 stages, with skip connections):
├── Dec 5: (B, 256, 32, 32)  + Skip from Enc Stage 4
├── Dec 4: (B, 128, 64, 64)  + Skip from Enc Stage 3
├── Dec 3: (B, 64, 128, 128) + Skip from Enc Stage 2
├── Dec 2: (B, 32, 256, 256) + Skip from Enc Stage 1
└── Dec 1: (B, 16, 512, 512) + Skip from Input

Segmentation Head:
└── Conv2d(16, 1, kernel_size=3) → (B, 1, 512, 512)  — Raw logits
```

---

## 4.5 Component 3: Full Model Assembly

```python
class TamperingDetector(nn.Module):
    """
    Full forensic tampering detection model.
    RGB + SRM noise dual-input → U-Net → tampering mask.
    """
    def __init__(self, encoder_name='efficientnet-b1', encoder_weights='imagenet'):
        super().__init__()
        
        # Forensic preprocessing
        self.srm = SRMFilterLayer()
        self.channel_reducer = ChannelReducer(in_channels=30, out_channels=3)
        
        # Segmentation backbone
        self.segmentation_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=6,   # 3 RGB + 3 noise
            classes=1,
            activation=None
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) — RGB image tensor (normalized)
        Returns:
            logits: (B, 1, H, W) — raw tampering logits (apply sigmoid for probabilities)
        """
        # Extract noise residuals
        noise = self.srm(x)                    # (B, 30, H, W)
        noise_compressed = self.channel_reducer(noise)  # (B, 3, H, W)
        
        # Concatenate RGB + noise
        combined = torch.cat([x, noise_compressed], dim=1)  # (B, 6, H, W)
        
        # Predict tampering mask
        logits = self.segmentation_model(combined)  # (B, 1, H, W)
        
        return logits
```

### Usage
```python
# Create model
model = TamperingDetector(
    encoder_name='efficientnet-b1',
    encoder_weights='imagenet'
)
model = model.to(device)

# Forward pass
images = batch_images.to(device)   # (B, 3, 512, 512)
logits = model(images)              # (B, 1, 512, 512)
probabilities = torch.sigmoid(logits)  # (B, 1, 512, 512) — values in [0, 1]
```

---

## 4.6 Loss Function Architecture

### Hybrid Loss: BCE + Dice + Edge

```python
class HybridLoss(nn.Module):
    """
    Combined loss for forensic segmentation:
    - BCE: Global pixel distribution matching
    - Dice: Region overlap optimization (class-imbalance robust)
    - Edge: Boundary precision supervision
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, edge_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target, smooth=1.0):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def edge_loss(self, pred, target):
        # Compute ground truth edges via morphological gradient
        # Dilate - Erode the target mask
        kernel = torch.ones(1, 1, 3, 3, device=target.device)
        dilated = torch.nn.functional.conv2d(target, kernel, padding=1)
        dilated = (dilated > 0).float()
        eroded = torch.nn.functional.conv2d(target, kernel, padding=1)
        eroded = (eroded >= 9).float()  # All 9 pixels in 3x3 must be 1
        edges = dilated - eroded
        
        # BCE on edge pixels only
        return self.bce(pred, edges)
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) — raw logits
            target: (B, 1, H, W) — binary ground truth
        """
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice_loss(pred, target)
        loss_edge = self.edge_loss(pred, target)
        
        total = (self.bce_weight * loss_bce + 
                 self.dice_weight * loss_dice + 
                 self.edge_weight * loss_edge)
        
        return total, {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'edge': loss_edge.item(),
            'total': total.item()
        }
```

### Why Each Loss Component

| Component | Optimizes | Without It... | Weight |
|-----------|----------|---------------|--------|
| **BCE** | Global pixel distribution | Dice alone can produce noisy, unstable masks | 1.0 |
| **Dice** | Region overlap (≈ F1 score) | Model predicts all-zeros (empty masks); 95% accuracy by ignoring tampering | 1.0 |
| **Edge** | Boundary sharpness | Mask boundaries are blurry halos instead of crisp edges | 0.5 |

---

## 4.7 Model Parameter Summary

```python
def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"Trainable:   {trainable:>12,}")
    print(f"Frozen (SRM): {frozen:>11,}")
    print(f"Total:       {total:>12,}")

count_parameters(model)
```

**Expected output**:
```
Trainable:     ~8,200,000
Frozen (SRM):      ~2,250
Total:         ~8,202,250
```

---

## 4.8 Architecture Diagram (For Notebook Markdown)

Include this in the notebook's architecture description cell:

```
┌─────────────────────────────────────────────────────────────┐
│                    TamperingDetector                         │
│                                                             │
│   Input: RGB Image (B, 3, 512, 512)                        │
│       │                                                     │
│       ├──────────────────┐                                  │
│       │                  ▼                                  │
│       │          SRM Filter Bank                            │
│       │          (30 fixed kernels)                         │
│       │                  │                                  │
│       │          Channel Reducer                            │
│       │          (30 → 3 channels)                          │
│       │                  │                                  │
│       └──────┬───────────┘                                  │
│              │                                              │
│        Concatenate (6 channels)                             │
│              │                                              │
│     ┌────────▼────────┐                                     │
│     │  U-Net Encoder   │                                    │
│     │ (EfficientNet-B1)│                                    │
│     │   ImageNet init  │                                    │
│     └────────┬────────┘                                     │
│              │ + Skip Connections                           │
│     ┌────────▼────────┐                                     │
│     │  U-Net Decoder   │                                    │
│     │ (5-stage upsample│                                    │
│     │  + concatenation)│                                    │
│     └────────┬────────┘                                     │
│              │                                              │
│     Segmentation Head                                       │
│     Conv2d(16, 1) → Logits                                  │
│              │                                              │
│   Output: Tampering Mask (B, 1, 512, 512)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4.9 Design Decisions to Explain in Notebook

When writing the architecture markdown cell, address these questions (the evaluators will wonder):

1. **Why U-Net over DeepLabV3+?** Skip connections preserve fine spatial detail needed for precise boundary localization. DeepLabV3+ ASPP is better for global context but uses more VRAM for marginal forensic benefit.

2. **Why EfficientNet-B1 over ResNet-34?** EfficientNet's compound scaling achieves better accuracy with fewer parameters. B1 (7.8M params) outperforms ResNet-34 (21.8M params) on ImageNet while using less memory.

3. **Why SRM instead of BayarConv?** SRM is fixed and requires zero training — reducing the risk of forensic preprocessing overfitting on our small dataset. BayarConv is learnable (better in theory) but adds a training variable. SRM is the safer choice for a 1-week project.

4. **Why 6-channel concatenation instead of two separate encoders?** A true dual-branch (two encoders) doubles VRAM usage. Concatenation at the input level keeps memory within T4 limits while still giving the encoder access to both feature types.

5. **Why sigmoid activation is NOT in the model?** `BCEWithLogitsLoss` is numerically more stable than `BCELoss(sigmoid(x))` because it uses the log-sum-exp trick internally. We apply sigmoid only during inference.
