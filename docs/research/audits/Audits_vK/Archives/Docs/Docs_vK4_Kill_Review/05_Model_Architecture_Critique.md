# 05 — Model Architecture Critique

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Architecture: Custom U-Net with Classifier Head

### Encoder
- `DoubleConv(3→64) → Down(64→128) → Down(128→256) → Down(256→512) → Down(512→1024)`
- Standard MaxPool2d(2) downsampling
- No pretrained weights — trained from scratch

### Decoder
- `Up(1024→512) → Up(512→256) → Up(256→128) → Up(128→64) → Conv2d(64→1)`
- ConvTranspose2d upsampling with skip connections

### Classifier Head
- `AdaptiveAvgPool2d → Flatten → Linear(1024→512) → ReLU → Dropout(0.5) → Linear(512→2)`

---

## Critique

### 1. No Pretrained Encoder — 🔴 CRITICAL

The single biggest architectural weakness. The encoder is trained from scratch on ~7K CASIA images. Modern tamper detection heavily relies on **pretrained feature extractors** (ImageNet ResNet34/50, EfficientNet) because:
- Tampered regions differ subtly from authentic regions — high-level semantic features help
- CASIA v2 is a small dataset (~7K images) — training a deep encoder from scratch risks severe overfitting
- v8 uses `smp.Unet(encoder_name='resnet34', encoder_weights='imagenet')` which has 24M parameters with pretrained features

The custom U-Net trains ~31M parameters from scratch on a small dataset. This is almost certainly **underfitting on feature quality and overfitting on dataset artifacts.**

### 2. No Justification for Custom Architecture — ⚠️ MEDIUM

The notebook states the architecture is "preserved from vK.2" but never explains **why** a custom from-scratch U-Net was chosen over SMP with pretrained encoder. The assignment requirement asks for "reasoned architecture decisions" — this requirement is not met.

### 3. Single-Scale Classification Head — ⚠️ MEDIUM

The classifier head operates only on the bottleneck (1024-dim features at 16×16). Multi-scale classification (fusing features from multiple encoder levels) could provide richer evidence for the tamper/authentic decision.

### 4. No Attention Mechanisms — ⚠️ LOW

Modern segmentation architectures (Attention U-Net, TransUNet) use attention gates to focus on relevant features. The vanilla U-Net skip connections pass everything, including noise.

### 5. Dual-Head Output Asymmetry — ⚠️ LOW

The model outputs `(cls_logits, seg_logits)` — classification logits are 2D (binary), segmentation logits are 1D (binary mask). The training loss combines both with weights ALPHA=1.5 / BETA=1.0. There's no mechanism to prevent the classification head from dominating gradients in the shared encoder, which could hurt segmentation quality.

---

## Parameter Count Check

| Component | Approximate Parameters |
|---|---|
| Encoder (inc+down1-4) | ~18M |
| Decoder (up1-4+outc) | ~12.5M |
| Classifier head | ~0.5M |
| **Total** | **~31M** |

This is a large model for ~7K images and no pretraining. The parameter-to-sample ratio suggests high overfitting risk.

---

## Verdict

The architecture is functional but suboptimal. Using a pretrained encoder (as v8 does) would be strictly better given the dataset size. The lack of architectural justification weakens the submission.

**Severity: HIGH** — architecture is the primary bottleneck for model quality.
