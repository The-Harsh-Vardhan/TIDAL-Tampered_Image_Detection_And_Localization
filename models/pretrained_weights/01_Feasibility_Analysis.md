# Pretrained Models for Image Tampering Detection — Feasibility Analysis

---

## Executive Summary

**Should you use pretrained models? Yes — emphatically.**

The evidence is overwhelming from three independent sources:

1. **Your own project history:** v6.5 (pretrained ResNet34) achieved Tam-F1 = 0.41 — nearly **2x better** than the best from-scratch model (Tam-F1 = 0.22)
2. **Research literature:** Top-performing forensic methods (ME-Net, EMT-Net) use pretrained backbones. 19 of 21 surveyed papers use or reference pretrained encoders.
3. **The assignment itself:** "The choice of architecture is **entirely up to you**." No restriction on pretraining. The only constraint is T4 GPU compatibility.

The current ETASR CNN has a **hard ceiling** — it cannot produce pixel-level localization (the assignment's core requirement), and its ablation study has stalled after the augmentation experiment regressed. Pretrained models offer a path to both better classification AND localization in a single architecture.

---

## 1. Why Pretrained Models Are the Right Move

### 1.1 The Data Efficiency Problem

| Approach | Training Images | Parameters | Data:Param Ratio |
|----------|----------------|------------|------------------|
| ETASR CNN (current) | 8,829 | 29,520,034 | **1 : 3,343** |
| ResNet34 (frozen) + head | 8,829 | ~500,000 trainable | **1 : 57** |
| ResNet50 (frozen) + head | 8,829 | ~600,000 trainable | **1 : 68** |
| EfficientNet-B0 (frozen) + head | 8,829 | ~400,000 trainable | **1 : 45** |

The ETASR CNN trains 29.5M parameters from scratch on 8,829 images — a 1:3,343 data-to-parameter ratio. This is absurdly data-inefficient. The pretrained approach freezes the backbone and only trains ~500K parameters, giving a 60x better data efficiency ratio.

### 1.2 Project History Evidence

| Experiment | Architecture | Pretrained? | Tam-F1 | Test Accuracy |
|------------|-------------|-------------|--------|---------------|
| **v6.5** | **SMP UNet + ResNet34** | **ImageNet** | **0.4101** | **—** |
| vK.10.6 | Custom UNet | No | 0.2213 | — |
| vR.1.1 | ETASR 2-layer CNN | No | 0.8606* | 88.38%* |

\* vR.1.1 metrics are for **classification only** (binary Authentic/Tampered). v6.5's Tam-F1 is for **pixel-level segmentation** — a fundamentally harder task. The pretrained model achieved 0.41 F1 on pixel masks while the from-scratch models couldn't exceed 0.22.

### 1.3 What the Literature Says

| Paper | Backbone | Pretrained? | Best Metric |
|-------|----------|-------------|------------|
| ME-Net (P17) | ConvNeXt + ResNet-50 | Yes (ImageNet) | F1=0.905 (NIST16) |
| VGGNet+Capsule (P20) | VGGNet | Yes | 99.23% acc (CASIAv1) |
| P14 Survey | CNN-SVM (VGG-16 features) | Yes | 97.8% acc |
| ETASR (P7) | 2-layer CNN | No | 96.21% acc (claimed) |
| EMT-Net (P13) | Swin Transformer | No (custom) | AUC=0.987 (NIST) |

Most top-performing methods use pretrained encoders. The ETASR paper's claim that its 2-layer CNN outperforms VGG16 and ResNet101 is viewed skeptically — the project's own audit notes this is likely due to evaluation differences.

### 1.4 Assignment Alignment

The assignment requires:
- **Pixel-level tampered region masks** — requires an encoder-decoder (U-Net style), not just a classifier
- **Original / Ground Truth / Predicted / Overlay visualization** — requires spatial output
- **T4 GPU compatible** — all three candidates (ResNet34, ResNet50, EfficientNet-B0) run comfortably on T4

The ETASR CNN **cannot** produce localization. Pretrained encoder-decoders can. This is the strongest argument.

---

## 2. The ELA Compatibility Question

### 2.1 The Core Tension

ImageNet pretrained models expect 3-channel RGB images of natural scenes. ELA maps are synthetic images with fundamentally different statistics:

| Property | ImageNet RGB | ELA Maps |
|----------|-------------|----------|
| Content | Natural scenes, objects | Compression artifact patterns |
| Color distribution | Full RGB spectrum | Sparse, high-contrast brightness |
| Texture | Natural textures, edges | Block artifacts, noise patterns |
| Dynamic range | Full [0, 255] | Concentrated in low range, brightness-scaled |
| Spatial structure | Semantic objects | Compression block boundaries |

### 2.2 Three Input Strategies

**Strategy A: RGB Only (No ELA) — Recommended Starting Point**

Feed raw RGB images directly to the pretrained encoder. The encoder learns to detect tampering from visual inconsistencies (lighting, noise, edges, blending artifacts) rather than ELA artifacts.

| Pros | Cons |
|------|------|
| Pretrained features transfer perfectly | Loses ELA-specific signal |
| No domain mismatch | May miss compression-based tampering |
| Standard pipeline, well-understood | |

This is what v6.5 used (achieving Tam-F1 = 0.41).

**Strategy B: ELA as Input (Replace RGB)**

Feed 3-channel ELA maps to the pretrained encoder. The encoder's low-level features (edges, textures, gradients from the first few layers) still provide useful feature extraction, even though the input distribution differs from ImageNet.

| Pros | Cons |
|------|------|
| Leverages the ETASR paper's core preprocessing | Domain mismatch with pretrained weights |
| ELA explicitly highlights tampered regions | Low-level features may not transfer as well |
| Compatible with current pipeline | Higher-level features (object detectors) are useless |

**Research insight:** Several papers (including Transfer Learning studies) show that pretrained low-level features (conv1, conv2 — edges, corners, textures) transfer reasonably well even to non-natural images. The concern is with deeper layers.

**Strategy C: Dual-Branch (RGB + ELA)**

Two parallel encoders — one for RGB, one for ELA — with feature fusion before the decoder. This is what ME-Net (P17) does with ConvNeXt (RGB) + ResNet-50 (noise).

| Pros | Cons |
|------|------|
| Both RGB and ELA signals available | 2x model size |
| Best of both worlds | More complex to implement |
| Matches state-of-the-art approaches | May exceed T4 memory for larger encoders |

### 2.3 Recommendation

**Start with Strategy A (RGB only).** This gives the cleanest transfer and matches v6.5 (your project best). After establishing a baseline, test Strategy B (ELA only) as a single-variable ablation. Only attempt Strategy C if the assignment timeline allows.

---

## 3. Architecture Comparison

### 3.1 Candidate Summary

| Model | Params (total) | Params (trainable, frozen backbone) | ImageNet Top-1 | Inference Speed (T4) | Memory (batch=16, 384×384) |
|-------|---------------|-------------------------------------|----------------|---------------------|---------------------------|
| **ResNet-34** | 21.8M | ~500K | 73.3% | Fast | ~3 GB |
| **ResNet-50** | 25.6M | ~600K | 76.1% | Medium | ~4 GB |
| **EfficientNet-B0** | 5.3M | ~400K | 77.1% | Medium | ~2.5 GB |
| **EfficientNet-B4** | 19.3M | ~500K | 82.9% | Slow | ~6 GB |
| ETASR CNN (current) | 29.5M | 29.5M (all) | N/A | Very fast | ~1.5 GB |

### 3.2 ResNet-34

**The proven choice for this project.**

```
Architecture: 34-layer residual network
Skip connections: Identity shortcuts (no bottleneck)
Feature pyramid: [64, 128, 256, 512] channels at 4 resolutions
Decoder: UNet-style with skip connections from encoder
```

| Pros | Cons |
|------|------|
| **Already validated** — v6.5 achieved Tam-F1=0.41 | Older architecture (2015) |
| Well-documented in SMP library | Fewer features than ResNet-50 |
| Small enough for comfortable T4 training | No squeeze-excite or modern blocks |
| Fast inference | |
| Rich ecosystem (pretrained weights, tutorials) | |

**Verdict: PRIMARY RECOMMENDATION.** Re-establish the v6.5 baseline before trying anything else. The project has proven this works.

### 3.3 ResNet-50

**The standard baseline in computer vision.**

```
Architecture: 50-layer residual network with bottleneck blocks
Skip connections: 1×1 -> 3×3 -> 1×1 bottleneck
Feature pyramid: [256, 512, 1024, 2048] channels at 4 resolutions
Decoder: UNet-style with skip connections
```

| Pros | Cons |
|------|------|
| Stronger feature representation than ResNet-34 | ~4M more parameters |
| Most widely used encoder in forensic literature | Slower training |
| Bottleneck blocks are more parameter-efficient | Diminishing returns over ResNet-34 for small datasets |
| Used by ME-Net (P17, F1=0.905) | More memory consumption |

**Verdict: SECONDARY OPTION.** Test after ResNet-34 if accuracy is insufficient. The marginal improvement over ResNet-34 may not justify the extra cost on a small dataset.

### 3.4 EfficientNet-B0

**The parameter-efficient modern choice.**

```
Architecture: Compound-scaled network with MBConv blocks
Key feature: Squeeze-excite attention + inverted residuals
Feature pyramid: [16, 24, 40, 112, 320] channels at 5 resolutions
Decoder: UNet-style or FPN
```

| Pros | Cons |
|------|------|
| Best accuracy-per-parameter ratio | Not validated in this project |
| Only 5.3M total params (vs 21.8M for ResNet-34) | Squeeze-excite blocks add complexity |
| Modern architecture with attention | Less proven for forensics than ResNets |
| Lower memory footprint | Different feature pyramid structure than ResNet |
| Recommended by recent forensic survey (P9) | May need SMP compatibility check |

**Verdict: EXPERIMENTAL OPTION.** Interesting for its efficiency but unproven in this project's context. Test as a third candidate after ResNet-34 and ResNet-50.

### 3.5 Head-to-Head Comparison for This Project

| Factor | ResNet-34 | ResNet-50 | EfficientNet-B0 |
|--------|-----------|-----------|-----------------|
| **Project evidence** | ✅ Proven (v6.5) | ❌ Untested | ❌ Untested |
| **Literature support** | ✅ Common in forensics | ✅ ME-Net uses it | ⚠️ Survey mentions it |
| **Parameter efficiency** | Good | Lower | Best |
| **T4 compatibility** | ✅ Comfortable | ✅ Fits | ✅ Comfortable |
| **SMP support** | ✅ Native | ✅ Native | ✅ Native |
| **Training speed** | Fast | Medium | Medium |
| **Feature quality** | Good | Better | Good-Best |
| **Risk** | Very low | Low | Medium |
| **Recommendation** | **1st choice** | 2nd choice | 3rd choice |

---

## 4. Frozen vs Fine-Tuned Weights

### 4.1 Three Approaches

| Strategy | What's Trained | Total Trainable Params | Risk |
|----------|---------------|----------------------|------|
| **Fully Frozen** | Only decoder + classification head | ~500K | Very low |
| **Gradual Unfreeze** | Freeze initially, unfreeze last 1-2 blocks after N epochs | ~2-5M | Low |
| **Full Fine-Tune** | Entire model with differential LR | ~22M+ | Medium |

### 4.2 Recommendation: Start Frozen, Then Gradually Unfreeze

**Phase 1 (epochs 1–10):** Encoder fully frozen. Only train the decoder and classification head. This prevents destroying pretrained features while the decoder learns to use them.

**Phase 2 (epochs 11–25):** Unfreeze the last 1-2 encoder blocks with a very low learning rate (10x lower than decoder). This allows the encoder to slightly adapt to the ELA/forensic domain.

This is exactly what v6.5 used (differential LR: encoder 1e-4, decoder 1e-3) and it produced the project's best result.

### 4.3 Why Not Always Fine-Tune Everything?

The vK.11-12 synthesis experiments showed what happens when pretrained features are carelessly modified:

> "Encoder features destroyed upon unfreeze due to conflicting gradients from multi-objective loss."

With only 8,829 training images, an unfrozen 22M-parameter encoder will overfit rapidly. Frozen weights act as a powerful regularizer — the encoder provides stable, general-purpose features that the lightweight decoder learns to interpret.

---

## 5. Classification vs Localization

### 5.1 The ETASR Approach (Current)

```
ELA Image -> CNN -> [P(Authentic), P(Tampered)]
```

Output: A single binary label. **No spatial information.**

### 5.2 The Pretrained Encoder-Decoder Approach

```
Image -> Encoder (ResNet34) -> Bottleneck -> Decoder (UNet) -> Pixel Mask (H×W)
```

Output: A full-resolution binary mask where each pixel is classified as authentic or tampered. **This is what the assignment requires.**

### 5.3 Why This Matters

The assignment explicitly says:

> "Train a model to predict **tampered regions**."
> "Visual results: **Original / Ground Truth / Predicted / Overlay**"

The ETASR CNN can never produce these outputs — it was designed for whole-image classification, not pixel-level localization. A pretrained encoder-decoder inherently produces spatial outputs.

### 5.4 Can We Keep ELA With Localization?

Yes. Two approaches:

1. **ELA as input:** Feed ELA maps to the U-Net encoder. The model learns to map ELA brightness patterns to pixel-level tampered masks.

2. **ELA as auxiliary channel:** Feed 4-channel input (RGB + ELA) to the encoder. Requires modifying the first conv layer (average pretrained RGB weights to initialize the 4th channel).

Both are valid ablation experiments for after the RGB baseline is established.

---

## 6. Integration With the Current Ablation Study

### 6.1 Current Status

The ETASR ablation study is at vR.1.3 (class weights, pending run). The planned roadmap goes up to vR.2.0 (ELA localization).

### 6.2 Proposed Two-Track Strategy

**Track 1: Continue ETASR Ablation (Classification)**

Continue vR.1.3 → vR.1.4 → ... → vR.1.7. This builds a thorough ablation study documenting what works and what doesn't for the ETASR paper reproduction. This has academic/documentation value.

**Track 2: Launch Pretrained Encoder-Decoder (Localization)**

In parallel, start a new version track `vR.P.x` for the pretrained approach:

| Version | Change | Expected Impact |
|---------|--------|-----------------|
| vR.P.0 | ResNet34 + UNet, RGB input, frozen encoder | Establish localization baseline |
| vR.P.1 | Gradual unfreeze (last 2 encoder blocks) | +2-5% F1 |
| vR.P.2 | ELA as input (replace RGB) | Test ELA with pretrained features |
| vR.P.3 | 4-channel input (RGB + ELA) | Test combined signal |

### 6.3 Why Both Tracks?

1. **The ETASR track** demonstrates your understanding of the paper, controlled experimentation, and ablation methodology — valuable for the assignment writeup
2. **The pretrained track** actually achieves the assignment's core requirement (localization) and produces competitive results

The final submission notebook would use the best pretrained model, with the ETASR ablation study documented as supporting evidence of your experimental process.

---

## 7. Implementation Quick-Start

### 7.1 Using SMP (Segmentation Models PyTorch)

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',   # Pretrained on ImageNet
    in_channels=3,                # RGB input (or 1 for ELA grayscale)
    classes=1,                    # Binary mask output
    activation='sigmoid'          # Pixel-wise probability
)

# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False
```

### 7.2 Using Keras Applications

```python
from tensorflow.keras.applications import ResNet50, EfficientNetB0

# ResNet50 encoder
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(384, 384, 3))
base_model.trainable = False  # Freeze

# Build U-Net decoder on top
# ... (custom decoder layers)
```

### 7.3 Key Configuration (From v6.5 Lessons)

```python
# What worked in v6.5:
IMAGE_SIZE = (384, 384)         # Higher resolution than ETASR's 128x128
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ENCODER_LR = 1e-4              # Low LR for pretrained features
DECODER_LR = 1e-3              # Higher LR for decoder
LOSS = BCEDiceLoss()            # Combination loss for segmentation
BATCH_SIZE = 16                 # Fits on T4 GPU
EPOCHS = 25
```

---

## 8. Verdict and Recommendation

### The Bottom Line

| Question | Answer |
|----------|--------|
| Should you use pretrained models? | **Yes** |
| Which one first? | **ResNet-34** (proven in v6.5) |
| Frozen or fine-tuned? | **Start frozen, gradually unfreeze** |
| RGB or ELA input? | **Start with RGB, then test ELA as ablation** |
| Classification or localization? | **Localization** (encoder-decoder, required by assignment) |
| Continue ETASR ablation? | **Yes, in parallel** (documentation value) |
| Framework? | **SMP (PyTorch)** or **Keras Applications** |

### Priority Order

1. **ResNet-34 + UNet, RGB, frozen encoder** — re-establish v6.5 baseline
2. **Gradual unfreeze** — squeeze out more performance
3. **ResNet-50 + UNet** — test if deeper encoder helps
4. **EfficientNet-B0 + UNet** — test parameter efficiency
5. **ELA input variants** — test if ELA adds value over RGB with pretrained

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Pretrained features don't transfer to forensics | Very low (disproven by v6.5) | High | Start with proven ResNet-34 |
| T4 GPU memory overflow | Low | Medium | Use batch_size=8-16, image_size=384 |
| ELA destroys pretrained features | Medium | Medium | Test ELA as separate ablation, not as first experiment |
| Overfitting with unfrozen encoder | Medium | Medium | Freeze first, then gradual unfreeze with low LR |
| Worse than ETASR CNN | Very low | Low | Unlikely given v6.5 evidence |
