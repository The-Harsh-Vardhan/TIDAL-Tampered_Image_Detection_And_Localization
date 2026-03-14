# 2. Possible Solutions: Tampered Image Detection & Localization

## 2.1 Solution Landscape Overview

Image tampering detection and localization has evolved through four distinct generations of approaches. Understanding each is critical for making an informed architecture choice — and for demonstrating to evaluators that the final selection was deliberate, not arbitrary.

```
Generation 1: Hand-Crafted Features (Pre-2016)
    ↓
Generation 2: Single-Stream Deep Learning (2016-2019)
    ↓
Generation 3: Multi-Stream / Multi-Task Networks (2019-2023)
    ↓
Generation 4: Transformer + Multimodal Forensics (2023-2025)
```

---

## 2.2 Generation 1: Traditional Computer Vision Approaches

### 2.2.1 Error Level Analysis (ELA)
**How it works**: Re-saves the image at a known JPEG quality level and computes the pixel-wise difference between the original and re-saved image. Authentic regions compress uniformly; tampered regions show different error levels because they have a different compression history.

**Pros**: Simple, fast, no training required, interpretable  
**Cons**: Fails on uncompressed images (PNG), highly sensitive to quality factor, easily defeated by re-compressing the forged image, produces noisy outputs  
**Verdict**: Useful as a quick sanity check but not reliable as a standalone solution

### 2.2.2 Noise Variance Analysis
**How it works**: Estimates the local noise variance across image blocks. Authentic images have a consistent noise floor determined by the camera sensor. Tampered regions exhibit different noise variance due to processing, scaling, or coming from a different source.

**Pros**: Camera-physics grounded, effective for splicing from different sources  
**Cons**: Fails when source and target have similar noise levels, sensitive to denoising post-processing  
**Verdict**: Valuable as a feature input to deep learning models (SRM filters formalize this concept)

### 2.2.3 CFA Pattern Detection
**How it works**: Analyzes the Color Filter Array (Bayer) demosaicing patterns. Each camera applies a specific demosaicing algorithm that creates periodic correlations. Tampering disrupts these correlations at the manipulation boundary.

**Pros**: Physics-based, hard to forge, detects boundary artifacts precisely  
**Cons**: Computationally expensive, fails after heavy post-processing, assumes a specific CFA layout  
**Verdict**: Powerful forensic signal — modern networks (BayarConv) learn this automatically

### 2.2.4 Copy-Move Detection via Keypoint Matching
**How it works**: Extracts local feature descriptors (SIFT, SURF, ORB) and matches them across the image. Clusters of matching keypoints in non-overlapping regions indicate copy-move forgery.

**Pros**: Robust to geometric transforms (rotation, scaling of copied region)  
**Cons**: Only detects copy-move (not splicing/inpainting), fails on smooth/textureless regions, high false positive rate  
**Verdict**: Specialist tool — useful as a baseline for copy-move but not a general solution

---

## 2.3 Generation 2: Single-Stream Deep Learning

### 2.3.1 Standard U-Net on RGB
**Architecture**: Encoder-decoder with skip connections. Takes RGB image as input, outputs a single-channel binary mask predicting tampered pixels.

**How it works**: The encoder (typically ResNet or VGG backbone, pre-trained on ImageNet) extracts multi-scale features. The decoder upsamples and combines features via skip connections to produce a pixel-level prediction.

**Pros**:
- Well-understood, battle-tested architecture
- Pre-trained encoders provide strong feature extraction
- `segmentation_models_pytorch` (SMP) provides clean, one-line implementation
- Trains well on limited data with transfer learning

**Cons**:
- Learns **semantic** features (shapes, textures, objects) rather than **forensic** features (noise, compression artifacts)
- Vulnerable to "semantic bias" — may learn to flag specific object categories rather than actual tampering
- Limited receptive field for detecting copy-move where source/target are far apart
- Struggles with extreme class imbalance (tiny tampered regions in large images)

**Expected Performance**: Pixel-F1 ~40-55% on CASIA v2.0 with standard BCE loss  
**Verdict**: Solid baseline but insufficient without forensic feature augmentation

### 2.3.2 Standard DeepLabV3+ on RGB
**Architecture**: Atrous Spatial Pyramid Pooling (ASPP) with dilated convolutions for multi-scale context, plus an encoder-decoder for boundary refinement.

**Pros**: Larger receptive field than U-Net (handles global context better), strong boundary delineation via ASPP, efficient with pre-trained encoders  
**Cons**: Same semantic bias problem as U-Net; heavier compute than U-Net for marginal forensic gains  
**Expected Performance**: Pixel-F1 ~42-58% on CASIA v2.0  
**Verdict**: Slightly better global context than U-Net but not forensically specialized

---

## 2.4 Generation 3: Multi-Stream / Multi-Task Networks

### 2.4.1 MVSS-Net++ (Multi-View Multi-Scale)
**Architecture**: Dual-branch network with:
- **Edge-Supervised Branch (ESB)**: Focuses on boundary artifacts at tampering edges
- **Noise-Sensitive Branch (NSB)**: Processes SRM noise residuals to detect noise distribution mismatches

Both branches share a ResNet-50 backbone. The outputs are fused through a multi-scale feature aggregation module, producing both an image-level score and a pixel-level mask.

**Pros**:
- Explicitly separates semantic and forensic analysis paths
- Edge supervision forces sharp localization boundaries
- Noise branch catches splicing artifacts invisible in RGB
- Strong published benchmarks (Pixel-F1 ~48.2% on COVERAGE, strong on CASIA)

**Cons**:
- ResNet-50 backbone is memory-heavy on T4 GPU (needs careful batch size management)
- Original implementation requires significant custom code
- Two-branch training can be unstable without careful loss balancing

**Expected Performance**: Pixel-F1 ~65-75% on CASIA v2.0  
**Verdict**: Strong contender — represents the forensic-aware CNN paradigm well

### 2.4.2 BusterNet (Copy-Move Specialist)
**Architecture**: Dual-branch with a Manipulation Detection Branch and a Similarity Detection Branch. The similarity branch specifically detects copy-move by finding matching feature regions within the same image.

**Pros**: Best-in-class for copy-move detection, explicitly models within-image similarity  
**Cons**: Weak on splicing (only one branch is relevant), complex training procedure, not available in standard libraries  
**Verdict**: Specialist — useful for bonus points on copy-move but not a primary architecture

### 2.4.3 CAT-Net (Compression Artifact Tracing)
**Architecture**: Dual-stream processing RGB image and raw DCT coefficients from the JPEG file. Detects compression artifact inconsistencies that indicate double-compression from tampering.

**Pros**: Catches compression-level forgeries invisible to RGB-only methods  
**Cons**: Requires access to raw JPEG data (not just decoded RGB), architecture is complex to implement from scratch, less effective on uncompressed images  
**Verdict**: Powerful forensic signal — the DCT concept can be borrowed even if the full architecture is not implemented

---

## 2.5 Generation 4: Transformer + Multimodal Forensics

### 2.5.1 SegFormer-Based Forensic Model
**Architecture**: Hierarchical Vision Transformer encoder (Mix Transformer / MiT) with a lightweight all-MLP decoder. Can be paired with forensic preprocessing (SRM/BayarConv) as a dual-stream system.

**Key Variants**:
- **SegFormer-B0**: 3.7M parameters — minimal, fast, fits easily on T4
- **SegFormer-B1**: 13.7M parameters — better accuracy, still T4-compatible
- **SegFormer-B2**: 27.3M parameters — needs careful memory management on T4

**Pros**:
- **Hierarchical multi-scale features** — captures both fine-grained boundary artifacts and global structure
- **Self-attention** — models long-range dependencies critical for copy-move detection (source and target can be far apart)
- No positional encoding (uses overlapping patch embeddings) — naturally handles variable resolution
- Lightweight MLP decoder is faster than CNN decoders
- Pre-trained checkpoints readily available from HuggingFace

**Cons**:
- More complex to fine-tune than CNN-based models
- Requires more careful learning rate scheduling (warm-up is critical)
- Not available in `segmentation_models_pytorch` — needs custom implementation or HuggingFace `transformers`

**Expected Performance**: Pixel-F1 ~70-82% on CASIA v2.0 (with forensic preprocessing)  
**Verdict**: Current best balance of performance and efficiency for Colab T4

### 2.5.2 TruFor (Trust Through Forensic Features)
**Architecture**: SegFormer encoder fused with **Noiseprint++** — a learned camera-specific noise fingerprint extracted by a side network. Outputs both a localization map and a **reliability map** (model confidence).

**Pros**: SOTA localization accuracy, confidence-aware predictions, handles diverse forgery types  
**Cons**: Noiseprint++ requires a separate pre-trained model, full pipeline is complex, higher memory footprint  
**Verdict**: Gold standard — can borrow concepts (noise fingerprinting, reliability maps) even if full architecture is too complex for 1-week timeline

### 2.5.3 REFORGE (Reinforcement Forensic Segmentation)
**Architecture**: U-Net segmentation combined with a reinforcement learning (RL) agent that iteratively refines the predicted mask. Each pixel is treated as an agent maximizing localization reward.

**Pros**: Iterative refinement produces cleaner mask boundaries, combines classification + segmentation  
**Cons**: RL training loop is slow and unstable, significantly more complex to implement and debug  
**Verdict**: Fascinating research — impractical for a 1-week assignment

### 2.5.4 FakeShield (Explainable Multimodal Forensics)
**Architecture**: Combines a vision encoder with a large language model to provide both visual localization masks and textual explanations for its forensic judgments (e.g., "inconsistent lighting direction in the upper-right region").

**Pros**: Provides evidence-based explanations (useful for legal/forensic applications), multimodal understanding  
**Cons**: Requires LLM inference (expensive, slow), far too heavy for T4 GPU training, research-stage only  
**Verdict**: Reference for "industry relevance" discussion only — not implementable in this project

---

## 2.6 Comparison Matrix

| Architecture | Gen | Forgery Types | Colab T4 Feasible | Pixel-F1 (CASIA v2.0) | Implementation Complexity | 1-Week Feasible | Key Limitation |
|-------------|-----|--------------|-------------------|----------------------|--------------------------|----------------|---------------|
| ELA (Traditional) | 1 | Splicing | Yes | ~20-30% | Low | Yes | Fragile; JPEG-dependent |
| U-Net (RGB only) | 2 | All (weak) | Yes | ~40-55% | Low (SMP) | Yes | Semantic bias; no forensic awareness |
| DeepLabV3+ (RGB) | 2 | All (weak) | Yes | ~42-58% | Low (SMP) | Yes | Same as U-Net + heavier compute |
| MVSS-Net++ | 3 | Splicing + CM | Tight | ~65-75% | High (custom) | Difficult | Custom code; memory-heavy |
| BusterNet | 3 | Copy-Move | Yes | N/A (CM only) | High | No | Specialist only |
| CAT-Net | 3 | Splicing | Tight | ~60-70% | High | No | Needs raw JPEG DCT access |
| **U-Net/DLV3+ + SRM** | **2+3** | **All** | **Yes** | **~60-72%** | **Medium (SMP)** | **Yes** | **Fixed forensic features** |
| **SegFormer + SRM** | **4** | **All** | **Yes (B0/B1)** | **~70-82%** | **Medium-High** | **Yes (tight)** | **Needs custom fusion code** |
| TruFor | 4 | All | Tight | ~75-85% | Very High | No | Noiseprint++ dependency |
| REFORGE | 4 | All | No | ~78-86% | Very High | No | RL training instability |
| FakeShield | 4 | All | No | N/A | Extreme | No | Requires LLM; research-only |

**Legend**: CM = Copy-Move, SMP = segmentation_models_pytorch library, Bold rows = viable candidates for this project

---

## 2.7 Key Takeaways

1. **Pure RGB models are insufficient**: Without forensic preprocessing (SRM/BayarConv/DCT), even the best architectures learn semantic shortcuts instead of actual tampering artifacts.

2. **The forensic preprocessing layer is more important than the backbone**: Adding SRM noise residuals to a simple U-Net yields larger performance gains than switching from U-Net to a Transformer on RGB alone.

3. **Class imbalance is the silent killer**: The choice of loss function (Dice + BCE + Edge) matters as much as the architecture. Many architectures fail not because they can't detect tampering, but because the loss function lets them ignore it.

4. **Two viable paths exist for this project**:
   - **Path A (Practical)**: U-Net or DeepLabV3+ with SRM preprocessing via SMP — lower risk, proven, easier to implement
   - **Path B (Ambitious)**: SegFormer-B1 with SRM dual-stream — higher ceiling, more impressive to evaluators, but more complex

5. **The decision should be made based on implementation confidence**, not theoretical superiority. A well-implemented U-Net+SRM will score higher than a buggy SegFormer.
