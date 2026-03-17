# FakeShield Pruning Analysis: From Research Model to Colab T4 Assignment

---

## 1. FakeShield Architecture Analysis

### 1.1 Overview

FakeShield is a multi-modal framework for **Explainable Image Forgery Detection and Localization (e-IFDL)**. It extends traditional IFDL by not only detecting and localizing forgeries but also providing natural language explanations for its decisions. The system is published at ICLR 2025 and handles three tampering types: PhotoShop (copy-move, splicing, removal), DeepFake (FaceApp), and AIGC-Editing (Stable Diffusion inpainting).

### 1.2 Component-by-Component Analysis

#### A. Domain Tag Generator (DTG)

**Purpose:** Classifies input images into one of three tampering domains (PhotoShop, DeepFake, AIGC-Editing) to resolve data domain conflicts during training.

**Architecture:** A classifier `G_dt` that takes the original image `I_ori` and produces a domain tag `T_tag` using the template: *"This is a suspected {PS/DeepFake/AIGC}-tampered picture."*

**Training:** Full parameter training with cross-entropy loss.

**Why it matters:** Different tampering methods leave fundamentally different artifacts. PS leaves edge artifacts, DeepFake causes facial blurring, AIGC-Editing produces disordered textures. The DTG tells the LLM which type of artifacts to look for.

#### B. DTE-FDM (Domain Tag-guided Explainable Forgery Detection Module)

**Purpose:** Takes a suspected image and instruction text, produces detection results (real/fake), location description, and judgment basis in natural language.

**Architecture:**
```
I_ori → Image Encoder (CLIP ViT-L/14-336) → F_enc → Linear Projection → F_proj → T_img
T_tag (from DTG) + T_img + T_ins → LLM (LLaVA-v1.5-13B with LoRA) → O_det
```

**Components:**
- **Image Encoder:** CLIP ViT-L/14-336 (frozen) — extracts visual features
- **Linear Projection:** Maps CLIP features to LLM embedding space
- **LLM:** LLaVA-v1.5-13B backbone with LoRA fine-tuning (rank=128, alpha=256)

**Output:** `O_det` — a natural language string containing:
1. Detection result (tampered/authentic)
2. Description of tampered area location
3. Judgment basis (artifacts, semantic errors)

**Loss:** `ℓ_det = ℓ_ce(Ô_det, O_det) + λ · ℓ_ce(T̂_tag, T_tag)`

#### C. MFLM (Multi-modal Forgery Localization Module)

**Purpose:** Converts the textual description from DTE-FDM into a precise binary segmentation mask.

**Architecture:**
```
T_img + O_det → Tamper Comprehension Module (LLM encoder) → <SEG> token → h_<SEG>
I_ori → SAM Encoder (ViT-H) → E_mid
h_<SEG> + E_mid → SAM Decoder → M_loc (binary mask)
```

**Components:**
- **Tamper Comprehension Module (TCM):** An LLM-based encoder that processes image tokens and tampered description text, producing a `<SEG>` token embedding
- **Text Projection Layer:** MLP (Linear → ReLU → Linear → Dropout) that maps LLM hidden states to SAM prompt space (dim 512)
- **SAM ViT-H:** Full Segment Anything Model with image encoder (ViT-H, 1280 dim, 32 depth) and mask decoder
- **SAM Prompt Encoder:** Takes `h_<SEG>` as text embed prompt
- **SAM Mask Decoder:** Two-way transformer that generates segmentation mask

**Loss:** `ℓ_loc = ℓ_ce(ŷ_txt, y_txt) + α · ℓ_bce(M̂_loc, M_loc) + β · ℓ_dice(M̂_loc, M_loc)`

#### D. Role of SAM (Segment Anything Model)

SAM serves as the **visual segmentation backbone** in MFLM:
- **SAM Encoder (ViT-H):** Processes the original image at 1024×1024 resolution, producing 64×64 feature maps with 256 channels
- **SAM Prompt Encoder:** Converts `h_<SEG>` embedding into sparse/dense prompt embeddings
- **SAM Mask Decoder:** Two-way transformer (depth=2) that cross-attends between image features and prompt embeddings, outputting the final mask
- The encoder is **frozen** during training; only the mask decoder is fine-tuned

#### E. Role of LLM

LLMs serve **dual roles** in FakeShield:
1. **In DTE-FDM:** LLaVA-v1.5-13B acts as the core reasoning engine — it processes visual tokens + domain tag + instruction to generate detection results and explanations via autoregressive prediction
2. **In MFLM (TCM):** A separate LLM instance acts as an encoder to align long-text descriptions with visual features, producing the `<SEG>` prompt for SAM

Both are fine-tuned with LoRA (different ranks for each module).

### 1.3 Architecture Diagram (Text Form)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FakeShield Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Image (I_ori)                                                │
│       │                                                             │
│       ├──────────────────┐                                          │
│       │                  │                                          │
│       ▼                  ▼                                          │
│  ┌─────────┐     ┌──────────────┐                                   │
│  │  Domain  │     │ CLIP ViT-L   │                                   │
│  │  Tag     │     │ (Image       │                                   │
│  │Generator │     │  Encoder)    │                                   │
│  │ (DTG)    │     └──────┬───────┘                                   │
│  └────┬─────┘            │                                          │
│       │                  ▼                                          │
│       │           ┌──────────────┐                                   │
│       │           │Linear Project│                                   │
│       │           └──────┬───────┘                                   │
│       │                  │                                          │
│       │    T_tag         │  T_img          T_ins (instruction)      │
│       │                  │                    │                      │
│       └──────────┐       │    ┌───────────────┘                      │
│                  ▼       ▼    ▼                                      │
│          ┌───────────────────────────┐                               │
│          │   DTE-FDM                 │                               │
│          │   LLaVA-v1.5-13B (LoRA)  │                               │
│          │   Autoregressive LLM      │                               │
│          └───────────┬───────────────┘                               │
│                      │                                              │
│                      ▼  O_det (text: detection + explanation)       │
│                      │                                              │
│       ┌──────────────┼──────────────┐                               │
│       │              │              │                                │
│       ▼              ▼              ▼                                │
│  ┌─────────┐  ┌────────────┐  ┌──────────────┐                      │
│  │ T_img   │  │  O_det     │  │  I_ori       │                      │
│  └────┬────┘  └─────┬──────┘  └──────┬───────┘                      │
│       │             │                │                               │
│       └──────┐      │                ▼                               │
│              ▼      ▼         ┌──────────────┐                       │
│       ┌────────────────┐      │  SAM Encoder │                       │
│       │  MFLM          │      │  (ViT-H)     │                       │
│       │  TCM (LLM      │      └──────┬───────┘                       │
│       │   Encoder)     │             │  E_mid                        │
│       └───────┬────────┘             │                               │
│               │  h_<SEG>             │                               │
│               │  (MLP projection)    │                               │
│               └──────────┐           │                               │
│                          ▼           ▼                               │
│                   ┌──────────────────────┐                           │
│                   │  SAM Decoder         │                           │
│                   │  (Prompt + Mask Dec) │                           │
│                   └──────────┬───────────┘                           │
│                              │                                      │
│                              ▼                                      │
│                        M_loc (binary mask)                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Data Flow Summary

1. **Image → DTG:** Classifies tampering domain → domain tag text
2. **Image → CLIP → Projection:** Visual features → image tokens
3. **Tag + Tokens + Instruction → LLM:** Generates explanation text (O_det)
4. **Tokens + O_det → TCM:** Produces <SEG> embedding (h_<SEG>)
5. **Image → SAM Encoder:** Produces intermediate visual features (E_mid)
6. **h_<SEG> + E_mid → SAM Decoder:** Generates tampered region mask (M_loc)

### 1.5 Model Size Estimates

| Component | Parameters | VRAM (fp16) |
|-----------|-----------|-------------|
| CLIP ViT-L/14-336 | ~304M | ~0.6 GB |
| LLaVA-v1.5-13B (DTE-FDM) | ~13B | ~26 GB |
| Domain Tag Generator | ~25M (ResNet-based) | ~0.05 GB |
| TCM (LLM in MFLM) | ~7-13B | ~14-26 GB |
| SAM ViT-H Encoder | ~632M | ~1.3 GB |
| SAM Mask Decoder | ~4M | ~0.01 GB |
| Text Projection MLP | ~2M | ~0.004 GB |
| **Total** | **~20-27B** | **~42-54 GB** |

**Colab T4 has 16 GB VRAM.** The full FakeShield is completely infeasible.

---

## 2. Assignment Gap Analysis

### 2.1 Requirement Matching Table

| FakeShield Component | Role in FakeShield | Required for Assignment? | Keep / Remove / Simplify | Justification |
|---|---|---|---|---|
| **CLIP Image Encoder** | Extracts visual features for LLM consumption | Yes (visual feature extraction) | **Simplify** → Use as standalone backbone | Core vision feature extractor; can be repurposed as encoder backbone without LLM |
| **Linear Projection (mm_projector)** | Maps CLIP features to LLM embedding space | No | **Remove** | Only needed for LLM token space alignment |
| **Domain Tag Generator (DTG)** | Classifies tampering type (PS/DF/AIGC) | Partially | **Simplify** → Binary classifier head | Assignment only needs binary detection (real/fake), not 3-class domain classification. Repurpose as detection head |
| **LLM (LLaVA-v1.5-13B) in DTE-FDM** | Generates text explanations | No | **Remove** | 13B LLM cannot fit on T4; text explanation is NOT required by assignment |
| **LoRA adapters for DTE-FDM** | Efficient LLM fine-tuning | No | **Remove** | LLM is removed |
| **GPT-4o Description Pipeline** | Creates MMTD-Set text annotations | No | **Remove** | Assignment doesn't require text generation or multi-modal datasets |
| **Tamper Comprehension Module (TCM)** | LLM encoder for text→prompt alignment | No | **Remove** | Without text explanation branch, TCM has no input; we'll use direct feature projection instead |
| **Text Projection MLP** | Maps TCM output to SAM prompt space | Yes (concept) | **Simplify** → Feature projection from encoder to SAM | Replace TCM output with direct encoder feature projection |
| **SAM ViT-H Encoder** | Encodes image for segmentation | Yes | **Simplify** → SAM ViT-B | ViT-H (632M) too large; ViT-B (91M) fits T4 and retains segmentation capability |
| **SAM Prompt Encoder** | Converts prompts to embeddings | Yes | **Simplify** → Use learned embedding prompt | Keep but use learned prompt instead of text-derived prompt |
| **SAM Mask Decoder** | Generates segmentation masks | Yes | **Keep** | Core segmentation component, only 4M params |
| **Dice Loss** | Segmentation loss | Yes | **Keep** | Directly required; same loss used in assignment |
| **BCE Loss** | Segmentation loss | Yes | **Keep** | Standard for binary mask prediction |
| **CE Loss (text)** | Text generation loss | Partially | **Simplify** → BCE for detection head | Adapt for binary classification |
| **Conversation/Chat Templates** | Multi-turn QA format | No | **Remove** | No conversational interface needed |
| **MMTD-Set construction** | Multi-modal dataset with text | No | **Remove** | Assignment uses standard IFDL datasets (CASIA, etc.) |

### 2.2 Summary of What Stays vs. What Goes

**KEEP (core FakeShield logic):**
- Vision encoder for feature extraction (simplified)
- SAM-based segmentation pipeline (encoder + decoder)
- Feature projection from encoder to SAM prompt space
- Dice + BCE loss for mask supervision
- Detection classification (simplified from DTG)

**REMOVE (resource-prohibitive or assignment-irrelevant):**
- LLM backbone (13B parameters — impossible on T4)
- Text explanation generation
- Tamper Comprehension Module (depends on LLM output)
- GPT-4o dataset pipeline
- Domain tag system (3-class → simplified to binary)
- LoRA adapters for LLM
- Conversation templates

---

## 3. FakeShield-lite Architecture

### 3.1 Design Philosophy

FakeShield-lite preserves the **dual-task architecture** (detection + localization) and the **SAM-based segmentation pipeline** from the original FakeShield, while removing the LLM-dependent components. The key insight is:

> In FakeShield, the LLM's role is to produce a semantic understanding of tampering (text) that guides SAM. In FakeShield-lite, we replace this text-mediated guidance with **direct visual feature projection** — the encoder features are projected into SAM's prompt space without the LLM intermediary.

This is a **minimal, justified modification**: we cut the text path but preserve the visual→prompt→mask pipeline.

### 3.2 Architecture

```
FakeShield-lite Architecture
═══════════════════════════════

Input Image (3 × 256 × 256)
       │
       ├───────────────────────────┐
       │                           │
       ▼                           ▼
┌──────────────┐           ┌──────────────┐
│ SAM ViT-B    │           │ CLIP ViT-B/16│
│ Image Encoder│           │ (Global      │
│ (Frozen)     │           │  Features)   │
│ 91M params   │           │  86M params  │
└──────┬───────┘           └──────┬───────┘
       │                          │
       │  E_mid                   │  F_global
       │  (64×64×256)             │  (197×768)
       │                          │
       │                          ├──────────────────┐
       │                          │                  │
       │                          ▼                  ▼
       │                   ┌────────────┐    ┌──────────────┐
       │                   │ Detection  │    │  Feature     │
       │                   │ Head       │    │  Projection  │
       │                   │ (MLP)      │    │  (MLP)       │
       │                   │ Binary     │    │  768 → 256   │
       │                   │ classifier │    │              │
       │                   └─────┬──────┘    └──────┬───────┘
       │                         │                  │
       │                    Detection               │ h_prompt
       │                    Score                   │ (1×256)
       │                    (real/fake)             │
       │                                            │
       │                          ┌─────────────────┘
       │                          │
       │                          ▼
       │                   ┌──────────────┐
       │                   │SAM Prompt    │
       │                   │Encoder       │
       │                   └──────┬───────┘
       │                          │
       │        ┌─────────────────┘
       ▼        ▼
┌────────────────────────┐
│   SAM Mask Decoder     │
│   (Trainable, 4M)      │
│   Two-Way Transformer  │
└───────────┬────────────┘
            │
            ▼
    Predicted Mask (H × W)
    Binary tampered region
```

### 3.3 Module Descriptions

#### A. Dual Encoder (Preserving FakeShield's Dual Visual Processing)

FakeShield uses two separate image processing paths:
1. CLIP encoder for semantic understanding (global features)
2. SAM encoder for segmentation features

FakeShield-lite **preserves this dual-encoder design**, directly inheriting FakeShield's architectural principle:

- **CLIP ViT-B/16** (86M params, frozen): Replaces CLIP ViT-L/14-336. Extracts global semantic features. Provides the feature representation that was originally fed to the LLM. Now feeds the Detection Head and Feature Projection directly.
- **SAM ViT-B Encoder** (91M params, frozen): Replaces SAM ViT-H. Produces 64×64 spatial feature maps at 256 channels for segmentation.

#### B. Detection Head (Simplified DTG)

FakeShield's DTG classifies tampering domain (PS/DF/AIGC). FakeShield-lite simplifies this to **binary classification** (real/fake):

```python
DetectionHead:
  CLIP [CLS] token (768) → Linear(768, 256) → ReLU → Linear(256, 1) → Sigmoid
```

This preserves the original DTG's concept of image-level classification while adapting to the assignment's binary requirement.

#### C. Feature Projection (Replaces TCM + Text Path)

In FakeShield, the LLM (TCM) aligns text descriptions with visual features, producing `h_<SEG>` for SAM. FakeShield-lite **directly projects CLIP features into SAM's prompt space**:

```python
FeatureProjection:
  CLIP [CLS] token (768) → Linear(768, 768) → ReLU → Linear(768, 256)
```

This is architecturally analogous to FakeShield's `text_hidden_fcs` MLP in GLaMM (Linear → ReLU → Linear → Dropout, mapping hidden_size → out_dim=512), but adapted for our dimensions. The projection replaces the TCM's role of producing a prompt embedding for SAM.

#### D. SAM Mask Decoder (Preserved from FakeShield)

Directly inherited from FakeShield's MFLM with no changes:
- Two-way transformer (depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)
- Takes image embeddings (from SAM encoder) + prompt embeddings (from Feature Projection)
- Outputs binary mask

**Training:** Mask decoder parameters are unfrozen (same as original FakeShield).

### 3.4 Loss Functions (Preserved from FakeShield)

```
ℓ_total = ℓ_det + α · ℓ_bce + β · ℓ_dice

Where:
  ℓ_det  = BCE(detection_pred, detection_label)     # Binary detection
  ℓ_bce  = BinaryCrossEntropy(mask_pred, mask_gt)    # Pixel-level BCE (from FakeShield's ℓ_bce)
  ℓ_dice = DiceLoss(mask_pred, mask_gt)              # Dice loss (from FakeShield's ℓ_dice)

  α = 2.0, β = 0.5 (matching FakeShield's balance)
```

This directly mirrors FakeShield's MFLM loss (Equation 5 in the paper), replacing the text CE loss with a binary detection CE loss.

### 3.5 Size Comparison

| Component | FakeShield | FakeShield-lite | Reduction |
|-----------|-----------|----------------|-----------|
| Vision Encoder | CLIP ViT-L (304M) | CLIP ViT-B (86M) | 3.5× smaller |
| LLM (DTE-FDM) | LLaVA-13B (13B) | Removed (0) | ∞ |
| LLM (TCM in MFLM) | ~7-13B | Removed (0) | ∞ |
| SAM Encoder | ViT-H (632M) | ViT-B (91M) | 7× smaller |
| SAM Decoder | 4M | 4M | Same |
| Detection Head | DTG ~25M | MLP ~0.4M | 63× smaller |
| Feature Projection | Text proj 2M | MLP ~0.8M | Similar |
| **Total** | **~20-27B** | **~182M** | **~110-150× smaller** |
| **VRAM (fp16)** | **~42-54 GB** | **~1.5-2 GB** | **Fits T4 easily** |

---

## 4. Pretrained Weights Strategy

### 4.1 SAM Pretrained Weights

**Status:** Directly reusable.

SAM ViT-B pretrained weights are publicly available from Meta:
- Checkpoint: `sam_vit_b_01ec64.pth`
- Source: [Meta SAM repository](https://github.com/facebookresearch/segment-anything)

**Loading strategy:**
```python
sam = build_sam_vit_b(checkpoint="sam_vit_b_01ec64.pth")
# Freeze encoder, unfreeze mask decoder (matching FakeShield's training)
for param in sam.image_encoder.parameters():
    param.requires_grad = False
sam.mask_decoder.train()
for param in sam.mask_decoder.parameters():
    param.requires_grad = True
```

This directly mirrors FakeShield's `_configure_grounding_encoder` and `_train_mask_decoder` methods in `GLaMM.py`.

### 4.2 CLIP Vision Encoder Weights

**Status:** Directly reusable.

CLIP ViT-B/16 pretrained weights from OpenAI:
- Available via `transformers`: `openai/clip-vit-base-patch16`
- Or via `open_clip`: `ViT-B-16`

**Loading strategy:**
```python
from transformers import CLIPVisionModel
clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
clip_encoder.requires_grad_(False)  # Frozen, matching FakeShield's approach
```

This mirrors FakeShield's `CLIPVisionTower` class which freezes the tower after loading.

### 4.3 FakeShield LoRA Checkpoints

**Status:** NOT reusable.

FakeShield's LoRA weights were trained for:
1. DTE-FDM: LoRA on LLaVA-13B (rank=128, alpha=256) — we don't have the LLM
2. MFLM: LoRA on TCM LLM (rank=8, alpha=16) — we don't have TCM

Since we removed both LLMs, these LoRA weights have no target model to apply to.

### 4.4 FakeShield's SAM Mask Decoder Weights

**Status:** Partially reusable IF available.

FakeShield fine-tunes SAM's mask decoder on tamper detection data. If the checkpoint is available:
```python
# Load FakeShield's fine-tuned mask decoder weights
state_dict = torch.load("fakeshield_mflm_checkpoint.pth")
mask_decoder_weights = {k.replace("grounding_encoder.mask_decoder.", ""): v
                        for k, v in state_dict.items()
                        if "mask_decoder" in k}
sam.mask_decoder.load_state_dict(mask_decoder_weights, strict=False)
```

However, FakeShield uses ViT-H while we use ViT-B. The mask decoder architecture is the same (transformer_dim=256), so weights **may** transfer if the prompt_embed_dim matches.

### 4.5 Weight Loading Summary

| Weight Source | Reusable? | Strategy |
|---|---|---|
| SAM ViT-B pretrained | Yes | Load from Meta, freeze encoder |
| CLIP ViT-B/16 pretrained | Yes | Load from OpenAI, freeze |
| FakeShield LoRA (DTE-FDM) | No | LLM removed |
| FakeShield LoRA (MFLM TCM) | No | TCM removed |
| FakeShield SAM Decoder | Maybe | Same architecture dim, could transfer |
| Detection Head | No | Train from scratch |
| Feature Projection | No | Train from scratch |

---

## 5. Implementation Plan

### 5.1 Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Image Size** | 256×256 (CLIP), 1024→resized to 256 (SAM) | SAM expects 1024 but ViT-B can work with interpolated pos embeddings at smaller sizes; we use 1024 for SAM path and 224 for CLIP path |
| **CLIP input** | 224×224 | Standard CLIP ViT-B/16 input size |
| **SAM input** | 1024×1024 (padded) | SAM's native resolution; images resized longest side to 1024 |
| **Batch Size** | 4 | T4 with dual encoders; tested for memory feasibility |
| **Training Epochs** | 25 | Sufficient for convergence with pretrained backbones |
| **Optimizer** | AdamW | Modern optimizer with weight decay; standard for fine-tuning |
| **Learning Rate** | 1e-4 (Detection Head, Projection), 5e-5 (SAM Decoder) | Lower LR for pretrained components, higher for new heads |
| **LR Schedule** | Cosine annealing with warmup (5% of steps) | Smooth convergence |
| **Weight Decay** | 0.01 | Standard AdamW regularization |
| **Mixed Precision** | FP16 (torch.cuda.amp) | Halves memory usage on T4 |
| **Gradient Accumulation** | 4 steps | Effective batch size of 16 |

### 5.2 Loss Configuration

```python
loss = (1.0 * detection_bce_loss) + (2.0 * mask_bce_loss) + (0.5 * mask_dice_loss)
```

Weights derived from FakeShield's MFLM loss balance (α, β parameters).

### 5.3 Data Augmentation

```
Training:
  - Random horizontal flip (p=0.5)
  - Random rotation (±15°)
  - Color jitter (brightness=0.2, contrast=0.2, saturation=0.1)
  - Random resized crop (scale=(0.8, 1.0))
  - JPEG compression simulation (quality 70-100)  # For robustness bonus
  - Gaussian noise (σ=0-0.02)                      # For robustness bonus

Validation/Test:
  - Resize to fixed size
  - Normalize
```

### 5.4 Dataset Strategy

**Primary:** CASIA v1.0 / v2.0 (well-studied, masks available)
**Secondary:** Coverage (copy-move), Columbia (splicing)

Split: 80% train / 10% validation / 10% test (stratified by tampering type)

---

## 6. Evaluation Metrics

### 6.1 Image-Level Detection Metrics

**Accuracy:**
```
ACC = (TP + TN) / (TP + TN + FP + FN)
```
Overall proportion of correct predictions.

**Precision:**
```
P = TP / (TP + FP)
```
Among predicted-tampered, how many are actually tampered.

**Recall:**
```
R = TP / (TP + FN)
```
Among actually-tampered, how many are correctly detected.

**F1 Score:**
```
F1 = 2 × (P × R) / (P + R)
```
Harmonic mean balancing precision and recall. This is the primary detection metric used in FakeShield's paper.

### 6.2 Pixel-Level Localization Metrics

**Intersection over Union (IoU):**
```
IoU = |P ∩ G| / |P ∪ G|
```
Where P = predicted mask, G = ground truth mask. Measures overlap between predicted and true tampered regions.

**Dice Score (F1 for pixels):**
```
Dice = 2 × |P ∩ G| / (|P| + |G|)
```
Equivalent to pixel-level F1. Directly corresponds to FakeShield's dice loss.

**Both metrics use threshold = 0.5** on the predicted probability mask, consistent with FakeShield's default.

### 6.3 Robustness Metrics (Bonus)

Evaluate IoU and F1 under:
- JPEG compression (quality: 90, 70, 50)
- Resize (×0.5, ×0.75)
- Gaussian noise (σ: 0.01, 0.03, 0.05)

Report metrics as percentage drop from clean performance.

---

## 7. Ablation Study

### 7.1 Ablation Design

| Experiment | What Changes | Expected Impact |
|---|---|---|
| **Full FakeShield-lite** | All components | Baseline performance |
| **Without Detection Head** | Remove detection branch, train localization only | Minor impact on localization; loss of detection capability. Tests whether joint training helps localization (FakeShield found that multitask interference is real) |
| **Without SAM Features** | Replace SAM encoder+decoder with simple U-Net decoder on CLIP features | Significant drop in localization quality. Validates FakeShield's architectural choice of using SAM for segmentation |
| **Without CLIP Features** | Remove CLIP encoder; use only SAM for both tasks | Drop in detection accuracy; localization may suffer from loss of global semantic understanding. Validates dual-encoder design |
| **Without Data Augmentation** | Train without any augmentation | Overfitting, poor generalization. Tests robustness contribution |
| **SAM ViT-B vs no pretrained** | Random init SAM encoder vs pretrained | Large drop without pretrained weights. Validates transfer learning strategy |

### 7.2 Expected Results Analysis

1. **Removing Detection Head:** FakeShield's paper notes joint training interference. In our case, the detection head is lightweight, so removing it might slightly improve localization but removes a key deliverable.

2. **Removing SAM Features:** This is the most impactful ablation. FakeShield's core insight is using SAM's segmentation capability. Without it, localization should degrade significantly (est. -15-25% IoU).

3. **Removing CLIP Features:** CLIP provides semantic understanding that helps the projection module generate meaningful prompts. Without it, the prompt to SAM becomes uninformed (est. -10-15% IoU).

4. **Removing Augmentation:** Standard finding — expect -5-10% on all metrics, larger drops on robustness tests.

---

## 8. Notebook Structure

### Notebook: `vF.0_FakeShield_Assignment.ipynb`

```
Section 1: Introduction & Problem Statement
  - Explain image forgery detection and localization
  - Describe FakeShield paper motivation
  - State assignment objectives

Section 2: Environment Setup & Dependencies
  - Install packages (torch, segment-anything, transformers, etc.)
  - Mount Google Drive / download datasets
  - Set random seeds for reproducibility

Section 3: Dataset Preparation
  - Download and extract CASIA dataset
  - Visualize sample images with masks
  - Implement dataset class with augmentations
  - Create train/val/test splits
  - Show data distribution statistics

Section 4: FakeShield Architecture Overview
  - Explain original FakeShield (with diagram)
  - Explain pruning rationale
  - Present FakeShield-lite architecture

Section 5: FakeShield-lite Architecture (Model Implementation)
  - Implement CLIP encoder wrapper
  - Implement SAM encoder/decoder integration
  - Implement Detection Head
  - Implement Feature Projection
  - Implement full FakeShield-lite model
  - Print model summary and parameter counts

Section 6: Data Augmentation & Preprocessing
  - Define training transforms (with robustness augmentations)
  - Define validation transforms
  - Visualize augmented samples

Section 7: Training Pipeline
  - Define loss functions (BCE + Dice)
  - Configure optimizer and scheduler
  - Training loop with mixed precision
  - Validation loop
  - Logging and checkpointing
  - Training curves visualization

Section 8: Evaluation
  - Load best checkpoint
  - Compute detection metrics (ACC, P, R, F1)
  - Compute localization metrics (IoU, Dice)
  - Print comprehensive results table

Section 9: Visualization Results
  - Grid of: Original | Ground Truth | Predicted | Overlay
  - Show best and worst predictions
  - Detection confidence distribution

Section 10: Robustness Testing (Bonus)
  - JPEG compression at different qualities
  - Resize robustness
  - Noise robustness
  - Results table

Section 11: Ablation Study
  - Run key ablations
  - Results comparison table
  - Analysis

Section 12: Conclusion & Limitations
  - Summary of results
  - Limitations
  - Future work
```

---

## 9. 48-Hour Execution Plan

```
═══════════════════════════════════════════════════════════
 Hour 0-4: Environment & Dataset Setup
═══════════════════════════════════════════════════════════
 - Set up Colab environment with all dependencies
 - Download CASIA dataset and verify integrity
 - Implement dataset class and data loaders
 - Verify images and masks load correctly
 - Create train/val/test splits

═══════════════════════════════════════════════════════════
 Hour 4-10: Model Architecture Implementation
═══════════════════════════════════════════════════════════
 - Implement CLIP encoder wrapper
 - Implement SAM ViT-B integration
 - Implement Detection Head (MLP)
 - Implement Feature Projection (MLP)
 - Implement FakeShield-lite combined model
 - Download pretrained weights (SAM-B, CLIP-B/16)
 - Verify forward pass works with sample batch
 - Check VRAM usage fits T4

═══════════════════════════════════════════════════════════
 Hour 10-12: Training Pipeline Setup
═══════════════════════════════════════════════════════════
 - Implement Dice Loss and combined loss function
 - Configure optimizer (AdamW with differential LR)
 - Configure LR scheduler (cosine with warmup)
 - Implement training loop with mixed precision
 - Implement validation loop
 - Set up checkpointing and logging
 - Run 1-2 epochs to verify training works

═══════════════════════════════════════════════════════════
 Hour 12-24: Training
═══════════════════════════════════════════════════════════
 - Launch full 25-epoch training run
 - Monitor loss curves and validation metrics
 - Save best model checkpoint
 - If training issues: debug and restart

═══════════════════════════════════════════════════════════
 Hour 24-32: Evaluation & Visualization
═══════════════════════════════════════════════════════════
 - Load best checkpoint
 - Compute all evaluation metrics
 - Generate visualization grids
 - Implement overlay visualization
 - Test on validation and test sets
 - Generate detection metrics table

═══════════════════════════════════════════════════════════
 Hour 32-38: Robustness & Ablation
═══════════════════════════════════════════════════════════
 - Implement JPEG compression testing
 - Implement resize/noise robustness testing
 - Run ablation experiments (at least 2-3)
 - Compile results tables
 - Analyze and interpret results

═══════════════════════════════════════════════════════════
 Hour 38-44: Documentation
═══════════════════════════════════════════════════════════
 - Write introduction and problem statement
 - Document architecture with diagrams
 - Write training strategy section
 - Write evaluation analysis
 - Write limitations and future work
 - Clean up all code cells

═══════════════════════════════════════════════════════════
 Hour 44-48: Polish & Submit
═══════════════════════════════════════════════════════════
 - Final notebook review and cleanup
 - Ensure all cells run sequentially
 - Verify outputs are rendered
 - Upload trained weights to Drive
 - Test notebook from fresh runtime
 - Submit
═══════════════════════════════════════════════════════════
```

---

## 10. Assignment Documentation

### Problem Explanation

Image forgery detection and localization (IFDL) is a critical task in digital forensics. With the advent of powerful image editing tools (Photoshop, DALL-E, FaceApp), it has become increasingly easy to create realistic tampered images. The IFDL task requires:

1. **Detection:** Determining whether a given image is authentic or tampered (binary classification).
2. **Localization:** Identifying the specific pixel regions that have been manipulated (semantic segmentation).

This project implements **FakeShield-lite**, a pruned version of the FakeShield framework (Xu et al., ICLR 2025). FakeShield is a multi-modal framework that uses LLMs, CLIP, and SAM for explainable forgery detection. Due to compute constraints (Google Colab T4, 16GB VRAM), we removed the LLM components (~13B parameters) while preserving the core visual detection and SAM-based localization pipeline.

### Architecture Summary

FakeShield-lite retains FakeShield's core insights:
- **Dual-encoder design:** CLIP for semantic understanding + SAM for segmentation (from FakeShield's Section 3.2-3.4)
- **SAM-based mask generation:** Using learned prompts to guide SAM's mask decoder (from FakeShield's MFLM)
- **Combined detection + localization losses:** BCE + Dice loss for masks (from FakeShield's Equation 5)

Removed components (with justification):
- LLM backbone (13B parameters; exceeds T4 VRAM)
- Text explanation generation (not required by assignment)
- Domain Tag Generator's 3-class system (simplified to binary detection)

### Training Strategy

- **Transfer learning:** Pretrained CLIP and SAM encoders (frozen), only fine-tuning detection head, projection module, and SAM mask decoder
- **Mixed precision (FP16):** Reduces memory by ~50%
- **Gradient accumulation (4 steps):** Effective batch size of 16 with batch_size=4
- **Cosine LR schedule with warmup:** Prevents learning rate spikes early in training

### Limitations

1. **No text explanations:** FakeShield's key selling point is explainability; our lite version cannot provide reasoning.
2. **Reduced segmentation quality:** SAM ViT-B is less capable than ViT-H for fine-grained segmentation.
3. **No domain-specific prompting:** Without the DTG's 3-class guidance, the model cannot adapt to different tampering types.
4. **Single dataset training:** FakeShield was trained on diverse data (PS, DF, AIGC); our model trains on CASIA only.
5. **No TCM alignment:** The direct feature projection is less sophisticated than FakeShield's TCM for aligning visual and semantic features.
