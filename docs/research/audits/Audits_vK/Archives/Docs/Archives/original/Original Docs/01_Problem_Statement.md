# 1. Problem Statement: Tampered Image Detection & Localization

## 1.1 What is Image Tampering?

Image tampering (also called image forgery or image manipulation) is the deliberate alteration of a digital photograph to change its visual content, meaning, or context. With the proliferation of powerful editing tools (Adobe Photoshop, GIMP) and generative AI models (Stable Diffusion, DALL-E), creating convincing forgeries has become trivially easy — while detecting them remains extraordinarily difficult.

The assignment objective is:

> *"Develop a deep learning model to detect and localize tampered (edited or manipulated) regions in images. The model should not only classify whether an image is tampered, but also generate a pixel-level mask highlighting altered regions."*

This is fundamentally a **dual-task problem**:
1. **Image-Level Detection** — Binary classification: Is this image authentic or tampered?
2. **Pixel-Level Localization** — Semantic segmentation: Which exact pixels were manipulated?

---

## 1.2 Types of Image Tampering

There are three primary categories of image manipulation that forensic systems must address:

### 1.2.1 Splicing
A region from **one image** is cut and pasted into **another image**. This is the most common type of forgery. The spliced region often comes from a different camera, lighting condition, or scene — creating subtle statistical inconsistencies at the boundary.

**Example**: Taking a person from photo A and placing them into the background of photo B.

**Forensic Clues**:
- Different noise patterns between the spliced region and original background
- Mismatched JPEG compression histories (double compression artifacts)
- Inconsistent Color Filter Array (CFA) demosaicing patterns
- Lighting and shadow direction mismatches

### 1.2.2 Copy-Move
A region within the **same image** is duplicated and moved to another location. This is typically used to hide objects (by covering them with copied background) or duplicate objects.

**Example**: Copying a section of grass to cover an unwanted object in a landscape photo.

**Forensic Clues**:
- Two regions with statistically identical noise characteristics
- Matching local feature descriptors (keypoints) across different spatial locations
- Identical pixel correlation patterns in non-adjacent regions

### 1.2.3 Removal / Inpainting
An object is **deleted** from the image and the resulting hole is filled using generative techniques or surrounding pixel interpolation.

**Example**: Removing a person from a group photo using content-aware fill.

**Forensic Clues**:
- Generative noise patterns that differ from camera sensor noise
- Over-smoothed regions lacking natural texture variation
- Broken local pixel correlations where the inpainting algorithm "invented" content

---

## 1.3 The Core Forensic Assumption

Every digital image carries a unique **forensic fingerprint** created by its acquisition pipeline:

```
Light → Camera Lens → Sensor (CFA/Bayer Filter) → ISP (Demosaicing) → Post-Processing → JPEG Compression → Stored Image
```

Each stage introduces specific, predictable statistical patterns:
- The **Bayer filter** creates periodic correlations between neighboring pixels (because each sensor pixel captures only one color channel — R, G, or B — and the others are interpolated)
- The **Image Signal Processor (ISP)** applies demosaicing algorithms that create camera-specific interpolation patterns
- **JPEG compression** quantizes DCT coefficients, leaving quantization table fingerprints

**When an image is tampered, these pipeline-specific patterns are disrupted.** A spliced region from a different camera will have different CFA patterns and noise characteristics. A copy-moved region will have identical noise in two places where it shouldn't. An inpainted region will have generative patterns that don't match sensor noise at all.

Modern forensic deep learning models exploit these **semantic-agnostic artifacts** — features that are independent of what the image depicts but highly sensitive to how it was captured and whether it was altered.

---

## 1.4 Passive vs. Active Forensics

| Aspect | Active Forensics | Passive Forensics |
|--------|-----------------|-------------------|
| **Method** | Embed watermarks or digital signatures at capture time | Analyze intrinsic image properties without prior information |
| **Prerequisite** | Requires cooperation of the image creator | Works on any image "in the wild" |
| **Applicability** | Limited to controlled environments | Universal — works even when the source is unknown |
| **Example** | Camera embeds invisible watermark; verification checks if watermark is intact | Neural network detects noise pattern inconsistencies in a social media image |

**This project uses passive forensics** — the most challenging and practically relevant approach, since we rarely have access to the original unaltered image or embedded authentication data.

---

## 1.5 Why Does This Matter?

Tampered image detection is not an academic exercise. It addresses critical real-world problems:

| Domain | Impact of Undetected Tampering |
|--------|-------------------------------|
| **Journalism & Media** | Manipulated photos can spread misinformation and propaganda at scale |
| **Legal & Law Enforcement** | Forged evidence can wrongly convict or acquit; digital forensics is now standard in courts |
| **Social Media & Platforms** | Fabricated images erode public trust and enable fraud, harassment, and identity manipulation |
| **Insurance & Finance** | Tampered documents and images are used for fraudulent claims |
| **National Security** | Satellite imagery or surveillance footage manipulation can have geopolitical consequences |
| **Healthcare** | Altered medical images can lead to misdiagnosis or research fraud |

The rise of **generative AI** has dramatically escalated this problem. Diffusion-based inpainting can create forgeries that are visually perfect, making robust automated detection systems an urgent necessity.

---

## 1.6 Assignment Decomposition into Engineering Tasks

The BigVision assignment has 4 formal sections plus bonus criteria. Below is the decomposition into concrete engineering tasks:

### Task 1: Dataset Selection & Preparation
| Requirement | Engineering Task |
|-------------|-----------------|
| Use publicly available datasets with authentic images, tampered images, and ground truth masks | Select and download CASIA v2.0 (and optionally COVERAGE) via Kaggle API |
| All dataset cleaning, preprocessing, mask alignment | Write validation scripts to fix 17 resolution-misaligned images, binarize masks (threshold > 128), map naming conventions |
| Train/validation/test split | Implement stratified split (85/7.5/7.5) maintaining authentic-to-tampered ratio |
| Data augmentation | Build `albumentations` pipeline with synchronized image-mask transforms |

### Task 2: Model Architecture & Learning
| Requirement | Engineering Task |
|-------------|-----------------|
| Train a model to predict tampered regions | Design and implement dual-stream architecture (RGB + forensic noise features) with pixel-level segmentation output |
| Choice of architecture and loss functions | Select encoder (EfficientNet-B1 or SegFormer-B1), implement hybrid loss (BCE + Dice + Edge) |
| Runnable on Google Colab T4 GPU | Optimize with AMP mixed precision, gradient accumulation, appropriate batch size and resolution |

### Task 3: Testing & Evaluation
| Requirement | Engineering Task |
|-------------|-----------------|
| Localization performance metrics | Compute Pixel-F1, Pixel-IoU, AUC-ROC, MCC |
| Image-level detection accuracy | Derive image-level predictions from mask predictions, compute accuracy/AUC |
| Visual results: Original, Ground Truth, Predicted, Overlay | Build visualization function with 4-column grid display |

### Task 4: Deliverables & Documentation
| Requirement | Engineering Task |
|-------------|-----------------|
| Single Google Colab Notebook | Structure notebook with markdown narrative + code cells covering all sections |
| Dataset explanation, architecture description, training strategy, hyperparameters, results, visualizations | Write inline documentation in notebook markdown cells |
| Colab link, model weights, additional scripts | Export model weights to Google Drive, share notebook with appropriate permissions |

### Bonus Tasks
| Bonus Criteria | Engineering Task |
|----------------|-----------------|
| Robustness against JPEG compression, resizing, cropping, noise | Apply test-time distortions and report degradation in a robustness table |
| Detect subtle copy-move and similar-texture splicing | Evaluate on COVERAGE dataset; ensure SRM/noise preprocessing captures copy-move artifacts |

---

## 1.7 Success Criteria

From the perspective of a Principal AI Engineer reviewing this submission, the evaluators are looking for:

1. **Strong problem-solving skills** — Demonstrated through thoughtful data cleaning, handling of edge cases (misaligned masks, class imbalance), and iterative debugging
2. **Thoughtful architecture choices** — Not just "I used a U-Net" but "I chose this architecture because of these forensic-specific requirements, and I validated the choice with these experiments"
3. **Rigorous evaluation methodologies** — Multiple complementary metrics, proper train/val/test separation, robustness testing, statistical confidence in results

The model does not need to achieve SOTA performance. What matters is the **engineering process**: clear reasoning, clean implementation, honest reporting of results (including failures), and professional-grade documentation.
