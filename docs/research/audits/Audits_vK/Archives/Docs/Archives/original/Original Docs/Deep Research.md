# Technical Analysis and Strategic

# Implementation Framework for

# Advanced Image Tampering Detection

# and Localization

The proliferation of sophisticated digital image editing tools and the advent of generative
artificial intelligence have rendered the task of distinguishing between authentic and
manipulated visual media increasingly complex. For digital forensics, the challenge is no longer
merely identifying visual anomalies but detecting subtle statistical inconsistencies that occur at
the sub-pixel level. This report provides an exhaustive technical analysis of the state-of-the-art
in image manipulation detection and localization (IMDL), focusing on the integration of artifact
extraction, robust mathematical optimization, and performance-oriented deployment within
resource-constrained environments such as Google Colab.

## Theoretical Foundations of Digital Image Forensics

Image forensics is broadly categorized into active and passive methodologies. Active methods,
such as digital watermarking and signatures, rely on the presence of embedded information to
verify authenticity.^1 However, these are often impractical for "in-the-wild" scenarios where the
original image is unavailable. Passive methods, therefore, represent the primary research
frontier, focusing on the intrinsic properties of the digital image itself.^2 Passive forgery
detection aims to identify specific classes of manipulation: splicing, where a region from one
image is pasted into another; copy-move, where a region is duplicated within the same image;
and removal or inpainting, where an object is deleted and the area is filled using generative
techniques.^4
The core assumption in digital forensics is that every stage of the image acquisition
pipeline—from the light hitting the sensor to the final JPEG compression—leaves a specific
digital footprint. Manipulations inevitably disrupt these footprints, creating inconsistencies in
noise patterns, compression artifacts, and pixel correlations.^7 Modern state-of-the-art
architectures seek to exploit these "semantic-agnostic" features, which are independent of the
image content but highly sensitive to the tampering process.^10

## Artifact Extraction and Forensic Preprocessing

## Methodologies

A critical prerequisite for effective tampering localization is the suppression of semantic
content to reveal forensic artifacts. Standard convolutional neural networks (CNNs) trained for


object recognition tend to focus on high-level shapes and textures, which can be misleading in
forensics. Specialized preprocessing layers are therefore employed to highlight low-level signal
inconsistencies.

### Spatial Rich Model (SRM) and Noise Residual Analysis

The Spatial Rich Model (SRM) represents a collection of handcrafted high-pass filters originally
developed for steganalysis. In image forensics, SRM kernels are utilized to extract noise
residuals by calculating the difference between a pixel and its neighbors.^12 These residuals
capture the local statistical distributions that are often altered during splicing or inpainting. For
instance, a common SRM kernel used in the first layer of forensic networks is designed to
capture the second-order horizontal or vertical residuals.^12
The mathematical utility of SRM lies in its ability to act as a regularizer, suppressing the
high-amplitude semantic information while amplifying the low-amplitude noise.^12 Research
indicates that initializing the initial layers of a CNN with SRM weights significantly boosts
localization accuracy compared to random initialization, as it provides the model with a strong
prior for noise-aware feature extraction.^13

### Constrained Convolution and BayarConv

While SRM filters are fixed, Bayar Convolution (BayarConv) introduces a learnable mechanism
for content suppression. BayarConv imposes a strict constraint on the convolutional kernels
during training. Specifically, for a kernel of size , the weights are constrained such
that the central weight is set to , and the sum of all other weights equals 5 :
This constraint ensures that the filter behaves as a high-pass operator, effectively learning the
"prediction error" for each pixel based on its neighborhood. By allowing the network to learn
the optimal high-pass coefficients, BayarConv can adapt to the specific artifact distributions of
the training data, often outperforming fixed SRM filters in detecting sophisticated
manipulations like deepfake-based inpainting.^16

### Frequency Domain and Compression Artifacts

Beyond spatial residuals, the frequency domain offers critical forensic clues. Most digital
images are stored in the JPEG format, which involves a Discrete Cosine Transform (DCT) and
quantization. When an image is tampered with and re-saved, the manipulated region often
exhibits a different compression history than the authentic background, leading to "double
compression" artifacts.^3 Architectures like CAT-Net and ObjectFormer leverage this by


processing the DCT coefficients directly, allowing the model to detect inconsistencies in the
quantization tables that are invisible in the RGB domain.^11
**Artifact Type Extraction
Mechanism
Domain Forensic Targeted
Inconsistency**
Noise Residual SRM Filters Spatial Local variance and
noise floor
mismatches 16
High-Pass Trace BayarConv Spatial Adaptive prediction
errors at
boundaries 11
Compression DCT Spectrum Frequency Mismatched JPEG
quantization/double
compression 18
Pixel Dependency Masked
Self-Attention
Spatial Disruptions in
demosaicing/CFA
patterns 7
Local Anomaly Noiseprint++ Spatial Deviations from
camera-specific ISP
fingerprints 20

## Advanced Architectures for Detection and

## Localization

The landscape of 2024 and 2025 shows a decisive shift toward multi-modal and multi-scale
architectures that bridge the gap between pixel-level artifacts and global semantic
consistency.

### Dual-Branch and Multi-View Networks

A common paradigm in SOTA models is the dual-branch architecture, where one stream
processes the RGB image for semantic clues (e.g., edge inconsistencies or lighting
mismatches) and a second stream processes forensic artifacts (e.g., SRM noise or DCT
coefficients). The MVSS-Net and its enhanced version, MVSS-Net++, are prototypical of this
approach. MVSS-Net++ utilizes an Edge-Supervised Branch (ESB) to trace boundary artifacts
and a Noise-Sensitive Branch (NSB) to identify noise distribution discrepancies.^10 This


multi-view learning ensures the model is both sensitive to local tampering and specific enough
to prevent false alarms on authentic images.^10
The REFORGE network further evolves this by combining a classification branch for image-level
detection with a segmentation branch for pixel-level localization.^22 REFORGE is particularly
innovative due to its integration of a reinforcement learning (RL) agent. After the segmentation
branch produces an initial mask, the RL agent iteratively refines the mask by treating each pixel
as an agent that seeks to maximize a reward based on localization accuracy.^22

### Transformer-Based Forensic Models

The adoption of Transformers, particularly the Vision Transformer (ViT) and SegFormer, has
addressed the limitation of CNNs in capturing long-range dependencies. In copy-move
forgery, the source and target regions may be far apart; Transformers, through self-attention,
can model these distant relationships more effectively than local convolutional kernels.^18
SegFormer is highly regarded in the forensics community for its hierarchical structure, which
outputs multi-scale features, and its lightweight "all-MLP" decoder.^2 For an internship project on
Google Colab, SegFormer-B0 or B1 offers an ideal balance between performance and
computational efficiency. Unlike standard ViTs that maintain a constant resolution, SegFormer’s
hierarchical approach allows for precise localization of small tampered regions while
maintaining global context.^2 TruFor, another SOTA framework, uses a SegFormer-based
encoder to fuse RGB features with Noiseprint++ (a learned camera fingerprint), producing not
only a localization map but also a reliability map that quantifies the model's confidence in its
predictions.^8

### Explainable and Multi-Modal Models

A rising trend is the development of "explainable" IMDL. FakeShield, for example, is a
multimodal framework that combines visual and language models to provide a textual basis for
its forensic judgment.^28 It evaluates image authenticity and generates masks while providing a
descriptive basis—such as identifying specific textural or lighting inconsistencies—that can be
used as evidence in forensic investigations.^28

## Optimization of Loss Functions for Class Imbalance

One of the most significant technical hurdles in image tampering localization is the extreme
class imbalance within the dataset. Typically, only a small percentage of pixels (often less than
5%) in a tampered image are actually modified.^29 Standard Binary Cross-Entropy (BCE) loss
treats every pixel with equal weight, which leads to models that are biased toward the majority
"authentic" class, often resulting in "empty" masks with high overall accuracy but near-zero
localization precision.^2


### Dice Loss and the F1 Optimization

Dice Loss, based on the Sorensen-Dice coefficient, is a region-based loss function that directly
optimizes for the overlap between the predicted mask ( ) and the ground truth ( ).^2 It is
inherently robust to class imbalance because the background pixels (the majority class) do not
contribute to the numerator of the coefficient 31 :
By focusing only on the intersection of the tampered pixels, Dice Loss forces the model to
prioritize the minority class.^32

### Focal Loss and Hard Example Mining

Focal Loss addresses imbalance by reshuffling the importance of examples based on how
"hard" they are for the model to classify. It adds a focusing parameter to the standard BCE
loss, which down-weights the loss contributed by "easy" (confidently predicted) background
pixels.^31
In forensic tasks, pixels at the boundaries of forgeries or in regions with similar textures are
"hard" examples. Focal Loss ensures that the gradient updates are dominated by these critical
regions rather than the vast authentic background.^31

### Hybrid and Multi-Scale Loss Formulations

The most robust SOTA models utilize a weighted combination of losses. A common
configuration includes BCE for global distribution matching, Dice for region overlap, and a
specialized Edge Loss to supervise the boundaries.^10
Edge loss is often implemented by calculating the binary cross-entropy between the predicted
edges and a ground truth edge map (derived by dilating and eroding the tampering mask).^10
This ensures that the model learns to produce sharp, distinct boundaries rather than blurry,
ambiguous localization maps.

## Performance Optimization for Google Colab T4 GPUs


The NVIDIA T4 GPU provided by Google Colab is an efficient accelerator with 16 GB of VRAM,
optimized primarily for inference but capable of training mid-sized forensic models.^37 To
maximize its utility for an internship assignment, several memory and compute optimizations
must be applied.

### Mixed Precision Training (AMP)

Mixed precision training involves performing the majority of forward and backward pass
computations in 16-bit floating point (FP16) while maintaining weight updates in 32-bit (FP32).^39
This technique leverages the T4's Tensor Cores, leading to a 1.5x to 2x speedup in training and a
reduction in memory usage by approximately 50%.^39 PyTorch's autocast and GradScaler
modules are the industry-standard tools for implementing this, ensuring numerical stability by
scaling gradients to prevent underflow.^39

### Gradient Checkpointing and Accumulation

For deeper architectures like ObjectFormer or ProFact, the activation maps can exceed the 16
GB VRAM limit. Gradient checkpointing is a technique that reduces memory consumption by
trading compute for memory; instead of storing all intermediate activations for the backward
pass, it recomputes them on-the-fly.^43 This allows for the use of larger batch sizes or higher
input resolutions (e.g., 512x512) which are critical for capturing high-frequency forensic
artifacts.^44
If the desired batch size still does not fit in memory, gradient accumulation can be used. This
involves accumulating gradients over multiple small "micro-batches" before performing a single
optimizer update, effectively simulating a larger batch size without increasing the peak VRAM
load.^46

### Memory-Efficient Optimizers and Quantization

Modern optimizers like AdamW can be memory-intensive due to the storage of two moments
per parameter. Using 8-bit optimizers (e.g., via the bitsandbytes library) can reduce the
optimizer's memory footprint by up to 75%.^46 Furthermore, if using a large pre-trained
backbone, Low-Rank Adaptation (LoRA) can be employed to fine-tune only a small fraction of
the parameters, significantly reducing the memory required for gradients and optimizer
states.^46
**Optimization Technique Mechanism Impact on T
Performance**
Mixed Precision (AMP) FP16/FP32 hybrid math ~2x faster training; 50%


```
VRAM reduction 39
Gradient Checkpointing Recomputes activations Enables 2x-3x larger batch
sizes 44
8-bit AdamW Quantizes optimizer states 75% reduction in optimizer
memory 46
LoRA / QLoRA Parameter-efficient tuning Fine-tunes large models on
<8GB VRAM 46
FlashAttention Reordered attention math Faster transformer training;
less memory 49
```
## Benchmarking and Evaluation Methodologies

Evaluating an image forgery detector requires more than simple accuracy metrics. The model
must be assessed for its localization precision and its robustness against common
"anti-forensic" attacks designed to hide tampering traces.

### Evaluation Metrics

1. **Pixel-Level F1-Score and IoU:** These are the primary metrics for localization, measuring
    the harmony between precision and recall at the pixel level.^51
2. **Image-Level Detection Accuracy:** Measures the model's ability to correctly classify an
    image as "authentic" or "tampered".^4
3. **Area Under the Curve (AUC-ROC):** Provides a threshold-independent measure of the
    model's ability to distinguish between forged and authentic pixels.^29
4. **Matthews Correlation Coefficient (MCC):** Useful for highly imbalanced datasets, as it
    accounts for all four quadrants of the confusion matrix (TP, TN, FP, FN).

### The Calibration Gap and Oracle-F

Recent benchmarks have identified a "Calibration Gap" in forensic models. While many models
achieve high AUC-ROC scores, their F1-scores at a fixed threshold (e.g., ) are often
near-zero.^29 This is because the probability scores are not well-calibrated for the extreme rarity
of tampered pixels. To address this, "Oracle-F1"—the best achievable F1-score across all
possible thresholds—is used as a secondary benchmark to evaluate the model's discriminative
potential independently of threshold selection.^29
**Dataset Metric SOTA Model (e.g., Performance**


```
MVSS / TruFor) Value
```
CASIA v2.0 Pixel-F1 VAAS (2025) (^) 94.1% 52
CASIA v2.0 Pixel-IoU VASLNet (2024) (^) 85.1% 52
COVERAGE Pixel-F1 MMFD-Net (2024) (^) 50.4% 51
COVERAGE Pixel-F1 MVSS-Net++ (2022) (^) 48.2% 51
CASIA v1.0 Pixel-F1 IF-OSN (2022) (^) 68.6% 54
NIST16 Pixel-F1 DAE-Net (2024) (^) 87.1% 53
IMD2020 Pixel-F1 DAE-Net (2024) (^) 33.8% 53

### Robustness Evaluation

A "bonus" requirement for high-tier forensic models is robustness against distortions.
Evaluation should be performed on a test set that has been subjected to:
● **JPEG Compression:** Testing at QF levels 50, 70, and 90.^10
● **Resizing and Scaling:** Bilinear and bicubic resizing between 0.5x and 2.0x.^10
● **Noise Addition:** Gaussian white noise with varying variance.^55
● **Screenshot/Re-capturing:** Simulating the artifacts introduced when an image is
displayed and then re-photographed or screen-captured.^10

## Strategic Research and Implementation Roadmap

For the internship assignment, a phased approach is recommended to ensure both a working
baseline and a competitive, SOTA-aligned final model.

### Phase 1: Data Infrastructure and Preprocessing

The focus should be on the CASIA v2.0 and COVERAGE datasets, which provide a robust mix of
splicing and copy-move forgeries.^4
● **Pipeline Development:** Use torchvision and albumentations to build a data loader that
performs online augmentation, including rotation, flipping, and random JPEG
compression.^10
● **Artifact Channel:** Implement an SRM filter layer as the first operation. A multi-channel
output representing different noise residuals should be concatenated with the RGB input.^12


```
● Mask Preparation: Ensure that ground truth masks are binary and aligned. For
copy-move, specifically distinguish between source and target regions if possible (e.g.,
using BusterNet principles).^6
```
### Phase 2: Architecture Selection and Baseline Training

Given the Colab T4 constraints, a hierarchical transformer or a dual-stream CNN is the optimal
choice.
● **Backbone:** Select a SegFormer-B1 encoder pre-trained on ADE20K or ImageNet. This
provides a strong starting point for semantic understanding.^2
● **Fusion Strategy:** Implement a dual-stream encoder where the first stream is the standard
SegFormer (RGB) and the second stream is a lightweight CNN (Noise Residuals). Features
should be fused using a Spatial Attention module to highlight local inconsistencies.^10
● **Initial Training:** Train with a combination of BCE and Dice Loss. Use a small batch size (e.g.,
4 or 8) with gradient accumulation to fit within the 16 GB VRAM.^2

### Phase 3: Forensic Refinement and Multi-Task Learning

Enhance the model by adding specialized forensic supervisory signals.
● **Edge Supervision:** Implement an auxiliary branch that predicts the boundaries of
tampered regions. Use a multi-scale edge loss to ensure the model focuses on the
delicate artifacts found at the periphery of spliced objects.^10
● **Optimization:** Enable Mixed Precision (AMP) to accelerate training. Profile the memory
usage and, if possible, increase the input resolution to 512x512 to preserve high-frequency
artifacts.^39
● **Refinement:** Implement a coarse-to-fine strategy where the model first predicts a
low-resolution mask and then refines it using a dedicated refinement block (e.g., as in
PSCC-Net or ProFact).^18

### Phase 4: Evaluation and Documentation

The final deliverable must align with the rigorous testing requirements of the assignment.^4
● **Benchmarking:** Evaluate the final model on the CASIA v2.0 test set and report Pixel-F1,
Pixel-IoU, and Image-Accuracy.^4
● **Visual Documentation:** Generate a set of comprehensive visualizations for the report:

1. Original Image.
2. Ground Truth Mask.
3. Predicted Localization Map (Probability Heatmap).
4. Overlay Visualization (Mask on Image).
5. Confidence/Reliability Map (if using a TruFor-style decoder).^4
● **Robustness Report:** Create a table showing how performance degrades under JPEG
compression (QF=50) and Resizing (0.5x), fulfilling the bonus point criteria.^4


## Mechanism of Pixel Correlation Disruption in Forensics

A deeper insight into why these models work lies in the demosaicing process. Digital cameras
use a Color Filter Array (CFA), most commonly a Bayer filter, to capture color information.
Because each pixel only records one of three colors (Red, Green, or Blue), the camera's Image
Signal Processor (ISP) must interpolate the missing colors using the surrounding pixels. This
creates a specific, periodic correlation between neighboring pixels.^7
When an image is tampered with, these periodic correlations are disrupted. A spliced object
from a different camera will have a different CFA pattern or demosaicing algorithm, creating a
statistical mismatch at the boundary.^7 Advanced forensic models, such as those employing
masked self-attention or constrained convolutions, are designed specifically to detect these
low-level geometric inconsistencies. This is why the SRM and BayarConv preprocessing steps
are so effective; they strip away the high-level semantic content (which can be easily faked by
generative AI) and focus on the fundamental mathematical structure of the digital signal.^7

## Future Outlook and Generative AI Challenges

The rapid advancement of generative AI (AIGC) presents a dual-edged sword for image
forensics. On one hand, diffusion-based inpainting can create forgeries that are visually perfect
and preserve the global semantic logic of a scene.^61 On the other hand, these generative
processes often leave distinct "generative noise" patterns that differ significantly from
camera-sensor noise.^61
SOTA research in 2025 is increasingly focusing on "Zero-Shot" localization and adaptation to
novel generative models. Techniques such as using the internal generative process of stable
diffusion models to reveal forgery artifacts (as seen in the CLUE framework) represent the next
step in this arms race.^61 For the current assignment, the focus remains on capturing the
transition between the traditional artifacts (splicing/copy-move) and the emerging generative
traces, ensuring a robust and versatile forensic toolset.
By adhering to this strategic framework, an intern can develop a model that not only meets the
assignment criteria but aligns with the cutting-edge methodologies of the professional
forensics community. The combination of lightweight SegFormer backbones,
artifact-suppressing preprocessing, and robust optimization for class-imbalanced data
provides a technically sound and high-performing solution for the task of image tampering
detection and localization.

#### Works cited

#### 1. A comprehensive review of deep learning techniques for image tampering

#### detection, accessed March 10, 2026,

#### https://www.researchgate.net/publication/398245250_A_comprehensive_review_


#### of_deep_learning_techniques_for_image_tampering_detection

#### 2. Hierarchically Structured Transformer Encoders: Image Forgery ..., accessed

#### March 10, 2026,

#### https://medium.com/@jsilvawasd/hierarchically-structured-transformer-encoders

#### -document-forgery-semantic-segmentation-5793c2bf3ec

#### 3. Tampering Detection and Segmentation Model for Multimedia Forensic - The

#### Science and Information (SAI) Organization, accessed March 10, 2026,

#### https://thesai.org/Downloads/Volume14No9/Paper_92-Tampering_Detection_and

#### _Segmentation_Model.pdf

#### 4. Internship Assignment_ Tampered Image Detection & Localization.pdf

#### 5. Can We Get Rid of Handcrafted Feature Extractors? SparseViT:

#### Nonsemantics-Centered, Parameter-Efficient Image Manipulation Localization

#### Through Spare-Coding Transformer - arXiv, accessed March 10, 2026,

#### https://arxiv.org/html/2412.14598v

#### 6. Enhancing Copy-Move Forgery Detection via Attention-Based Similarity Modeling

- World Scientific Publishing, accessed March 10, 2026,

#### https://www.worldscientific.com/doi/pdf/10.1142/S0218001425500442?download

#### =true

#### 7. Pixel-Inconsistency Modeling for Image Manipulation Localization - arXiv.org,

#### accessed March 10, 2026, https://arxiv.org/html/2310.00234v

#### 8. TruFor: Leveraging All-Round Clues for ... - CVF Open Access, accessed March

#### 10, 2026,

#### https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Levera

#### ging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_

#### 23_paper.pdf

#### 9. Forged Copy-Move Recognition Using Convolutional Neural Network - Al-Nahrain

#### Journal of Science (ANJS), accessed March 10, 2026,

#### https://anjs.edu.iq/index.php/anjs/article/download/2335/1837/

#### 10. MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image ..., accessed

#### March 10, 2026, https://pubmed.ncbi.nlm.nih.gov/35671312/

#### 11. Can We Get Rid of Handcrafted Feature Extractors? SparseViT:

#### Nonsemantics-Centered, Parameter-Efficient Image Manipulation Localization

#### through Spare-Coding Transformer - arXiv.org, accessed March 10, 2026,

#### https://arxiv.org/html/2412.14598v

#### 12. Multitask Image Splicing Tampering Detection Based on Attention Mechanism,

#### accessed March 10, 2026,

#### https://www.researchgate.net/publication/364464845_Multitask_Image_Splicing_

#### Tampering_Detection_Based_on_Attention_Mechanism

#### 13. Data-Dependent Scaling of CNN's First Layer for Improved Image Manipulation

#### Detection - Gipsa-lab, accessed March 10, 2026,

#### https://www.gipsa-lab.grenoble-inp.fr/~kai.wang/papers/IWDW20.pdf

#### 14. Learning Rich Features for Image Manipulation Detection - CVF Open Access,

#### accessed March 10, 2026,

#### https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Learning_Rich_F

#### eatures_CVPR_2018_paper.pdf


#### 15. Image Forgery Detection. Introduction | by Kishan | Medium, accessed March 10,

#### 2026,

#### https://kishanraj-16649.medium.com/image-forgery-detection-ae7a8101ddf

#### 16. MMFusion: Combining Image Forensic Filters for Visual Manipulation Detection

#### and Localization - arXiv.org, accessed March 10, 2026,

#### https://arxiv.org/html/2312.01790v

#### 17. Exploring Multi-Modal Fusion for Image Manipulation Detection and Localization -

#### arXiv, accessed March 10, 2026, https://arxiv.org/html/2312.01790v

#### 18. DocForge-Bench: A Comprehensive Benchmark for Document Forgery Detection

#### and Analysis - arXiv, accessed March 10, 2026,

#### https://arxiv.org/html/2603.01433v

#### 19. ObjectFormer for Image Manipulation Detection and Localization ..., accessed

#### March 10, 2026,

#### https://www.researchgate.net/publication/363909282_ObjectFormer_for_Image_

#### Manipulation_Detection_and_Localization

#### 20. TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and

#### Localization - grip-unina, accessed March 10, 2026,

#### https://grip-unina.github.io/TruFor/

#### 21. MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation

#### Detection, accessed March 10, 2026,

#### https://www.researchgate.net/publication/357114617_MVSS-Net_Multi-View_Multi

#### -Scale_Supervised_Networks_for_Image_Manipulation_Detection

#### 22. REFORGE: A Robust Ensemble for Image Forgery Detection and Localization in

#### Social Network Images - IEEE Xplore, accessed March 10, 2026,

#### https://ieeexplore.ieee.org/iel8/6287639/11323511/11346498.pdf

#### 23. (PDF) REFORGE: A Robust Ensemble for Image Forgery Detection and

#### Localization in Social Network Images - ResearchGate, accessed March 10, 2026,

#### https://www.researchgate.net/publication/399712748_REFORGE_A_Robust_Ense

#### mble_for_Image_Forgery_Detection_and_Localization_in_Social_Network_Images

#### 24. Enhanced Copy-Move Forgery Detection in Images Using Hybrid Deep Learning

#### Approach - EPJ Web of Conferences, accessed March 10, 2026,

#### https://www.epj-conferences.org/articles/epjconf/pdf/2025/26/epjconf_icatcict

#### 25_01042.pdf

#### 25. SegFormer - NVIDIA Docs, accessed March 10, 2026,

#### https://docs.nvidia.com/tao/tao-toolkit-archive/5.2.0/text/semantic_segmentation

#### /segformer.html

#### 26. Research on the performance of the SegFormer model with fusion of edge

#### feature extraction for metal corrosion detection - PMC, accessed March 10, 2026,

#### https://pmc.ncbi.nlm.nih.gov/articles/PMC11890780/

#### 27. Image Forgery Detection - grip-unina, accessed March 10, 2026,

#### https://www.grip.unina.it/multimedia-forensics/image-forgery-detection

#### 28. FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal

#### Large Language Models | OpenReview, accessed March 10, 2026,

#### https://openreview.net/forum?id=pAQzEY7M

#### 29. DOCFORGE-BENCH: A Comprehensive Benchmark for Document Forgery


#### Detection and Analysis - arXiv, accessed March 10, 2026,

#### https://arxiv.org/pdf/2603.

#### 30. (PDF) DOCFORGE-BENCH: A Comprehensive Benchmark for Document Forgery

#### Detection and Analysis - ResearchGate, accessed March 10, 2026,

#### https://www.researchgate.net/publication/401469940_DOCFORGE-BENCH_A_Co

#### mprehensive_Benchmark_for_Document_Forgery_Detection_and_Analysis

#### 31. Instance segmentation loss functions - SoftwareMill, accessed March 10, 2026,

#### https://softwaremill.com/instance-segmentation-loss-functions/

#### 32. Understanding Loss Functions for Deep Learning Segmentation Models -

#### Medium, accessed March 10, 2026,

#### https://medium.com/@devanshipratiher/understanding-loss-functions-for-deep-l

#### earning-segmentation-models-30187836b30a

#### 33. The combined focal loss and dice loss function improves the segmentation of

#### beta-sheets in medium-resolution cryo-electron-microscopy density maps -

#### PMC, accessed March 10, 2026,

#### https://pmc.ncbi.nlm.nih.gov/articles/PMC11590252/

#### 34. The Loss Functions That Actually Matter in 2025 | by Pranav Prakash I GenAI I

#### AI/ML I DevOps I | Medium, accessed March 10, 2026,

#### https://medium.com/@pranavprakash4777/the-loss-functions-that-actually-matt

#### er-in-2025-41b044b2645e

#### 35. Progressive Feedback-Enhanced Transformer for Image Forgery Localization -

#### arXiv, accessed March 10, 2026, https://arxiv.org/html/2311.08910v

#### 36. Multi-scale and deeply supervised network for image splicing localization -

#### Frontiers, accessed March 10, 2026,

#### https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.

#### 5.1655073/full

#### 37. arch.ipynb - Colab, accessed March 10, 2026,

#### https://colab.research.google.com/github/d2l-ai/d2l-tvm-colab/blob/master/chapt

#### er_gpu_schedules/arch.ipynb

#### 38. Understanding Google Colab Free GPU in detail | by Mehul Gupta | Data Science

#### in Your Pocket | Medium, accessed March 10, 2026,

#### https://medium.com/data-science-in-your-pocket/understanding-google-colab-f

#### ree-gpu-in-detail-15074081d

#### 39. How to implement mixed precision training with PyTorch? - Tencent Cloud,

#### accessed March 10, 2026, https://www.tencentcloud.com/techpedia/

#### 40. NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch | NVIDIA

#### Technical Blog, accessed March 10, 2026,

#### https://developer.nvidia.com/blog/apex-pytorch-easy-mixed-precision-training/

#### 41. What Every User Should Know About Mixed Precision Training in pytorch | by Hey

#### Amit | Data Scientist's Diary | Medium, accessed March 10, 2026,

#### https://medium.com/data-scientists-diary/what-every-user-should-know-about-

#### mixed-precision-training-in-pytorch-63c6544e5a

#### 42. Mixed Precision Training | Explanation and PyTorch Implementation from Scratch -

#### YouTube, accessed March 10, 2026,

#### https://www.youtube.com/watch?v=hHpC9Sywh4U


#### 43. Efficient Llama Training with Gradient Checkpointing and Adapters - Colab,

#### accessed March 10, 2026,

#### https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/noteb

#### ooks/Gradient_Checkpointing_Llama.ipynb

#### 44. How to implement gradient checkpointing for transformer models to reduce

#### memory usage?, accessed March 10, 2026,

#### https://massedcompute.com/faq-answers/?question=How%20to%20implement

#### %20gradient%20checkpointing%20for%20transformer%20models%20to%20re

#### duce%20memory%20usage?

#### 45. Memory consumption qlora with gradient checkpointing - Hugging Face Forums,

#### accessed March 10, 2026,

#### https://discuss.huggingface.co/t/memory-consumption-qlora-with-gradient-che

#### ckpointing/

#### 46. Fine-Tuning Code GenAI model on Google Colab T4 GPU: A Step-by-Step Guide,

#### accessed March 10, 2026,

#### https://www.tothenew.com/blog/fine-tuning-code-genai-model-on-google-cola

#### b-t4-gpu-a-step-by-step-guide/

#### 47. Understanding the Limits of Google Colab Pro RAM and GPU Usage - YouTube,

#### accessed March 10, 2026, https://www.youtube.com/watch?v=GY7yObI6Zi

#### 48. google-research/vision_transformer - GitHub, accessed March 10, 2026,

#### https://github.com/google-research/vision_transformer

#### 49. A practical guide to GPU memory for fine-tuning AI models | Google Cloud Blog,

#### accessed March 10, 2026,

#### https://cloud.google.com/blog/topics/developers-practitioners/decoding-high-ba

#### ndwidth-memory-a-practical-guide-to-gpu-memory-for-fine-tuning-ai-models/

#### 50. Introducing Mixed Precision Training in Opacus - PyTorch, accessed March 10,

#### 2026, https://pytorch.org/blog/introducing-mixed-precision-training-in-opacus/

#### 51. MMFD-Net: A Novel Network for Image Forgery Detection and ..., accessed

#### March 10, 2026, https://www.mdpi.com/2227-7390/13/19/

#### 52. VAAS: Vision-Attention Anomaly Scoring for Image Manipulation Detection in

#### Digital Forensics - arXiv.org, accessed March 10, 2026,

#### https://arxiv.org/html/2512.15512v

#### 53. DAE-Net: Dual Attention Mechanism and Edge Supervision Network for Image

#### Manipulation Detection and Localization - IEEE Xplore, accessed March 10, 2026,

#### https://ieeexplore.ieee.org/iel8/19/4407674/10658981.pdf

#### 54. Weakly-Supervised Image Forgery Localization via Vision-Language Collaborative

#### Reasoning Framework - arXiv, accessed March 10, 2026,

#### https://arxiv.org/html/2508.01338v

#### 55. Robust Detection and Localization of Image Copy-Move Forgery Using

#### Multi-Feature Fusion, accessed March 10, 2026,

#### https://pmc.ncbi.nlm.nih.gov/articles/PMC12941880/

#### 56. image_tampering_detection/phase2 (SRM+Fake,ELA+Unet).ipynb at master -

#### GitHub, accessed March 10, 2026,

#### https://github.com/enviz/image_tampering_detection/blob/master/phase2%20(SR

#### M%2BFake%2CELA%2BUnet).ipynb


#### 57. SegFormer Tutorial: Master Semantic Segmentation Fast - Labellerr, accessed

#### March 10, 2026, https://www.labellerr.com/blog/segformer/

#### 58. (PDF) Effective Image Tampering Localization via Enhanced Transformer and

#### Co-Attention Fusion - ResearchGate, accessed March 10, 2026,

#### https://www.researchgate.net/publication/379266243_Effective_Image_Tampering

#### _Localization_via_Enhanced_Transformer_and_Co-Attention_Fusion

#### 59. TruFor: Leveraging all-round clues for trustworthy image forgery detection and

#### localization, accessed March 10, 2026,

#### https://www.researchgate.net/publication/366497579_TruFor_Leveraging_all-roun

#### d_clues_for_trustworthy_image_forgery_detection_and_localization

#### 60. Convolutional PyTorch debayering / demosaicing layers - GitHub, accessed

#### March 10, 2026, https://github.com/cheind/pytorch-debayer

#### 61. CLUE: Leveraging Low-Rank Adaptation to Capture Latent Uncovered Evidence

#### for Image Forgery Localization - arXiv, accessed March 10, 2026,

#### https://arxiv.org/html/2508.07413v

#### 62. SDiFL: Stable Diffusion-Driven Framework for Image Forgery Localization - arXiv,

#### accessed March 10, 2026, https://arxiv.org/pdf/2508.

#### 63. CVPR Poster Towards Enhanced Image Inpainting: Mitigating Unwanted Object

#### Insertion and Preserving Color Consistency, accessed March 10, 2026,

#### https://cvpr.thecvf.com/virtual/2025/poster/

#### 64. No Pixel Left Behind: A Detail-Preserving Architecture for Robust High-Resolution

#### AI-Generated Image Detection | OpenReview, accessed March 10, 2026,

#### https://openreview.net/forum?id=9QQ3Kc2hj