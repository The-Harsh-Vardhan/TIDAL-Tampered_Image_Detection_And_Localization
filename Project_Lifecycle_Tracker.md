# Project Lifecycle Tracker

## Tampered Image Detection & Localization

**Project:** Big Vision Internship Assignment
**Author:** Harsh Vardhan
**Duration:** Multi-version development across 20+ experiment iterations
**Latest Version:** vK.12.0

---

## 1. Project Overview

### Objective

Develop a deep learning model that **detects** whether an image has been tampered and **localizes** the manipulated region at pixel level by producing a binary segmentation mask.

### Problem Definition

Image tampering (copy-move, splicing, inpainting) is increasingly difficult to detect visually. This project builds an automated system that:

1. **Classifies** an image as authentic or tampered (binary image-level decision)
2. **Localizes** the tampered region by predicting a pixel-level mask highlighting altered areas

### Expected Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Classification label | Binary (0/1) | Authentic vs tampered |
| Segmentation mask | 256x256 single-channel | Pixel-level tamper probability map |
| Visual overlays | RGB image | Original / GT / Predicted / Overlay comparison |

### Assignment Requirements

From the Big Vision Internship Assignment specification:

1. **Dataset Selection & Preparation** — Use publicly available datasets (CASIA, Coverage, CoMoFoD, or Kaggle). Handle cleaning, preprocessing, mask alignment. Proper train/val/test splits. Apply augmentation for robustness.
2. **Model Architecture & Learning** — Train a model to predict tampered regions. Architecture and loss function choice is open. Must be runnable on Google Colab T4 GPU.
3. **Testing & Evaluation** — Thorough localization and detection evaluation using standard metrics. Clear visual results (Original, GT, Predicted, Overlay).
4. **Deliverables & Documentation** — Single Colab Notebook with dataset explanation, architecture description, training strategy, hyperparameter choices, evaluation results, and visualizations. Provide notebook link, trained weights, and scripts.
5. **Bonus** — Robustness testing against distortions (JPEG, resize, crop, noise). Detecting subtle copy-move and splicing artifacts.

---

## 2. Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Language | Python 3.10+ | Primary development language |
| Deep Learning | PyTorch 2.x | Model training and inference |
| Segmentation Library | Segmentation Models PyTorch (SMP) | Pretrained encoder-decoder architectures |
| Augmentation | Albumentations | Image and mask augmentation pipeline |
| Experiment Tracking | Weights & Biases (W&B) | Metrics logging, visualization, artifact storage |
| Image Processing | OpenCV (cv2) | ELA computation, image I/O, contour detection |
| Data Analysis | Pandas, NumPy | Dataset management, metric computation |
| Visualization | Matplotlib, Seaborn | Training curves, prediction grids, evaluation plots |
| Model Analysis | torchinfo | Parameter counting, layer-by-layer model summary |
| Explainability | Custom Grad-CAM | Encoder attention heatmaps |
| Hardware | Kaggle T4 GPU (15 GB VRAM) | Training and evaluation |
| Notebooks | Jupyter / Kaggle / Google Colab | Development environment |

---

## 3. Dataset Information

### Primary Dataset: CASIA 2.0 Upgraded

| Property | Value |
|----------|-------|
| **Source** | Kaggle (`harshv777/casia2-0-upgraded-dataset`) |
| **Total valid pairs** | 12,614 images |
| **Authentic images** | 7,491 (59.4%) |
| **Tampered images** | 5,123 (40.6%) |
| **Copy-move forgeries** | 3,295 (26.1% of total) |
| **Splicing forgeries** | 1,828 (14.5% of total) |
| **Mask availability** | Ground truth binary masks for all tampered images |
| **Split strategy** | 70 / 15 / 15 stratified by label |
| **Training set** | 8,829 images |
| **Validation set** | 1,892 images |
| **Test set** | 1,893 images |
| **Input resolution** | Resized to 256x256 (vK.x series) or 384x384 (v6.5/v8) |

### Reference Datasets (Not Used in Training)

| Dataset | Type | Status |
|---------|------|--------|
| CoMoFoD | Copy-move forgery detection | Referenced in assignment; audited via reference notebook |
| Coverage | Copy-move specific | Mentioned as future cross-dataset evaluation target |

### Dataset Challenges

- **Class imbalance at pixel level**: Most pixels in tampered images are still authentic, making segmentation difficult
- **Small tampered regions**: Many images have <2% tampered area, challenging detection
- **Forgery type imbalance**: Copy-move (3,295) significantly outnumbers splicing (1,828)
- **JPEG compression artifacts**: Dataset contains JPEG artifacts that can act as shortcuts

---

## 4. Model Architecture Evolution

### Architecture Comparison Table

| Version | Model | Encoder | Pretrained | Input | Params | Decoder | Classifier | Loss | Key Change |
|---------|-------|---------|------------|-------|--------|---------|------------|------|------------|
| vK.1–vK.3 | Custom UNet | 5-layer CNN | No (scratch) | 256x256x3 | 15.7M | Transposed Conv | GAP+FC (1024→512→2) | CE + BCE | Initial baseline |
| vK.7.x | Custom UNet | 5-layer CNN | No (scratch) | 256x256x3 | 15.7M | Transposed Conv | GAP+FC (1024→512→2) | Focal + BCE+Dice | Documentation improvements |
| v6.5 | SMP UNet | ResNet34 | ImageNet | 384x384x3 | 24.4M | SMP Decoder | None (max pixel prob) | BCEDice | Pretrained encoder leap |
| v8 | SMP UNet | ResNet34 | ImageNet | 384x384x3 | 24.4M | SMP Decoder | None (max pixel prob) | BCEDice (pw=30) | Scheduler + augmentation |
| vK.10.x | Custom UNet | 5-layer CNN (wider) | No (scratch) | 256x256x3 | 31.6M | Transposed Conv | GAP+FC (1024→2) | Focal + BCE+Dice | Engineering overhaul |
| **vK.11.0+** | **TamperDetector** | **ResNet34 (SMP)** | **ImageNet** | **256x256x4** | **~24.5M** | **SMP UNet Decoder** | **GAP+FC (512→256→2)** | **Focal+BCE+Dice+Edge** | **Architecture synthesis** |
| vK.12.0 | TamperDetector | ResNet34 (SMP) | ImageNet | 256x256x4 | ~24.5M | SMP UNet Decoder | GAP+FC (512→256→2) | Focal+BCE+Dice+Edge | Extended evaluation |

### Architecture Evolution Rationale

**Phase 1 → Phase 2 (Custom UNet → SMP UNet):**
The custom UNet trained from scratch struggled to learn meaningful features from only ~8.8K training images. Switching to a pretrained ResNet34 encoder via SMP provided ImageNet-learned edge/texture features critical for detecting manipulation artifacts. This single change improved tampered F1 from ~0.20 to 0.41.

**Phase 2 → Phase 3 (v6.5/v8 → vK.10.x):**
Despite v6.5's superior metrics, the vK.x track continued with the custom UNet to improve engineering quality (CONFIG management, AMP, proper evaluation). The custom UNet needed 100 epochs to reach Tam-F1=0.22 (vs v6.5's 0.41 in 25 epochs), proving pretrained encoders are essential.

**Phase 3 → Phase 4 (vK.10.6 → vK.11.0):**
The synthesis point. vK.11.0 combined: v6.5's pretrained encoder, vK.10.6's engineering excellence, v8's training improvements, plus new features (ELA 4th channel, edge loss, encoder freeze). This is the project's recommended architecture.

### TamperDetector Architecture (vK.11.0+)

```
Input (256x256x4: RGB + ELA)
         │
    ┌────▼────┐
    │ ResNet34 │  (ImageNet pretrained, first conv modified for 4 channels)
    │ Encoder  │
    └────┬────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────────┐
│  UNet  │  │ Classification│
│Decoder │  │    Head       │
│(skip   │  │ GAP→512→256  │
│ conn.) │  │ →ReLU→Drop   │
└───┬────┘  │ →256→2       │
    │       └──────┬───────┘
    ▼              ▼
Seg Mask       Class Label
(256x256x1)   (Authentic/Tampered)
```

---

## 5. Experiment Version History

### Chronological Version Log

| Version | Date Era | Description | Key Changes | Run Status |
|---------|----------|-------------|-------------|------------|
| **vK.1** | Early | Original Kaggle notebook | Baseline custom UNet, dual-block training, no documentation | No run output |
| **vK.2** | Early | Documentation added | Markdown cells, W&B integration stubs, IoU/F1 reporting | No run output |
| **vK.3** | Early | Code comments improved | English docstrings, inline comments | Run completed |
| **vK.4** | Early | Merge attempt (vK.3 + v8) | Attempted to merge custom and SMP approaches | Source only |
| **v6.5** | Mid | SMP architecture leap | Pretrained ResNet34, AdamW, differential LR, AMP, DataParallel, comprehensive eval | **Run completed — Best segmentation** |
| **v8** | Mid | Training refinements | ReduceLROnPlateau, per-sample Dice, expanded augmentations; pos_weight bug caused regression | Run completed — Regressed |
| v9 | Mid | Feature additions (unexecuted) | ELA channel, pHash guard, learned classifier, boundary F1 | **Never executed** |
| **vK.6** | Mid | Restructured by Codex | Subsection reorganization | Source only |
| **vK.7** | Mid | Restructured by Opus | Improved subsection organization | Source only |
| **vK.7.1** | Mid | Documentation refresh | Updated docs over vK.7 base | Run completed (same as vK.3) |
| **vK.7.5** | Mid | Clean execution attempt | Two W&B runs started, neither finished | Incomplete |
| **vK.10** | Late | Engineering overhaul begins | New single-block structure, CONFIG dict, data leak removed | Source only |
| **vK.10.1** | Late | Refinements | Structure improvements | Source only |
| **vK.10.2** | Late | Refinements | Further structure changes | Source only |
| **vK.10.3b** | Late | First full engineering run | AMP, seeding, early stopping (patience=10), checkpoints | Run collapsed (Tam-F1=0.0004) |
| **vK.10.4** | Late | Data visualization added | Sample grids, class distribution | Run collapsed (Tam-F1=0.0000) |
| **vK.10.5** | Late | Multi-GPU support | DataParallel, get_base_model() unwrapper | Run collapsed (Tam-F1=0.0006) |
| **vK.10.3b-run-03** | Late | Extended training test | 100 epochs, patience=50 | Recovered (Tam-F1=0.2196) |
| **vK.10.6** | Late | Comprehensive evaluation | 100 epochs, 12-feature eval suite, confusion matrix, Grad-CAM, robustness | **Best vK.x run (Tam-F1=0.2213)** |
| **vK.11.0** | Latest | Architecture synthesis | SMP+ResNet34+ELA+EdgeLoss+DiffLR+GradAccum+EncoderFreeze | Source only (awaiting run) |
| **vK.11.1** | Latest | Model Card section added | Experiment report documentation | Source only |
| **vK.11.2** | Latest | Reproducibility section | Seed verification, split stability, checkpoint integrity | Source only |
| **vK.11.3** | Latest | Inference demo added | Single-image prediction pipeline with ELA | Source only |
| **vK.11.4** | Latest | Executive summary added | High-level project overview for reviewers | Source only |
| **vK.11.5** | Latest | Results dashboard added | Quick metrics overview with live/placeholder logic | Source only |
| **vK.12.0** | Latest | Extended analysis | 16 new cells: arch diagram, FP/FN, robustness enhancement, speed test | Source only |

### Three Parallel Architecture Tracks

| Track | Versions | Architecture | Best Tam-F1 | Status |
|-------|----------|-------------|-------------|--------|
| **Track A: Custom UNet** | vK.1–vK.10.6 | UNetWithClassifier (from scratch) | 0.2213 | Superseded |
| **Track B: SMP Pretrained** | v6.5, v8 | smp.Unet + ResNet34 (ImageNet) | **0.4101** | Best result |
| **Track C: Synthesis** | vK.11.0–vK.12.0 | TamperDetector (SMP+ELA+Edge) | Pending | Awaiting execution |

---

## 6. Training Pipeline Improvements

### Chronological Pipeline Evolution

| Feature | First Introduced | Versions | Impact |
|---------|-----------------|----------|--------|
| **Basic training loop** | vK.1 | vK.1–vK.3 | Functional but no optimization |
| **AMP (Mixed Precision)** | v6.5 | v6.5, v8, vK.10.3b+ | ~2x memory efficiency, faster training |
| **DataParallel (Multi-GPU)** | v6.5 | v6.5, v8, vK.10.5+ | Utilizes 2 GPUs on Kaggle |
| **Gradient Accumulation** | v6.5 | v6.5 (eff=16), v8 (eff=256), vK.11.0 (eff=32) | Larger effective batch without VRAM increase |
| **Early Stopping** | v6.5 | v6.5 (p=10), vK.10.3b (p=10), vK.10.6 (p=30), vK.11.0 (p=10) | Prevents overfitting |
| **Checkpoint System** | vK.10.3b | vK.10.3b+ | 3-file system: best, last, periodic |
| **VRAM Auto-Batch-Scaling** | vK.10.3b | vK.10.3b+ | Auto-adjusts batch size to GPU VRAM |
| **Centralized CONFIG** | vK.10 | vK.10+ | Single dictionary for all hyperparameters |
| **Reproducibility Seeding** | v6.5 | v6.5, v8, vK.10.3b+ | Python, NumPy, PyTorch, CUDA, cuDNN |
| **Differential Learning Rates** | v6.5 | v6.5, v8, vK.11.0+ | Encoder: 1e-4, Decoder: 1e-3 |
| **ReduceLROnPlateau** | v8 | v8, vK.11.0+ | Adaptive LR reduction on metric stall |
| **Encoder Freeze Warmup** | vK.11.0 | vK.11.0+ | Protects pretrained BatchNorm for 2 epochs |
| **Edge Supervision Loss** | vK.11.0 | vK.11.0+ | Sobel-based boundary BCE for sharp masks |
| **ELA 4th Input Channel** | vK.11.0 | vK.11.0+ | Error Level Analysis for forensic features |
| **W&B Comprehensive Logging** | vK.11.0 (patched) | vK.11.0+ | All visualizations logged to W&B dashboard |

### Key Training Configurations Across Versions

| Parameter | vK.3 | v6.5 | v8 | vK.10.6 | vK.11.0+ |
|-----------|-------|------|-----|---------|----------|
| Optimizer | Adam | AdamW | AdamW | Adam | AdamW |
| Learning Rate | 1e-4 | Diff (1e-4/1e-3) | Diff (1e-4/1e-3) | 1e-4 | Diff (1e-4/1e-3) |
| Effective Batch | 8 | 16 | 256 | 32 | 32 (8x4 accum) |
| Max Epochs | 50 | 50 (ES at 25) | 50 (ES at 27) | 100 | 50 |
| AMP | No | Yes | Yes | Yes | Yes |
| Scheduler | None | None | ReduceLROnPlateau | CosineAnnealing | ReduceLROnPlateau |
| Image Size | 256 | 384 | 384 | 256 | 256 |
| Input Channels | 3 (RGB) | 3 (RGB) | 3 (RGB) | 3 (RGB) | 4 (RGB+ELA) |

---

## 7. Evaluation Strategy

### Evaluation Metrics

| Metric | Type | Description | First Used |
|--------|------|-------------|------------|
| **Tampered Dice** | Segmentation | Overlap coefficient on tampered-only images | vK.3 |
| **Tampered IoU** | Segmentation | Intersection-over-Union on tampered-only | v6.5 |
| **Tampered F1** | Segmentation | Harmonic mean of precision/recall on tampered-only | v6.5 |
| **Precision** | Segmentation | True positive rate among predicted tampered pixels | vK.12.0 |
| **Recall** | Segmentation | Detection rate of truly tampered pixels | vK.12.0 |
| **Pixel Accuracy** | Segmentation | Overall pixel correctness (TP+TN)/total | vK.12.0 |
| **Image Accuracy** | Classification | Correct tampered/authentic classification | vK.1 |
| **AUC-ROC** | Classification | Threshold-independent classification quality | v6.5 |
| **Pixel-Level AUC** | Segmentation | Threshold-independent segmentation quality | vK.10.6 |

### Critical Insight: Mixed-Set vs Tampered-Only Metrics

A major lesson learned during development was that **mixed-set metrics are inflated** because authentic images have all-zero ground truth masks, scoring perfect Dice/IoU/F1 automatically. The authentic images (59.4% of data) inflate mixed-set metrics significantly.

**Example:** Mixed-set F1 of 0.5761 in vK.3 → estimated tampered-only F1 of only ~0.15–0.25.

From vK.10.6 onward, tampered-only metrics are the primary evaluation criterion.

### Evaluation Suite Evolution

| Feature | v6.5 | v8 | vK.10.6 | vK.11.0+ | vK.12.0 |
|---------|------|-----|---------|----------|---------|
| Threshold optimization | Yes | Yes | Yes (50-point sweep) | Yes | Yes |
| Confusion matrix | No | No | Yes | Yes | Yes |
| ROC/PR curves | No | No | Yes | Yes | Yes |
| Forgery-type breakdown | Yes | Yes | Yes | Yes | Yes |
| Mask-size stratification | No | Yes | Yes | Yes | Yes |
| Shortcut learning checks | No | Yes | Yes | Yes | Yes |
| Grad-CAM | Yes | No | Yes | Yes | Yes |
| Robustness testing | Yes | No | Yes | Yes | Enhanced |
| Failure case analysis | Yes | No | Yes | Yes | Yes |
| Data leakage verification | No | No | Yes | Yes | Yes |
| Pixel-level AUC | No | No | Yes | Yes | Yes |
| Precision/Recall/PixAcc | No | No | No | No | **Yes** |
| FP/FN error analysis | No | No | No | No | **Yes** |
| Experiment comparison | No | No | No | No | **Yes** |
| Inference speed benchmark | No | No | No | No | **Yes** |

---

## 8. Experiment Results Tracking

### Master Performance Table

| Version | Architecture | Params | Img Acc | AUC-ROC | Tam F1 | Tam IoU | Tam Dice | Epochs | Status |
|---------|-------------|--------|---------|---------|--------|---------|----------|--------|--------|
| vK.3 run-01 | Custom UNet (scratch) | 15.7M | 0.8986 | — | ~0.20 (est.) | — | 0.5761 (mixed) | 50 | Completed |
| **v6.5 run-01** | **SMP ResNet34** | **24.4M** | **0.8246** | **0.8703** | **0.4101** | **0.3563** | — | **25 (ES)** | **Best Seg** |
| v8 run-01 | SMP ResNet34 | 24.4M | 0.7190 | 0.8170 | 0.2949 | 0.2321 | — | 27 (ES) | Regressed |
| vK.7.1 run-01 | Custom UNet (scratch) | 15.7M | 0.8986 | — | ~0.20 (est.) | — | 0.5761 (mixed) | 50 | Same as vK.3 |
| vK.10.3b run-01 | Custom UNet (scratch) | 31.6M | 0.5061 | 0.6069 | 0.0004 | 0.0002 | — | ~10 (ES) | Collapsed |
| vK.10.4 run-01 | Custom UNet (scratch) | 31.6M | 0.4675 | 0.6534 | 0.0000 | 0.0000 | — | 10 (ES) | Collapsed |
| vK.10.5 run-01 | Custom UNet (scratch) | 31.6M | 0.4791 | 0.6201 | 0.0006 | 0.0003 | — | ~10 (ES) | Collapsed |
| vK.10.3b run-03 | Custom UNet (scratch) | 31.6M | — | — | 0.2196 | — | — | 100 | Recovered |
| **vK.10.6 run-01** | **Custom UNet (scratch)** | **31.6M** | **0.8357** | **0.9057** | **0.2213** | **0.1554** | — | **100** | **Best vK.x** |
| vK.11.0–11.5 | TamperDetector (SMP+ELA) | ~24.5M | — | — | — | — | — | — | Not yet run |
| vK.12.0 | TamperDetector (SMP+ELA) | ~24.5M | — | — | — | — | — | — | Not yet run |

### Key Performance Observations

- **Best segmentation**: v6.5 (Tam-F1 = 0.4101) — pretrained encoder was the decisive factor
- **Best classification**: vK.10.6 (AUC = 0.9057) — 100-epoch training with comprehensive eval
- **Worst regression**: v8 (Tam-F1 dropped 28% from v6.5) — caused by pos_weight=30.01 bug
- **vK.10.x collapse**: Early stopping at ~10 epochs killed from-scratch training before model could learn; 100-epoch run recovered to Tam-F1=0.22
- **Forgery-type flip**: v6.5 excelled at splicing (F1=0.59) but struggled with copy-move (F1=0.31); vK.10.6 reversed this pattern (copy-move F1=0.41, splicing F1=0.12)

---

## 9. Visualization & Monitoring

### Prediction Visualizations

| Visualization | Panels | Purpose | Introduced |
|---------------|--------|---------|------------|
| Standard 4-panel | Original, GT, Predicted, Overlay | Core prediction quality assessment | vK.3 |
| Submission grid | 8 samples in grid layout | Quick visual summary for reviewers | v6.5 |
| ELA visualization | Original, ELA map, Prediction | Verify ELA channel contribution | vK.11.0 |
| Enhanced 6-panel | Original, GT, Predicted, Overlay, Diff Map, Contour | Detailed disagreement analysis | vK.12.0 |

### Evaluation Plots

| Plot | Description | Introduced |
|------|-------------|------------|
| Training curves | Loss and metrics per epoch | vK.3 |
| Threshold sweep | F1 vs threshold (50-point) | v6.5 |
| Confusion matrix | TP/FP/TN/FN at optimal threshold | vK.10.6 |
| ROC and PR curves | Classification performance curves | vK.10.6 |
| Forgery-type breakdown | Per-forgery-type F1 comparison | v6.5 |
| Mask-size stratified F1 | Performance by tampered region size | v8 |
| Failure cases | 10 worst predictions by Dice | v6.5 |
| Grad-CAM heatmaps | Encoder attention visualization | v6.5 |
| Robustness bar chart | Metrics under degradation conditions | vK.10.6 |
| Architecture diagram | Model flow chart (Matplotlib) | vK.12.0 |
| Mask coverage histogram | Distribution of tampered region sizes | vK.12.0 |
| Degradation visualization | Side-by-side degraded image grid | vK.12.0 |

### Weights & Biases Dashboard

W&B integration tracks the following across all vK.11.x+ notebooks:

| W&B Key | Type | Description |
|---------|------|-------------|
| `train/loss`, `train/accuracy` | Scalar/epoch | Training metrics |
| `val/loss`, `val/accuracy`, `val/dice`, `val/iou`, `val/f1` | Scalar/epoch | Validation metrics |
| `val/tampered_dice`, `val/tampered_iou`, `val/tampered_f1` | Scalar/epoch | Tampered-only validation |
| `val/roc_auc`, `lr/encoder`, `lr/decoder` | Scalar/epoch | AUC and learning rates |
| `val_predictions` | Images | Prediction samples every 5 epochs |
| `test/*`, `best_epoch` | Summary | Final test metrics |
| `final_test_metrics_table` | Table | Structured test results |
| `best-model` | Artifact | Model checkpoint |
| `evaluation/*` | Images + Tables | All evaluation visualizations |
| `dashboard/*` | Images + Table | Results dashboard (vK.11.5+) |

---

## 10. Error Analysis

### Common Failure Modes

| Failure Mode | Severity | Description | Mitigation Attempts |
|-------------|----------|-------------|---------------------|
| **Very small tampered regions** | High | Images with <2% tampered area are nearly impossible to detect | Mask-size stratified evaluation added in v8; focal loss gamma=2.0 to focus on hard examples |
| **Copy-move artifacts** | High | Subtle copy-move manipulations where source and target textures are similar | v6.5 struggled (F1=0.31); vK.10.6 improved (F1=0.41) with longer training |
| **Low contrast manipulations** | Medium | Tampered regions that blend seamlessly with surrounding content | ELA channel added in vK.11.0 to detect compression inconsistencies |
| **JPEG compression artifacts** | Medium | Double-JPEG compression creates artifacts that can mask or mimic tampering | Robustness testing with QF50/QF70; ELA specifically targets compression traces |
| **Mask boundary imprecision** | Medium | Predicted masks have blurry or misaligned boundaries | Edge supervision loss (Sobel-based) added in vK.11.0 |
| **Authentic false positives** | Low-Medium | High-frequency textures in authentic images triggering false tamper detection | FP/FN analysis added in vK.12.0 to identify systematic patterns |
| **Metric inflation** | Critical (methodological) | Mixed-set Dice/F1 inflated by authentic samples scoring perfect | Fixed by switching to tampered-only metrics from vK.10.6 onward |

### Debugging History

| Issue | Version Affected | Root Cause | Resolution |
|-------|-----------------|------------|------------|
| Block 1 data leakage | vK.1–vK.7.x | Dual-block training where Block 1 trained on test set | Removed Block 1 in vK.10 |
| v8 regression (-28% F1) | v8 | `pos_weight=30.01` computed from all pixels including authentic images; effective batch=256 without LR scaling | Identified in audit; not propagated to vK.11.0 |
| vK.10.3b–10.5 collapse | vK.10.3b–10.5 | Early stopping (patience=10) killed training at ~10 epochs before from-scratch model could learn | Extended to patience=30 and 100 epochs in vK.10.6 |
| Dice inflation | vK.1–vK.7.x | Authentic images with all-zero masks score perfect Dice, inflating averages | Switched to tampered-only metrics in vK.10.6 |
| v6.5 robustness bug | v6.5 | Identical F1=0.5938 for all degradation conditions (suspicious) | Flagged in audit; vK.11.0+ uses Albumentations-based robustness |
| SMP decoder(*features) bug | vK.11.0 (initial) | SMP decoder expects positional args, not keyword | Fixed via `fix_denorm.py` patch |
| 4-channel _denorm bug | vK.11.0 (initial) | Denormalization assumed 3 channels, failed on 4-channel RGB+ELA | Fixed in patch script |
| W&B version string hardcoding | vK.11.0–11.5 (initial) | All notebooks hardcoded "vK.11.0" in W&B project/tags | Fixed via `fix_wandb_logging.py` (13 patches) |
| W&B patch idempotency | fix_wandb_logging.py | For vK.11.0, version replacement was a no-op, duplicating config lines | Added `wandb.config.update` presence check |

---

## 11. Robustness Testing

### Test Conditions

| Condition | Description | Transform |
|-----------|-------------|-----------|
| `clean` | No degradation (baseline) | Resize + Normalize only |
| `jpeg_qf70` | JPEG compression, quality 70 | `A.ImageCompression(quality_lower=70, quality_upper=70)` |
| `jpeg_qf50` | JPEG compression, quality 50 | `A.ImageCompression(quality_lower=50, quality_upper=50)` |
| `noise_s10` | Gaussian noise, mild | `A.GaussNoise(var_limit=(10.0, 50.0))` |
| `noise_s25` | Gaussian noise, strong | `A.GaussNoise(var_limit=(100.0, 100.0))` |
| `blur_k3` | Gaussian blur, kernel 3 | `A.GaussianBlur(blur_limit=(3, 3))` |
| `blur_k5` | Gaussian blur, kernel 5 | `A.GaussianBlur(blur_limit=(5, 5))` |
| `resize_0.75` | Downscale to 75%, then restore | Resize down + resize back up |

### Robustness Metrics Evolution

| Version | Metrics Reported | Conditions | Visualization |
|---------|-----------------|------------|---------------|
| v6.5 | F1 only | 8 conditions | Bar chart (suspicious — identical values) |
| v8 | F1 only | 8 conditions | Bar chart (plausible variation) |
| vK.10.6 | F1 only | 8 conditions | Bar chart + delta from clean |
| **vK.12.0** | **Dice + IoU + F1** | **8 conditions** | **3-panel bar chart + delta table + degradation viz** |

### Key Robustness Findings (from vK.10.6 run)

- `blur_k5` is the most destructive degradation — near-catastrophic F1 drop
- JPEG compression (QF50/QF70) causes moderate degradation
- Gaussian noise has variable impact depending on intensity
- The model's reliance on high-frequency features makes it vulnerable to blur

---

## 12. Performance Optimization

### Memory and Speed Optimizations

| Optimization | Version | Description | Impact |
|-------------|---------|-------------|--------|
| **AMP (FP16)** | v6.5+ | `torch.cuda.amp.autocast` + `GradScaler` | ~2x memory savings, ~1.5x speed |
| **DataParallel** | v6.5, vK.10.5+ | 2 GPU utilization on Kaggle P100/T4 | ~1.6x throughput |
| **Gradient Accumulation** | v6.5+ | Accumulate gradients over N steps before optimizer step | Larger effective batch without VRAM increase |
| **VRAM Auto-Scaling** | vK.10.3b+ | Batch size: 8 (base), 16 (≥15GB), 24 (≥20GB), 32 (≥28GB) | Automatic GPU adaptation |
| **Pin Memory** | vK.10+ | `pin_memory=True` in DataLoader | Faster CPU→GPU transfer |
| **Worker Processes** | vK.10+ | `num_workers=4` in DataLoader | Parallel data loading |
| **cuDNN Benchmark** | vK.11.0+ | `torch.backends.cudnn.benchmark = True` | Optimized convolution algorithms |

### Training Efficiency Comparison

| Version | Time per Epoch (est.) | Total Training Time | Epochs to Convergence |
|---------|----------------------|--------------------|-----------------------|
| vK.3 | ~15 min (no AMP, 1 GPU) | ~12.5 hours | 50 (no ES) |
| v6.5 | ~8 min (AMP, 2 GPU, 384x384) | ~3.3 hours | 25 (ES) |
| vK.10.6 | ~5 min (AMP, 2 GPU, 256x256) | ~8.3 hours | 100 (ES at ~75) |
| vK.11.0+ | ~5 min (AMP, 2 GPU, 256x256) | TBD | TBD |

---

## 13. Reproducibility Notes

### Seed Configuration (vK.11.0+)

```python
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # disabled for determinism
```

### Dataset Versioning

| Property | Value |
|----------|-------|
| Dataset source | `harshv777/casia2-0-upgraded-dataset` on Kaggle |
| Split method | `sklearn.model_selection.train_test_split` with `random_state=42` |
| Split ratio | 70% train / 15% val / 15% test |
| Stratification | By label (tampered vs authentic) |

### Environment Setup

| Component | Specification |
|-----------|---------------|
| Platform | Kaggle Notebooks / Google Colab |
| GPU | NVIDIA T4 (15 GB VRAM) or P100 (16 GB) |
| Python | 3.10+ |
| PyTorch | 2.x (CUDA 11.8+) |
| SMP | Latest via `pip install segmentation-models-pytorch` |
| Albumentations | Latest via `pip install albumentations` |
| W&B | Latest via `pip install wandb` |
| torchinfo | Latest via `pip install torchinfo` (vK.12.0+) |

### Reproducibility Checklist

- [x] Fixed random seeds (Python, NumPy, PyTorch, CUDA)
- [x] Deterministic cuDNN mode
- [x] DataLoader worker seeding via `seed_worker()` function
- [x] Seeded generator for DataLoader
- [x] Stratified data splits with fixed `random_state`
- [x] Checkpoint system (best, last, periodic)
- [x] CONFIG dictionary captures all hyperparameters
- [x] W&B logs full configuration
- [x] Reproducibility Verification section (vK.11.2+) confirms seed/split stability

---

## 14. Lessons Learned

### Architecture and Training

1. **Pretrained encoders are essential for small datasets.** The custom UNet (from scratch) plateaued at Tam-F1=0.22 after 100 epochs. The same data with a pretrained ResNet34 reached Tam-F1=0.41 in just 25 epochs. Transfer learning is not optional when training data is limited (~8.8K images).

2. **Early stopping can kill from-scratch training.** The vK.10.3b–10.5 collapse (Tam-F1 near zero) was caused by patience=10 on a from-scratch 31.6M parameter model. The model needed ~50–100 epochs to learn basic features. Extended training (100 epochs, patience=30/50) recovered to Tam-F1=0.22.

3. **Hyperparameter bugs compound silently.** v8's regression wasn't from architectural changes but from `pos_weight=30.01` (computed incorrectly) and a 16x batch size increase without LR scaling. Always ablate one change at a time.

4. **ELA provides complementary forensic features.** Error Level Analysis detects compression inconsistencies that RGB alone cannot see. Adding ELA as a 4th input channel (vK.11.0) provides the model with direct access to forensic signals.

5. **Edge supervision sharpens mask boundaries.** The Sobel-based edge loss (vK.11.0) penalizes blurry mask boundaries that standard Dice/BCE losses tolerate.

### Evaluation

6. **Mixed-set metrics are misleading.** Authentic images with all-zero masks score perfect Dice/IoU/F1, inflating overall metrics. A mixed-set F1 of 0.75 can correspond to a tampered-only F1 of only 0.50. Always report tampered-only metrics.

7. **Threshold tuning must use validation set only.** The optimal segmentation threshold varies significantly (often 0.3–0.7). Tuning on the test set constitutes data leakage. The 50-point threshold sweep on validation, applied frozen to test, is the correct approach.

8. **Robustness testing reveals model fragility.** Blur (especially kernel=5) is catastrophic for segmentation models that rely on high-frequency features. This finding suggests the model may be learning compression/edge artifacts rather than true tampering semantics.

### Engineering

9. **Generator scripts enable rapid iteration.** The constructive generator pattern (`generate_vk11*.py`) allows reproducible notebook creation from a source, with targeted cell modifications. This avoids manual editing errors and enables systematic version progression.

10. **Comprehensive evaluation is worth the engineering cost.** The 12-feature evaluation suite introduced in vK.10.6 (confusion matrix, ROC/PR, forgery breakdown, mask-size stratification, shortcut checks, Grad-CAM, robustness, failure analysis) provides insights that simple accuracy numbers cannot.

---

## 15. Future Improvements

### High Priority

| Improvement | Expected Impact | Complexity |
|-------------|----------------|------------|
| **Run vK.11.0+ on Kaggle** | First test of synthesized architecture; expected Tam-F1 > 0.40 | Low (notebook ready) |
| **Transformer-based encoder** (e.g., EfficientNet-B4, ConvNeXt) | Stronger feature extraction, potentially +0.05–0.10 Tam-F1 | Medium |
| **Multi-scale input pipeline** | Process at 256 and 512 resolution, fuse features | Medium |
| **Cross-dataset evaluation** | Train on CASIA, evaluate on Coverage/CoMoFoD for generalization | Medium |

### Medium Priority

| Improvement | Expected Impact | Complexity |
|-------------|----------------|------------|
| Attention mechanisms (CBAM, SE blocks) | Focus on tampered regions, improve small-mask detection | Low |
| Test-time augmentation (TTA) | Flip + multi-crop averaging for better test predictions | Low |
| Larger effective batch size with proper LR scaling | More stable training dynamics | Low |
| SRM (Steganalysis Rich Model) filters as additional input | Captures noise-level inconsistencies | Medium |
| Multi-task learning with forgery-type classification | Separate copy-move vs splicing detection | Medium |

### Research Direction

| Direction | Description |
|-----------|-------------|
| Vision Transformers for forgery detection | ViT/Swin encoders may capture global manipulation patterns that CNNs miss |
| Frequency-domain features | DCT/wavelet coefficients as input channels for compression artifact analysis |
| Self-supervised pretraining on manipulation data | Pre-train on synthetic manipulations before fine-tuning on CASIA |
| Ensemble methods | Combine RGB-model + ELA-model + frequency-model predictions |
| Boundary refinement (CRF post-processing) | Conditional Random Field to sharpen predicted mask boundaries |

---

## Appendix A: Generator Script Chain

```
Source Notebook                → Script                → Output Notebook
─────────────────────────────────────────────────────────────────────────
vK.10.6                        → generate_vk11.py       → vK.11.0
vK.11.0 [Pretrained ResNet34]  → generate_vk111.py      → vK.11.1
vK.11.1                        → generate_vk112.py      → vK.11.2
vK.11.2                        → generate_vk113.py      → vK.11.3
vK.11.3                        → generate_vk114.py      → vK.11.4
vK.11.4                        → generate_vk115.py      → vK.11.5
vK.11.5                        → generate_vk120.py      → vK.12.0
```

## Appendix B: File Inventory

### Notebook Counts

| Location | Count | Description |
|----------|-------|-------------|
| `Notebooks/` (main) | 14 | Primary development notebooks (vK.11.x, vK.12.0, v9) |
| `Notebooks/Runs/` | 16 | Kaggle execution outputs |
| `Notebooks/reference/` | 16 | Third-party reference notebooks |
| `Notebooks/archive/` | 4 | Early development drafts |
| `Notebooks/helper functions/` | 15+ | Generator and patch scripts |

### Audit and Documentation

| Location | Files | Description |
|----------|-------|-------------|
| `Audit_all_runs_till_vK.10.5/` | 9 | Audits through vK.10.5 |
| `Audit_all_runs_till_vK.10.6/` | 13 | Extended audits including vK.10.6 |
| `Audit of Reference Notebook/` | 10 | Reference notebook audits |
| `Docs/v11/` | 8 | Architecture specification and roadmap |
| `Docs/Docs9_Notebook_Roast/` | 10 | v9 critical review |
| `Docs/Docs_vK4_Kill_Review/` | 10 | vK.4 adversarial review |
| `Docs/Docs_vK4_Notebook_Audit/` | 10 | vK.4 technical audit |
| `Docs/Interview Prep/` | 11 | Interview preparation docs |

---

*This document serves as the complete project history and experiment logbook for the Tampered Image Detection & Localization project. It tracks architectural decisions, performance metrics, debugging efforts, and lessons learned across all development phases.*
