# Best Notebooks -- Curated Selection

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Top 5 notebooks by result quality and assignment alignment |
| **Selection Pool** | 129 notebooks across ETASR, Pretrained, Standalone, and Legacy tracks |
| **Assignment** | Tampered Image Detection & Localization (Big Vision Internship) |

---

## Selection Criteria

The assignment requires **both image-level detection AND pixel-level localization**. Notebooks were ranked by:

1. **Result quality** -- Pixel F1, IoU, Image Accuracy, Macro F1, ROC-AUC
2. **Assignment alignment** -- Does the notebook produce localization masks, visual results (Original/GT/Predicted/Overlay), and standard metrics?
3. **Experimental quality** -- Methodology rigor, reproducibility, documentation

---

## Tier 1: Full Assignment Alignment (Detection + Localization)

These notebooks produce pixel-level tampered region masks and satisfy all assignment requirements.

### 1. vR.P.19 -- Multi-Q RGB ELA 9ch (New Best Localization)

| Metric | Value |
|--------|-------|
| Pixel F1 | **0.7965** (series best) |
| Pixel IoU | **0.6622** (series best) |
| Pixel AUC | **0.9707** (series best) |
| Image Accuracy | 90.85% |
| Image ROC-AUC | **0.9740** (series best) |
| Quality Score | 82/100 |
| Verdict | **POSITIVE** (+10.45pp from P.3 baseline) |

**Why #1:** Highest localization metrics across all experiments by a wide margin. Multi-quality RGB ELA (Q=75/85/95) captures forensic artifacts across the compression spectrum in 9 channels. +6.88pp F1 over previous best P.10 without any attention mechanism -- proves input pipeline is the dominant factor.

---

### 2. vR.P.10 -- CBAM Attention (Best Attention-Based Localization)

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7277 |
| Pixel IoU | 0.5719 |
| Pixel AUC | 0.9573 |
| Image Accuracy | 87.32% |
| Image ROC-AUC | 0.9633 |
| Quality Score | 87/100 |
| Verdict | **POSITIVE** (+3.57pp from P.3 baseline) |

**Why selected:** Best localization metrics across all 129 notebooks. CBAM (Convolutional Block Attention Module) attention mechanism focuses the decoder on tampered regions. Achieves the lowest false positive rate (2.0%) in the pretrained track. Uses ELA input with frozen ResNet-34 encoder.

**Contents:** Source notebook, Kaggle run, audit report, 5 doc files (experiment description, implementation plan, expected outcomes, results template, implementation script).

---

### 3. vR.P.7 -- Extended Training (Runner-up Localization)

| Metric | Value |
|--------|-------|
| Pixel F1 | **0.7154** |
| Pixel IoU | **0.5569** |
| Pixel AUC | 0.9504 |
| Image Accuracy | 87.37% |
| Quality Score | **88/100** (highest in pretrained track) |
| Verdict | **POSITIVE** (+2.34pp from P.3 baseline) |

**Why selected:** Highest quality score in the pretrained track thanks to rigorous methodology. Extended training from 25 to 50 epochs with best model at epoch 36. Proves that the P.3 ELA architecture had not converged at 25 epochs. Lowest false negative rate (25.9%) -- misses the fewest tampered images.

**Contents:** Source notebook, Kaggle run, audit report, 5 doc files (experiment description, implementation plan, expected outcomes, results template, implementation script).

---

### 3. vR.P.4 -- 4-Channel RGB+ELA Input (3rd Best Localization)

| Metric | Value |
|--------|-------|
| Pixel F1 | **0.7053** |
| Pixel IoU | **0.5447** |
| Pixel AUC | 0.9433 |
| Image Accuracy | 84.42% |
| Quality Score | **86/100** (best execution quality) |
| Verdict | NEUTRAL (+1.33pp from P.3 baseline) |

**Why selected:** Third highest Pixel F1 with the highest execution quality score (all cells pass, model saved, comprehensive evaluation). Innovative 4-channel RGB+ELA input design that modifies ResNet-34's first convolutional layer to accept both RGB and ELA as a unified input.

**Contents:** Source notebook, Kaggle run, audit report, 3 doc files (experiment description, implementation plan, expected outcomes).

---

## Tier 2: Best Classification (Partial Alignment -- No Localization)

These notebooks achieve the best image-level classification but do **not** produce pixel-level localization masks.

### 4. vR.1.6 -- Deeper CNN (Best Classification Metrics)

| Metric | Value |
|--------|-------|
| Test Accuracy | **90.23%** (series best) |
| Macro F1 | **0.9004** (only version to cross 0.90) |
| ROC-AUC | **0.9657** (series best) |
| Quality Score | 90/100 |
| Verdict | **POSITIVE** (+1.27pp from baseline) |

**Why selected:** Best classification result across all ETASR experiments. The **only** version to improve all three core metrics (Accuracy, Macro F1, ROC-AUC) from the honest baseline simultaneously. Deeper architecture (+Conv64+MaxPool) with 53% parameter reduction from the original paper architecture.

**Limitation:** Classification only -- cannot produce pixel-level masks.

**Contents:** Source notebook, Kaggle run, audit report, 3 doc files (version notes, architecture change, expected impact).

---

### 5. vR.1.7 -- Global Average Pooling (Best Experimental Quality + Efficiency)

| Metric | Value |
|--------|-------|
| Test Accuracy | 89.17% |
| Macro F1 | 0.8901 |
| ROC-AUC | 0.9495 |
| Total Parameters | **63,970** (99.8% reduction from paper) |
| Quality Score | **91/100** (highest across ALL tracks) |
| Verdict | NEUTRAL (-1.06pp from vR.1.6) |

**Why selected:** Highest experimental quality score in the entire project. Achieves 89.17% accuracy with only 64K parameters -- 214x more parameter-efficient than the next-best version. GlobalAveragePooling2D replaces Dense layers, proving that flattening is a bottleneck in the paper architecture. Lowest false negative rate (5.3%) -- catches the most tampered images.

**Limitation:** Classification only -- cannot produce pixel-level masks.

**Contents:** Source notebook, Kaggle run, audit report, 3 doc files (version notes, architecture change, expected impact).

---

## Cross-Track Summary

| Rank | Version | Track | Best At | Pixel F1 | Img Acc | Score |
|------|---------|-------|---------|----------|---------|-------|
| 1 | **vR.P.10** | Pretrained | Localization | 0.7277 | 87.32% | 87 |
| 2 | **vR.P.7** | Pretrained | Quality + Localization | 0.7154 | 87.37% | 88 |
| 3 | **vR.P.4** | Pretrained | Execution + Localization | 0.7053 | 84.42% | 86 |
| 4 | **vR.1.6** | ETASR | Classification | N/A | 90.23% | 90 |
| 5 | **vR.1.7** | ETASR | Efficiency + Quality | N/A | 89.17% | 91 |

**For assignment submission:** Use **vR.P.10** (best localization) or **vR.P.7** (best overall quality). These satisfy all assignment deliverables including pixel-level masks, visual results, and standard metrics.

---

## Folder Contents

Each subfolder contains:

| File Type | Description |
|-----------|-------------|
| Source notebook (`.ipynb`) | The notebook as uploaded to Kaggle |
| Run notebook (`.ipynb`) | The executed Kaggle output with all results |
| Audit report (`.md`) | Detailed post-run analysis and verdict |
| `docs/` folder | Pre-experiment documentation (description, plan, expected outcomes) |
