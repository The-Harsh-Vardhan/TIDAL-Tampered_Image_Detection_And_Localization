# Audit: Image Tempering Doc1

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `Image Tempering Doc1.md` (38 KB)

---

## Notebook Overview

This is **not a notebook** — it is a **literature review / survey document** covering five IEEE research papers on digital image forensics. It provides academic context for the project's tampering detection approach, summarizing key methodologies, datasets, and benchmark results from the published literature.

| Attribute | Value |
|---|---|
| Format | Markdown (.md), not Jupyter notebook |
| Length | ~194 lines of text content |
| Content | Survey of 5 IEEE papers |
| References | 25 citations |
| Date Accessed | ~March 11, 2026 |

---

## Dataset Pipeline Review

N/A — this is a literature review, not executable code.

---

## Model Architecture Review

N/A — no model implemented. The document surveys architectures from published papers:

### Papers Covered

| # | IEEE ID | Title | Method | Dataset | Reported Metric |
|---|---|---|---|---|---|
| 1 | 10444440 | Tempered Image Detection Using ELA and CNNs | ELA + 2-layer CNN | CASIA2 + Kaggle | Accuracy: 87.75% |
| 2 | 10052973 | Real or Fake? A Practical Method | Dual-task (cls + seg) + DQ analysis | — | — |
| 3 | 10168896 | Robust Algorithm for Copy-Rotate-Move | Zernike Moments + ACO matching | MICC-F220 | Accuracy: 98.44% |
| 4 | 10652417 | Deep Localization Using U-Net | U-Net + frequency features (FENet) | CASIA_v2 | Accuracy: ~95% |
| 5 | 10895348 | Image Forgery Detection Using MD5 & OpenCV | MD5 hashing + anomaly detection | — | — |

### Benchmark Comparison Table (from document)

| Method | Dataset | Metric |
|---|---|---|
| ELA + CNN | CASIA2 | 87.75% accuracy |
| Regularized U-Net | Validation set | 0.96 F1 |
| MobileNetV2 | Splicing/Copy-Move | 95% accuracy |
| Zernike + ACO | MICC-F220 | 98.44% accuracy |
| EfficientNetV2B0 | Transfer learning | Superior to MVSS-Net++, DRRU-Net |

---

## Training Pipeline Review

N/A — no training code.

---

## Evaluation Metrics Review

N/A — no evaluation code. The document reports metrics from external papers.

---

## Visualization Assessment

The document contains embedded base64 images for:
- Zernike moment mathematical formulas
- MD5 hash notation diagrams
- These are non-interactive static images

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| Academic Rigor | **Good** | Proper IEEE citations, structured summaries |
| Coverage Breadth | **Good** | Covers ELA, CNN, U-Net, traditional CV, and hashing methods |
| Practical Relevance | **High** | Directly relevant to the project's approach (ELA + U-Net) |
| Formatting | **Fair** | Markdown with some rendering issues from base64 images |
| Completeness | **Good** | 25 references covering the core forensics literature |

---

## Strengths

1. **Academically grounded** — proper IEEE paper citations with publication IDs
2. **Broad coverage** — spans from traditional CV (Zernike moments) to deep learning (U-Net, EfficientNet)
3. **Directly relevant** — papers 1 and 4 use the same ELA + CNN/U-Net approach as this project
4. **Benchmark context** — provides external accuracy targets for comparison
5. **Open access notes** — identifies which papers are freely available

---

## Weaknesses

1. **No critical analysis** — papers are summarized but not critiqued (e.g., fairness of metric comparisons, dataset overlaps)
2. **Mixed metric types** — some papers report accuracy, others F1, making comparisons misleading
3. **No segmentation metrics** — most papers only report classification accuracy, not IoU/Dice for localization
4. **No discussion of CASIA dataset issues** — known label noise, duplicate images, and version inconsistencies in CASIA 2.0 are not mentioned
5. **Base64 images bloat file size** — 38KB for ~194 lines suggests heavy embedded image content

---

## Critical Issues

1. **Misleading benchmark comparison.** The 98.44% accuracy (Zernike + ACO on MICC-F220) and 87.75% (ELA + CNN on CASIA2) are not directly comparable — they use different datasets, different tasks (copy-move vs general tampering), and different evaluation protocols. Presenting them in the same table without caveats suggests false equivalence.

2. **No mention of CASIA 2.0 dataset quality issues.** The CASIA 2.0 dataset has known problems:
   - Many tampered images have misaligned masks
   - Some authentic/tampered labels are incorrect
   - Multiple versions exist with different image counts (5,123 vs 7,354 vs 12,614)
   These issues directly affect the project's results and should be discussed.

---

## Suggested Improvements

1. Add a critical comparison section noting dataset and metric differences between papers
2. Include discussion of CASIA 2.0 dataset limitations
3. Add a "Relevance to this Project" column explaining which techniques were adopted
4. Replace base64 embedded images with proper figure references
5. Add a section mapping each paper's innovations to the project's vK.11.x architecture

---

## Roast Section

This document does what a literature review should do — survey the landscape and cite the sources — but stops short of what a _useful_ literature review does: critically analyze whether the reported results are comparable, identify which techniques are practically applicable, and flag the known weaknesses of the benchmarks.

The benchmark table is the core problem. It puts "98.44% accuracy on MICC-F220" (a tiny copy-move dataset) next to "87.75% accuracy on CASIA2" (a large mixed-forgery dataset) and "0.96 F1 on a validation set" (unknown dataset, unknown split). These numbers look like a leaderboard but they're actually apples, oranges, and a mystery fruit from an unspecified validation set. A reader could walk away thinking 98.44% is achievable — when that result is specific to copy-move detection on 220 images with Zernike moments, not general-purpose tampering detection.

The document is genuinely useful as a starting point for understanding the field. It correctly identifies the key approaches (ELA, CNN classification, U-Net segmentation, frequency-domain analysis) and provides IEEE references for deeper reading. But it would be 10× more useful with a paragraph per paper explaining: "Here's why this result doesn't directly apply to our project" or "Here's the specific technique we adopted and adapted."

**Bottom line:** A solid literature survey that provides academic context. Its primary value to the project is confirming that the ELA + U-Net approach (papers 1 and 4) is well-established in the literature, and that pretrained encoders (EfficientNetV2B0, MobileNetV2) consistently outperform training from scratch — which is exactly what the project's own experiments confirmed.
