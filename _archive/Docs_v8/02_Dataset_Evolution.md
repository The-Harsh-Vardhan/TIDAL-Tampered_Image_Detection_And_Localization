# 02 — Dataset Evolution

## Purpose

Trace how dataset handling evolved from Docs7 design → Audit critique → Run01 evidence, and define improvements for v8.

---

## Phase 1: Docs7 Design

Docs7 specified:

- **CASIA v2.0** from Kaggle, selected for convenience (image–mask pairs, public, structured)
- **12,614 total pairs:** 7,491 authentic + 3,295 copy-move + 1,828 splicing
- **Mask binarization:** `mask > 0` threshold, converting grayscale to binary
- **Resizing:** All images and masks to 384×384 (bilinear for images, nearest for masks)
- **Splits:** Stratified 70/15/15 by `forgery_type` (authentic, splicing, copy-move)
- **Augmentation:** HorizontalFlip, VerticalFlip, RandomRotate90 only
- **Leakage check:** 0 path overlaps verified across splits
- **Excluded augmentations:** JPEG compression, noise injection, blur — rationale was to avoid destroying forensic cues during training

## Phase 2: Audit Critique

### Critical Findings

| Finding | Severity | Source |
|---|---|---|
| CASIA is a 2013 legacy benchmark — no modern manipulation types | HIGH | Audit6 Pro §01 Finding 3 |
| Path disjointness ≠ content disjointness; no perceptual hash check | HIGH | Audit6 Pro §01 Finding 5 |
| Stratifying only by forgery type ignores mask size, difficulty, scene bias | MEDIUM | Audit6 Pro §01 Finding 6 |
| 384×384 resizing may erase sub-pixel forensic traces | MEDIUM | Audit6 Pro §01 Finding 8 |
| Annotation quality not measured — noisy masks accepted as ground truth | MEDIUM | Audit6 Pro §01 Finding 7 |
| Minimal augmentation leaves model vulnerable to overfitting | HIGH | Audit6 Pro §02 Finding 7 |

### The Augmentation Dilemma

Audit6 Pro identified a trap: the project excluded augmentations to "preserve forensic cues," but this left the model with insufficient regularization for a small dataset. The result is either overfitting (which Run01 confirmed) or underfitting because the safe augmentation space is too narrow.

## Phase 3: Run01 Evidence

### What the data actually showed

| Observation | Implication |
|---|---|
| 0 file exclusions post-cleaning | Pipeline is robust — no corrupt/broken pairs |
| Copy-move F1=0.31, splicing F1=0.59 | Architecture or loss cannot handle copy-move's subtlety |
| 8/10 worst failures are copy-move | Clear systematic failure mode per forgery type |
| 6/10 worst failures have mask area <2% | Small tampered regions are disproportionately hard |
| Overfitting onset at epoch 15 | Insufficient augmentation/regularization confirmed |
| ~13% F1 drop under JPEG compression | Model partially relies on compression artifacts |
| 4 degradation conditions produce identical F1≈0.593 | Suggests model collapses to baseline under any distribution shift |
| Threshold=0.1327 | Background pixel dominance suppresses positive predictions |

### Copy-Move vs Splicing Breakdown

| Metric | Splicing (274) | Copy-Move (495) |
|---|---|---|
| F1 | 0.5901 ± 0.3850 | 0.3105 ± 0.3968 |
| % of test tampered | 36% | 64% |
| In worst-10 failures | 2/10 | 8/10 |

Copy-move is both the majority class and the worst-performing class, meaning it drags down all aggregate metrics.

---

## v8 Dataset Improvements

### P0: Must Fix

**1. Add augmentation to combat overfitting**
```python
# Add to train_transform:
A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3)
A.GaussNoise(var_limit=(10, 50), p=0.3)
A.GaussianBlur(blur_limit=(3, 5), p=0.2)
```
**Rationale:** Training with JPEG compression augmentation should reduce the 13% robustness gap and address shortcut learning. Noise and blur augmentations will force the model to learn structural rather than artifact-based features.

**2. Compute and use BCE pos_weight from training masks**

Calculate the foreground-to-background pixel ratio across training masks and pass as `pos_weight` to `BCEWithLogitsLoss`. This directly addresses the threshold=0.1327 anomaly.

### P1: Important

**3. Add mask-area stratification to evaluation**
```python
# Bucket masks by tampered area percentage
size_buckets = {'tiny': (0, 0.02), 'small': (0.02, 0.05), 
                'medium': (0.05, 0.15), 'large': (0.15, 1.0)}
```
Report F1 per bucket. Run01 showed small-region failure — quantifying this informs loss/architecture changes.

**4. Run perceptual hash near-duplicate check**

Use pHash or CLIP embeddings to identify potential content leakage across splits. If near-duplicates exist, group them into the same split.

### P2: Moderate

**5. Track per-forgery-type training loss**

Log separate BCE+Dice loss for copy-move and splicing batches to detect whether one type is harder to fit.

**6. Consider mask quality audit**

Manually review 50 random masks to estimate annotation noise. Report findings as a caveat on pixel-level metrics.

---

## What NOT to Change

- **Keep CASIA.** It is the assignment's expected dataset. Replacing it would introduce risk without assignment justification.
- **Keep 384×384.** Higher resolution would exceed T4 memory with the current architecture. Forensic signal loss is a known tradeoff, documented.
- **Keep stratified 70/15/15.** The split ratios are standard and sufficient for the dataset size.
- **Keep mask binarization threshold at 0.** Binary prediction is the stated task. Soft masks are a future experiment.
