# Project Experiment Evolution Report

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Scope** | Complete project history from vK.1 through vK.12.0b |
| **Total Experiment Runs Analyzed** | 22 (16 prior + 6 new vK.11.x/12.0x) |
| **Project** | Tampered Image Detection & Localization (Big Vision Internship Assignment) |

---

## 1. Executive Summary

This report traces the complete evolution of the Tampered Image Detection & Localization project across 22 experiment runs, 5 documentation eras, and 3 distinct architecture tracks. The project has been in development through multiple iteration cycles, producing an extensive record of architectural decisions, training experiments, and evaluation refinements.

**The bottom line: v6.5 remains the best-performing run after 13 additional experiment versions.**

The project's trajectory can be summarized in three phases:

1. **Documentation Era (v1-v5)**: Architecture designed on paper, progressively refined through 5 audit cycles. No training executed. Score improved from 6/10 to 8.8/10.

2. **Execution Era (vK.1-vK.10.6)**: First training runs. Custom from-scratch architectures. Data leakage discovered and eventually fixed. Best from-scratch result: vK.10.6 (Tam-F1=0.22). Best overall result: v6.5 (Tam-F1=0.41) using pretrained ResNet34.

3. **Synthesis Era (vK.11.x)**: Attempt to combine all prior learnings into one architecture. **Catastrophic failure** -- Tam-F1=0.13, worse than every prior run including from-scratch models.

The project demonstrates excellent engineering maturity growth (CONFIG systems, evaluation suites, reproducibility) but has been unable to improve upon a 2-GPU, 25-epoch run from the early execution phase.

---

## 2. Complete Run Timeline

### Phase 0: Documentation Only (No Executed Notebooks)

| # | Version | Audit Score | Key Advancement | Status |
|---|---------|------------|-----------------|--------|
| — | Docs v1 | 6.0/10 | Initial architecture design (UNet + ResNet34) | Hard-coded bugs, API mismatches |
| — | Docs v2 | 8.0/10 | Fixed counts, split procedure, model API | Image scoring not locked |
| — | Docs v3 | 8.0/10 | Pipeline locked: smp.Unet, BCEDice, AdamW, AMP | ELA channel-count conflict |
| — | Docs v4 | 8.5/10 | ELA conflict resolved, W&B guarded, v4 notebook | Minor artifact drift |
| — | Docs v5 | 8.8/10 | No critical blockers, design ready for training | CASIA leakage risk noted |

### Phase 1: First Training Runs

| # | Version | Architecture | Params | Img Acc | AUC | Tam-F1 | Key Finding |
|---|---------|-------------|--------|---------|-----|--------|-------------|
| 1 | vK.1 | Custom UNet (scratch) | 15.7M | 0.8986 | — | ~0.20* | Block 1 data leakage |
| 2 | vK.2 | Custom UNet (scratch) | 15.7M | 0.8986 | — | ~0.20* | Docs only, same pipeline |
| 3 | vK.3 | Custom UNet (scratch) | 15.7M | 0.8986 | — | ~0.20* | Dice inflation (auth=1.0) |
| 4 | vK.3-run-01 | Custom UNet (scratch) | 15.7M | 0.8986 | — | ~0.20* | Same model, Kaggle run |

*Estimated; tampered-only metrics not reported. Mixed Dice=0.5761 inflated by authentic images.

### Phase 2: Pretrained Encoder Era (BEST RESULTS)

| # | Version | Architecture | Params | Img Acc | AUC | Tam-F1 | Key Finding |
|---|---------|-------------|--------|---------|-----|--------|-------------|
| 5 | **v6.5** | **SMP UNet + ResNet34 (pretrained)** | **24.4M** | **0.8246** | **0.8703** | **0.4101** | **PROJECT BEST -- pretrained encoder proven critical** |
| 6 | v8 | SMP UNet + ResNet34 (pretrained) | 24.4M | 0.7190 | 0.8170 | 0.2949 | **REGRESSION** -- pos_weight=30.01 bug, 16x batch without LR scaling |

### Phase 3: Custom Architecture Revival

| # | Version | Architecture | Params | Img Acc | AUC | Tam-F1 | Key Finding |
|---|---------|-------------|--------|---------|-----|--------|-------------|
| 7 | vK.7.1 | Custom UNet (scratch) | 15.7M | 0.8986 | — | ~0.20* | Block 1 leak still present |
| 8 | vK.7.5 | Custom UNet (scratch) | 15.7M | — | — | — | Incomplete run (W&B crash) |
| 9 | vK.10.3b | Custom UNet (scratch) | 31.6M | 0.5061 | 0.6069 | 0.0004 | **COLLAPSED** -- patience=10 killed training |
| 10 | vK.10.3b-r02 | Custom UNet (scratch) | 31.6M | 0.5061 | 0.6069 | 0.0004 | Exact duplicate of run-01 |
| 11 | vK.10.3b-r03 | Custom UNet (scratch) | 31.6M | — | — | 0.2196 | Extended training: 100 epochs, patience=50 |
| 12 | vK.10.4 | Custom UNet (scratch) | 31.6M | 0.4675 | 0.6534 | 0.0000 | **COLLAPSED** -- patience=10 |
| 13 | vK.10.5 | Custom UNet (scratch) | 31.6M | 0.4791 | 0.6201 | 0.0006 | **COLLAPSED** -- patience=10, DataParallel added |
| 14 | **vK.10.6** | **Custom UNet (scratch)** | **31.6M** | **0.8357** | **0.9057** | **0.2213** | **Best from-scratch** -- 100 epochs, patience=30, best eval suite |

### Phase 4: Synthesis Architecture (NEW -- This Audit)

| # | Version | Architecture | Params | Img Acc | AUC | Tam-F1 | Key Finding |
|---|---------|-------------|--------|---------|-----|--------|-------------|
| 15 | vK.11.1 | Synthesis (SMP+ELA+Edge+CLS) | 24.5M | — | — | — | **Unexecuted** (edge loss AMP bug) |
| 16 | **vK.11.4** | **Synthesis (SMP+ELA+Edge+CLS)** | **24.5M** | **0.4142** | **0.6434** | **0.1321** | **FAILED** -- constant output, pixel-AUC=0.50 |
| 17 | **vK.11.5** | **Synthesis (SMP+ELA+Edge+CLS)** | **24.5M** | **0.4194** | **0.6466** | **0.1272** | **FAILED** -- worse than 11.4, epoch 3 peak |
| 18 | **vK.12.0** | **Synthesis (SMP+ELA+Edge+CLS)** | **24.5M** | **0.4062** | **0.5637** | **0.1321** | **FAILED + CRASHED** -- `KeyError: 'true_mask'`, 42 cells blocked |
| 19 | **vK.11.1-R2** | **Synthesis (SMP+ELA+Edge+CLS)** | **24.5M** | **0.5235** | **0.6550** | **0.1274** | **FAILED** -- worst pixel-AUC (0.4482), best cls metrics, online W&B |
| 20 | **vK.12.0b** | **Synthesis (SMP+ELA+Edge+CLS)** | **24.5M** | **0.4062** | **0.6175** | **0.1322** | **FAILED + CRASHED** -- `AttributeError`, 16 cells blocked |

---

## 3. Architecture Track Analysis

The project explored three distinct architecture families:

### Track A: Custom UNet from Scratch

| Versions | Best Tam-F1 | Approach | Verdict |
|----------|------------|----------|---------|
| vK.1-vK.3, vK.7.1, vK.7.5, vK.10.x | 0.2213 (vK.10.6) | Custom encoder/decoder, no pretraining, 256x256 | **Viable with enough training time** (100 epochs). Ceiling limited by lack of pretrained features. |

Key learnings:
- Patience=10 is fatal for from-scratch training (vK.10.3b-10.5 collapsed)
- 100 epochs with patience=30 is minimum viable (vK.10.6, vK.10.3b-r03)
- 31.6M parameters from scratch on 8,829 images requires careful regularization

### Track B: SMP Pretrained ResNet34

| Versions | Best Tam-F1 | Approach | Verdict |
|----------|------------|----------|---------|
| v6.5, v8 | **0.4101** (v6.5) | SMP UNet, ImageNet pretrained, 384x384, BCEDice | **Best track -- nearly 2x better than from-scratch** |

Key learnings:
- Pretrained encoder is the single most impactful architectural choice (+85% Tam-F1 over from-scratch)
- v6.5's simplicity was its strength: BCEDice, differential LR, no fancy components
- v8 proved that poorly-tuned hyperparameters (pos_weight=30, 16x batch) can destroy a good architecture

### Track C: Synthesis Architecture

| Versions | Best Tam-F1 | Approach | Verdict |
|----------|------------|----------|---------|
| vK.11.1-vK.12.0b | 0.1321 (vK.11.4/12.0) | SMP + ELA + EdgeLoss + CLS head, 256x256 | **FAILED -- worst pretrained-encoder result. 6 runs (5 executed), same constant output** |

Key learnings:
- Combining 5 new components simultaneously produced catastrophic interference
- Multi-objective loss (Focal + BCE + Dice + Edge) creates un-navigable optimization landscape
- Classification head competes with segmentation for encoder features (proven by vK.11.1-R2: best cls metrics, worst pixel-AUC)
- ELA channel value unvalidated on CASIA dataset
- Five independent training runs converge to bitwise identical stratified metrics -- the constant-output attractor is universal

---

## 4. Performance Progression Chart

```
Tampered-Only F1 Score Across All Executed Runs
(higher is better, 1.0 = perfect)

v6.5    ████████████████████████████████████████ 0.4101  <- PROJECT BEST (pretrained ResNet34)
v8      ████████████████████████████▌            0.2949  <- regression (pos_weight bug)
vK.10.6 ██████████████████████▏                  0.2213  <- best from-scratch (100 epochs)
10.3b³  █████████████████████▊                   0.2196  <- extended training recovery
vK.3    ████████████████████                    ~0.20    <- estimated (data leak)
vK.7.1  ████████████████████                    ~0.20    <- estimated (data leak)
vK.11.4 █████████████▏                           0.1321  <- SYNTHESIS FAILURE
vK.12.0 █████████████▏                           0.1321  <- SYNTHESIS FAILURE + CRASH
vK.12.0b█████████████▏                           0.1322  <- SYNTHESIS FAILURE + CRASH
11.1-R2 ████████████▊                            0.1274  <- SYNTHESIS FAILURE (worst pixel-AUC)
vK.11.5 ████████████▋                            0.1272  <- SYNTHESIS FAILURE (worst Tam-F1)
vK.10.5 ▏                                        0.0006  <- collapsed (patience=10)
vK.10.3b▏                                        0.0004  <- collapsed (patience=10)
vK.10.4                                          0.0000  <- collapsed (patience=10)
```

```
Image-Level AUC-ROC Across All Executed Runs
(higher is better, 1.0 = perfect)

vK.10.6 █████████████████████████████████████████████ 0.9057  ← BEST CLASSIFICATION
v6.5    ███████████████████████████████████████████▌   0.8703
v8      ████████████████████████████████████████▋      0.8170
vK.11.5 ████████████████████████████████▎              0.6466
vK.11.4 ████████████████████████████████▏              0.6434
vK.10.4 ████████████████████████████████▋              0.6534
vK.10.3b████████████████████████████████               0.6069
vK.10.5 ███████████████████████████████                0.6201
```

---

## 5. What Went Wrong with the Synthesis Architecture

The vK.11 series was supposed to be the culmination -- combining every lesson from v6.5, v8, and vK.10.6 into a single architecture. Instead, it produced the worst pretrained-encoder results in project history. Here is the analysis:

### 5.1 The v6.5 Baseline

v6.5 achieved Tam-F1=0.41 with:
- SMP UNet + ResNet34 pretrained
- **3-channel input** (RGB only)
- **BCEDice loss** (single segmentation objective)
- **No classification head**
- **No edge loss**
- 384x384 resolution
- Differential LR (enc=1e-4, dec=1e-3)
- 25 epochs, early stopped at 15

### 5.2 What vK.11 Changed

| Component | v6.5 | vK.11.4 | Impact |
|-----------|------|---------|--------|
| Input channels | 3 (RGB) | **4 (RGB+ELA)** | Requires first conv adaptation, unknown ELA signal quality |
| Loss function | BCEDice | **1.5*Focal + BCEDice + 0.3*Edge** | Three competing objectives |
| Classification head | None | **FC 512→256→2** | Encoder gradient contamination |
| Resolution | 384x384 | **256x256** | 44% fewer pixels per image |
| Loss weighting | 1.0 (seg only) | **1.5 cls + 1.0 seg + 0.3 edge** | Classification dominates |
| Max epochs | 25 | **50** | Adequate |
| Image-level output | None (seg only) | **Softmax 2-class** | New task competing for capacity |

### 5.3 Root Cause Breakdown

**Every change from v6.5 either added complexity or removed resolution, and NO change was validated independently.**

1. **Multi-objective loss conflict**: The 1.5x classification weight means the encoder receives stronger gradient signal from the classification task than from segmentation. The encoder optimizes for "is this image tampered?" at the expense of "where is the tampering?"

2. **ELA channel noise injection**: CASIA stores images as JPEG. Repeated JPEG compression produces ELA artifacts even in authentic regions, potentially adding noise rather than signal. The 4th channel was never validated independently.

3. **Resolution reduction**: 384→256 loses 44% of spatial information. For pixel-level localization of often-subtle tampering, this matters.

4. **Encoder unfreeze destruction**: vK.11.5's epoch-3 peak is direct evidence. The pretrained features are being corrupted upon unfreeze, likely because three loss objectives pull the encoder in conflicting directions.

### 5.4 The Lesson

**Combining five good ideas does not produce a good result.** Each of ELA, edge loss, and dual-task classification may individually improve v6.5. But adding them simultaneously created an optimization landscape where the model cannot find a useful gradient path. The project needed ablation studies, not a synthesis experiment.

---

## 6. Evaluation Methodology Evolution

One area of consistent improvement across the project is the evaluation suite:

| Era | Metrics Available | Key Addition |
|-----|------------------|--------------|
| vK.1-vK.3 | Accuracy, mixed Dice only | None -- metrics inflated by authentic images |
| v6.5 | + Tampered-only metrics, threshold sweep, forgery-type breakdown, Grad-CAM, robustness | **First comprehensive eval** |
| v8 | + Per-sample Dice, mask-size stratification | Better small-mask handling |
| vK.10.6 | + Pixel-AUC, confusion matrix, ROC/PR curves, shortcut detection, failure analysis | **Most rigorous eval suite** |
| vK.11.4/11.5 | Same as vK.10.6 + W&B prediction logging | No new eval features |

### The Irony

The project's most thorough evaluation machinery (vK.11.4/11.5) now definitively demonstrates that the model it evaluates is non-functional. The shortcut test, robustness test, and pixel-AUC were designed to validate a working model -- instead, they proved the model learned nothing. The evaluation suite works perfectly; the model does not.

### Key Evaluation Milestones

| Milestone | Version | Impact |
|-----------|---------|--------|
| Tampered-only metrics | v6.5 | Eliminated authentic-inflated Dice (0.58 → 0.41 true score) |
| Shortcut learning detection | vK.10.6 | First falsification test for genuine learning |
| Pixel-level AUC | vK.10.6 | Independent measure of pixel discriminative power |
| Per-forgery-type analysis | v6.5 | Revealed copy-move near-failure (F1=0.31) |
| Mask-size stratification | v8 | Revealed tiny-mask blindness |

---

## 7. Five Fatal Flaws -- Project-Wide

These systematic errors have recurred across the project's history:

### Flaw 1: Architectural Regression

The project repeatedly abandons proven approaches:
- v6.5 proved pretrained ResNet34 is critical → vK.7.1-vK.10.5 used from-scratch models
- vK.10.6 proved 100 epochs are needed → vK.11.4/11.5 used 50 epochs
- v6.5's simple BCEDice worked → vK.11 added 3 competing loss objectives

### Flaw 2: Metric Inflation

Mixed-set Dice/F1 inflated by 59.4% authentic images scoring 1.0. Only versions from v6.5 onward report honest tampered-only metrics. vK.1-vK.3 reported Dice=0.5761 which masked true performance of ~0.20.

### Flaw 3: Block 1 Data Leakage

vK.1, vK.2, vK.3, vK.7.1, and vK.7.5 trained on the test set in Block 1 (`TRAIN_CSV = test_metadata.csv`). This inflated accuracy to 0.8986 -- a number that includes test leakage. Fixed in vK.10.x via single-block structure.

### Flaw 4: Premature Early Stopping

| Patience | Versions | Outcome |
|----------|----------|---------|
| 10 | vK.10.3b, 10.4, 10.5, vK.11.4, vK.11.5 | Collapsed or failed |
| 20 | vK.11.1 | Never executed |
| 30 | vK.10.6 | Best from-scratch result |
| 50 | vK.10.3b-r03 | Recovered from collapse |

**Every run with patience=10 either collapsed (vK.10.x) or failed (vK.11.x).** The evidence is overwhelming: patience=10 is insufficient.

### Flaw 5: No Ablation Studies

The project has NEVER run a controlled ablation. Every version changes multiple variables simultaneously:
- v6.5→v8: changed loss, batch size, augmentations, pos_weight, scheduler
- vK.10.5→vK.10.6: changed epochs, patience, scheduler
- v6.5→vK.11.4: changed input channels, loss function, added classification head, reduced resolution

Without ablation, it is impossible to determine which changes help and which hurt.

---

## 8. Engineering Maturity Progression

Despite the performance plateau, engineering quality has steadily improved:

| Feature | First Appeared | Current State |
|---------|---------------|---------------|
| CONFIG centralization | vK.10.3b | Standard across all vK.10+ |
| Seed reproducibility | vK.10.3b | torch + numpy + random + CUDA + cuDNN |
| AMP (mixed precision) | v6.5 | Standard with proper edge loss casting |
| Gradient accumulation | v6.5 | With partial-window flush |
| DataParallel (multi-GPU) | v6.5 | Standard |
| VRAM auto-scaling | vK.10.4 | Batch size adapts to GPU count |
| Checkpoint system | vK.10.3b | 3-file (best/last/periodic) |
| Data leakage verification | vK.10.3b | Explicit cross-set overlap check |
| W&B integration | vK.11.4 | Offline mode on Kaggle, prediction logging |
| Reproducibility section | vK.11.4 | Seed verification, split determinism, environment info |
| Results Dashboard | vK.11.5 | Concept good, implementation broken |

**The project's engineering infrastructure is production-quality.** The gap between infrastructure maturity and model performance is the project's central paradox.

---

## 9. Assignment Requirements Status (Final)

| Requirement | Best Version | Status | Notes |
|-------------|-------------|--------|-------|
| **Dataset Selection & Preparation** | vK.11.4 | **PASS** | CASIA v2.0, proper splits, ELA, augmentation |
| **Model Architecture** | vK.11.4 | **PASS** | TamperDetector documented and implemented |
| **Resource Constraints** | v6.5 | **PASS** | Runs on T4 GPUs, AMP enabled |
| **Performance Metrics** | vK.10.6 | **PARTIAL** | Comprehensive eval, but best Tam-F1=0.41 (not competitive) |
| **Visual Results** | v6.5 | **PARTIAL** | Best visualizations in v6.5 (Grad-CAM, predictions, overlays) |
| **Single Notebook** | All | **PASS** | All versions are single .ipynb files |
| **Trained Weights** | v6.5 | **PARTIAL** | Weights exist but model performance is below competitive |
| **Bonus: Robustness** | vK.10.6 | **PARTIAL** | Suite exists, reveals weaknesses (blur -66% for vK.10.6) |
| **Bonus: Subtle Tampering** | v6.5 | **PARTIAL** | Copy-move F1=0.31 (v6.5), splicing F1=0.59 (v6.5) |

**Overall Assignment Rating: PARTIAL COMPLIANCE**

The project demonstrates strong engineering discipline and comprehensive evaluation methodology. The architecture choices are reasonable and well-documented. However, the model performance (best Tam-F1=0.41) is below competitive benchmarks for CASIA-2 (0.65+), and the most recent runs (vK.11.x) represent a significant regression.

---

## 10. Lessons Learned (Complete)

### What Worked

| Decision | Evidence | First Appeared |
|----------|----------|---------------|
| Pretrained encoder (ResNet34) | +85% Tam-F1 over from-scratch (0.41 vs 0.22) | v6.5 |
| Differential LR (enc < dec) | Standard practice, confirmed by v6.5 | v6.5 |
| Per-sample Dice loss | Better small-mask handling vs batch-level | v8 |
| Comprehensive evaluation suite | Reveals true model capability | vK.10.6 |
| Extended training (100 epochs) | Recovery from collapse (0.0004 → 0.22) | vK.10.6 |
| Data leakage verification | Caught Block 1 leak | vK.10.3b |
| AMP with gradient scaling | 2x throughput, no accuracy loss | v6.5 |

### What Failed

| Decision | Evidence | Lesson |
|----------|----------|--------|
| pos_weight=30 (v8) | -28% Tam-F1 regression | Compute from tampered pixels only |
| From-scratch with patience=10 | Three consecutive collapses (vK.10.3b-10.5) | Minimum patience=20 for from-scratch |
| Synthesis architecture (vK.11) | Worst pretrained-encoder result (0.13) | Never add 5 components simultaneously |
| 16x batch without LR scaling (v8) | -28% regression | Scale LR with batch |
| Reduced resolution 384→256 (vK.11) | Performance degraded | Maintain v6.5's 384x384 |
| Heavy classification loss weight (vK.11) | Encoder gradient contamination | Seg loss should dominate |

### What Was Never Tested

| Potential Improvement | Status | Priority |
|----------------------|--------|----------|
| ELA channel (isolated) | Added in vK.11 but never tested alone | P0 |
| Edge loss (isolated) | Added in vK.11 but never tested alone | P0 |
| Classification head (isolated) | Added in vK.11 but never tested alone | P0 |
| Attention mechanisms (SE, CBAM) | Never implemented | P1 |
| Larger encoders (ResNet50, EfficientNet) | Never tested | P1 |
| Test-time augmentation | Never implemented | P1 |
| Cross-dataset validation | Never implemented | P2 |
| Boundary-aware losses (independent) | Never tested alone | P2 |

---

## 11. Recommendations for vK.12+

### Priority 0: Establish a Reproducible Baseline

**Run v6.5's EXACT architecture again** with vK.10.6's evaluation suite.

This serves two purposes:
1. Verify that v6.5's Tam-F1=0.41 is reproducible (it may have been a lucky run)
2. Get pixel-AUC, shortcut test, and robustness results for v6.5's architecture (these were not available in the original v6.5 run)

Configuration: SMP UNet, ResNet34 pretrained, 3-channel RGB, BCEDice loss, 384x384, enc_lr=1e-4, dec_lr=1e-3, batch=4 (eff 16), 25 epochs, patience=10.

### Priority 1: Controlled Ablation Study

Add ONE component at a time to the v6.5 baseline:

| Run | Change from Baseline | Expected Impact | Purpose |
|-----|---------------------|-----------------|---------|
| A | + ELA 4th channel (4ch input) | Unknown | Validate ELA signal on CASIA |
| B | + Edge loss (0.3 weight) | Small positive | Test boundary supervision |
| C | + Classification head (0.5 weight) | Unknown | Test dual-task learning |
| D | Reduce enc_lr to 1e-5 | Positive | Prevent encoder feature corruption |
| E | 256x256 instead of 384x384 | Negative | Quantify resolution impact |

### Priority 2: Fix Known Issues

| Issue | Fix |
|-------|-----|
| Classification loss weight too high (1.5x) | Reduce to 0.3-0.5x if keeping CLS head |
| Encoder LR too high for fine-tuning | Use 1e-5 or 1e-6 |
| Training budget too low (50 epochs, patience=10) | Restore 100 epochs, patience=20-30 |
| ELA channel not validated | Profile ELA output on CASIA images (check signal-to-noise) |
| Albumentations API deprecated | Update to current API for robustness transforms |
| Results Dashboard execution order | Move after training, or use conditional rendering |

### Priority 3: Push Beyond v6.5

Only attempt after ablation study is complete:
- Combine components that individually demonstrated improvement
- Consider larger encoder (ResNet50) if T4 memory allows
- Add test-time augmentation (flip + multi-scale)
- Explore attention mechanisms (SE blocks, CBAM)
- Consider boundary F1 as additional metric

---

## 12. Project Timeline

```
Timeline of Major Milestones
═══════════════════════════════════════════════════════════════════

DOCUMENTATION ERA
│
├─ Docs v1 ────── Initial design (6/10)
├─ Docs v2 ────── Fixed API mismatches (8/10)
├─ Docs v3 ────── Pipeline locked (8/10)
├─ Docs v4 ────── ELA resolved (8.5/10)
└─ Docs v5 ────── Ready for training (8.8/10)

FIRST RUNS
│
├─ vK.1-vK.3 ─── Custom UNet from scratch, Block 1 data leak
│                 Tam-F1 ~0.20 (inflated by leak)
│
├─ v6.5 ◀════════ PROJECT BEST: Tam-F1 = 0.4101
│                 Pretrained ResNet34, BCEDice, 384x384
│                 First comprehensive evaluation suite
│
├─ v8 ─────────── REGRESSION: Tam-F1 = 0.2949
│                 pos_weight=30 bug, 16x batch without LR scaling
│
├─ vK.7.1 ─────── Back to scratch UNet, Block 1 leak still present
├─ vK.7.5 ─────── Incomplete run (W&B crash)
│
├─ vK.10.3b ───── COLLAPSED: Tam-F1 = 0.0004 (patience=10)
├─ vK.10.4 ────── COLLAPSED: Tam-F1 = 0.0000 (patience=10)
├─ vK.10.5 ────── COLLAPSED: Tam-F1 = 0.0006 (patience=10)
├─ vK.10.3b-r03 ─ RECOVERED: Tam-F1 = 0.2196 (patience=50)
│
├─ vK.10.6 ◀═════ BEST FROM-SCRATCH: Tam-F1 = 0.2213
│                 100 epochs, patience=30, best eval suite
│                 AUC-ROC = 0.9057 (best classification)

SYNTHESIS ERA
│
├─ vK.11.1 ────── Synthesis design (unexecuted, AMP bug)
├─ vK.11.4 ────── FAILED: Tam-F1 = 0.1321, pixel-AUC = 0.50
└─ vK.11.5 ────── FAILED: Tam-F1 = 0.1272, pixel-AUC = 0.52
                   Both models output constant predictions
                   Shortcut test: zero correlation with image content

NEXT ─── vK.12+:  Ablation study needed
```

---

## 13. What Improved Over Time

| Dimension | Early (vK.1-vK.3) | Middle (v6.5-vK.10.6) | Late (vK.11.x) |
|-----------|-------------------|----------------------|-----------------|
| **Data integrity** | Block 1 leak, no verification | Leak fixed (vK.10.x), verification added | Full leakage check, stratified splits |
| **Architecture** | Custom 15.7M scratch | Pretrained 24.4M (v6.5) or Custom 31.6M (vK.10) | Synthesis 24.5M pretrained + ELA |
| **Training rigor** | No AMP, no scheduling, no early stopping | AMP, differential LR, ReduceLROnPlateau | + Gradient accumulation, encoder freeze |
| **Evaluation** | Mixed Dice only | + Tampered-only, threshold sweep, Grad-CAM, robustness | + Pixel-AUC, shortcut detection, failure analysis |
| **Reproducibility** | No seeds, no checkpoints | Seeds, 3-file checkpoints | + Reproducibility verification section |
| **Documentation** | Minimal markdown | Architecture diagrams, model cards | + Executive summary, results dashboard |
| **Performance** | ~0.20 (leaked) | **0.41** (v6.5, BEST) | 0.13 (WORST pretrained) |

---

## 14. Mistakes Repeated

| Mistake | First Occurrence | Repeated In | Impact |
|---------|-----------------|-------------|--------|
| Insufficient training time | vK.10.3b (patience=10) | vK.11.4/11.5 (patience=10) | Collapse/failure in both cases |
| Multiple simultaneous changes | v6.5→v8 (4 changes) | v6.5→vK.11 (6 changes) | Regression in both cases |
| No isolated ablation | Every version | Every version | Cannot attribute improvements/regressions |
| Abandoning proven approach | vK.7 abandoned v6.5's pretrained encoder | vK.11 added untested components to v6.5 | Performance never improved beyond v6.5 |

---

## 15. Final Verdict

The Tampered Image Detection & Localization project tells a cautionary tale about ML experimentation without disciplined ablation.

**The Good:**
- Engineering maturity grew consistently from vK.1 to vK.11.5
- The evaluation suite (vK.10.6+) is genuinely rigorous -- shortcut detection, pixel-AUC, forgery-type breakdown, robustness testing
- The CONFIG system, reproducibility infrastructure, and checkpoint strategy are production-quality
- v6.5 proved that pretrained encoders are the right approach

**The Bad:**
- v6.5 (run 5 of 19) remains the best result after 14 additional experiments
- The project's most ambitious experiment (vK.11 synthesis) produced its worst results
- No controlled ablation has ever been performed
- The same mistakes (insufficient patience, multiple simultaneous changes) keep recurring

**The Path Forward:**
- Stop adding features. Start validating them individually.
- v6.5's architecture is the proven baseline. Everything else is hypothesis.
- Each new component (ELA, edge loss, classification head) needs its own isolated experiment before combination.
- The project needs fewer ambitious designs and more disciplined experimentation.

**Overall Project Rating: 5.5/10**

Strong engineering, comprehensive evaluation, well-documented architecture -- built around a model that has never exceeded 0.41 Tam-F1 and currently sits at 0.13. The infrastructure is ready for a breakthrough; the experimentation methodology is not.
