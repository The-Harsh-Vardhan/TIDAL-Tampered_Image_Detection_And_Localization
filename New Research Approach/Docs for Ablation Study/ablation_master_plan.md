# ETASR Ablation Study — Master Plan

| Field | Value |
|-------|-------|
| **Date** | 2026-03-14 |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Dataset** | CASIA v2.0 (12,614 images: 7,491 Au + 5,123 Tp) |
| **Platform** | Kaggle (T4 / P100 GPU) |
| **Methodology** | Single-variable ablation study — one change per version |

---

## 1. Baseline Analysis (vR.1.0)

### Configuration Frozen as Baseline

```
PREPROCESSING
  ELA quality:       90
  Image size:        128×128
  Normalization:     [0, 1]
  Augmentation:      None
  Input channels:    3 (ELA RGB)

ARCHITECTURE
  Conv2D(32, 5×5, valid, ReLU)
  Conv2D(32, 5×5, valid, ReLU)
  MaxPooling2D(2×2)
  Dropout(0.25)
  Flatten
  Dense(256, ReLU)
  Dropout(0.5)
  Dense(2, Softmax)
  Total params:      29,520,034

TRAINING
  Optimizer:         Adam
  Learning rate:     0.0001
  Loss:              categorical_crossentropy
  Batch size:        32
  Max epochs:        50
  Early stopping:    val_accuracy, patience=5, restore_best_weights=True
  Split:             80/20 train/val (stratified, seed=42)
  Class weights:     None
  LR scheduler:      None

EVALUATION
  Metrics:           accuracy, precision (weighted), recall (weighted), F1 (weighted)
  Test set:          None (eval on val)
  ROC-AUC:           Not computed
  Model save:        Commented out
```

### Baseline Results (Run 01)

| Metric | Value |
|--------|-------|
| Accuracy | 89.89% |
| Precision (weighted) | 0.9068 |
| Recall (weighted) | 0.8989 |
| F1 (weighted) | 0.8997 |
| Tampered Precision | 0.8279 |
| Tampered Recall | 0.9483 |
| Tampered F1 | 0.8840 |
| Epochs trained | 13 (best: 8) |
| Confusion: TN/FP/FN/TP | 1296/202/53/972 |

### Key Observations

- Model converges fast (73% → 89% in 8 epochs)
- Overfitting begins at epoch 8 (train-val gap opens)
- Epoch 11 instability (val acc drops 4% in one epoch)
- High FP rate: 13.5% of authentic images misclassified as tampered
- 6.3% accuracy gap vs paper claims (89.89% vs 96.21%)

---

## 2. Weakness Analysis

### Category 1: Evaluation Methodology (CRITICAL)

| ID | Weakness | Audit Reference |
|----|----------|-----------------|
| W1 | No hold-out test set — val set used for both model selection and final metrics | MAJOR-2 |
| W2 | `average='weighted'` inflates metrics toward majority class; not comparable to paper | MAJOR-3 |
| W3 | No ROC-AUC metric | Minor-6 |
| W4 | No ELA visualization | Minor-5 |
| W5 | Model weights not saved | Minor-3 |

### Category 2: Data Pipeline

| ID | Weakness | Audit Reference |
|----|----------|-----------------|
| W6 | No data augmentation — assignment requires it | Section 9 |
| W7 | Class imbalance (1.46:1) not addressed | Section 4 |
| W8 | Non-JPEG images (TIF, BMP) may degrade ELA effectiveness | Section 8 |

### Category 3: Model Architecture

| ID | Weakness | Audit Reference |
|----|----------|-----------------|
| W9 | No BatchNormalization — contributes to training instability | Section 6 |
| W10 | 99.9% of params in single Flatten→Dense layer — inefficient | Section 5 |

### Category 4: Training Configuration

| ID | Weakness | Audit Reference |
|----|----------|-----------------|
| W11 | No learning rate scheduler — contributes to epoch 11 instability | Section 6 |
| W12 | Early stopping patience=5 may be too short | Section 6 |

### Category 5: Assignment Alignment

| ID | Weakness | Audit Reference |
|----|----------|-----------------|
| W13 | No localization — assignment explicitly requires pixel-level masks | MAJOR-1 |
| W14 | No Original/GT/Predicted/Overlay visualization | Section 9 |

### Category 6: Model Paradigm (NEW — from Pretrained Models Analysis)

| ID | Weakness | Audit Reference |
|----|----------|-----------------|
| W15 | No pretrained encoder — 29.5M params trained from scratch on only 8,829 images (1:3,343 data:param ratio) | Pretrained Models/01_Feasibility_Analysis |
| W16 | No spatial invariance — Flatten→Dense memorizes pixel-exact patterns, incompatible with augmentation | vR.1.2 audit, Pretrained Models/04_ELA_Compatibility_Analysis |
| W17 | ETASR CNN cannot produce localization — fundamental architecture limitation, not fixable with ablations | Pretrained Models/01_Feasibility_Analysis |

---

## 3. Versioned Ablation Roadmap

### Design Principles

1. **One change per version** — isolates the effect of each modification
2. **Fix methodology first** — proper eval before improving the model
3. **Low-effort high-impact first** — data split and metrics before architecture changes
4. **Architecture changes last** — only after pipeline and training are solid
5. **Constants across all versions**: same dataset (CASIA v2.0), same seed (42), same sorted file loading

### Roadmap

| Version | Change | Category | Weakness Fixed | Expected Impact |
|---------|--------|----------|----------------|------------------|
| **vR.1.0** | Baseline paper reproduction | — | — | 89.89% accuracy |
| **vR.1.1** | Proper 70/15/15 train/val/test split + fix metrics (per-class, macro, ROC-AUC) + ELA visualization + model save | Eval | W1, W2, W3, W4, W5 | Honest baseline; accuracy may drop 1-3% on true test set |
| **vR.1.2** | Add data augmentation (horizontal flip, vertical flip, random rotation ±15°) | Data | W6 | +1-3% accuracy; reduces overfitting gap |
| **vR.1.3** | Add class weights (inversely proportional to class frequency) | Data | W7 | Balances FP/FN; tampered precision should improve |
| **vR.1.4** | Add BatchNormalization after each Conv2D layer | Architecture | W9 | Stabilizes training; removes epoch 11-style spikes |
| **vR.1.5** | Add ReduceLROnPlateau (factor=0.5, patience=3, monitor val_loss) | Training | W11 | Better convergence at plateau; may push past 90% |
| **vR.1.6** | Add 3rd Conv2D(64, 3×3) + MaxPool before Flatten (deeper feature extraction) | Architecture | W10 | More expressive features; reduces Flatten→Dense size |
| **vR.1.7** | Replace Flatten→Dense with GlobalAveragePooling2D→Dense(256) | Architecture | W10 | Dramatically reduces params; better generalization |
| **vR.2.0** | Add ELA-based thresholding localization: threshold ELA map → binary mask + Original/ELA/Mask overlay visualization | Assignment | W13, W14 | Addresses localization requirement; completes assignment |

### Rationale for Order

1. **vR.1.1 (eval fix)** must come first. Without honest metrics on a proper test set, we cannot measure whether subsequent changes help or hurt. Every experiment after this uses the test-set numbers.

2. **vR.1.2 (augmentation)** is the single most likely change to close the 6.3% accuracy gap. Augmentation is explicitly required by the assignment and is the standard fix for overfitting in small datasets.

3. **vR.1.3 (class weights)** addresses the 13.5% FP rate. The model over-predicts tampered because the loss treats both classes equally despite the 1.46:1 ratio.

4. **vR.1.4 (BatchNorm)** stabilizes the training dynamics that caused the epoch 11 spike. Must come before the LR scheduler because BatchNorm changes the loss landscape.

5. **vR.1.5 (LR scheduler)** builds on the stabilized training from BatchNorm to squeeze out convergence gains.

6. **vR.1.6 and vR.1.7 (architecture changes)** are the most disruptive changes. They come last because we want a stable training pipeline first. vR.1.7 (GAP) especially should dramatically reduce overfitting by cutting params from 29.5M to ~250K.

7. **vR.2.0 (localization)** is the capstone. It adds ELA-based pseudo-localization to address the assignment's mask requirement. This is a separate major version because it adds a fundamentally new output modality.

---

## 3B. Pretrained Model Track — vR.P.x Roadmap (NEW)

### Rationale

The ETASR ablation study (Section 3) optimizes a classification-only model. The pretrained track addresses the fundamental limitations identified as W13, W15, W16, and W17:

- **W13/W17:** ETASR CNN cannot localize. A pretrained encoder-decoder (UNet) inherently produces pixel-level masks.
- **W15:** Data efficiency problem (1:3,343 ratio). Pretrained encoder freezes ImageNet weights, training only ~500K decoder params (1:57 ratio).
- **W16:** Pretrained ResNets have built-in spatial invariance from residual blocks and multi-scale features.

### Pretrained Roadmap

| Version | Change | Input | Encoder | Weakness Fixed | Expected Impact |
|---------|--------|-------|---------|----------------|-----------------|
| **vR.P.0** | ResNet-34 + UNet, frozen encoder | RGB 384×384 | ResNet-34 (ImageNet, frozen) | W13, W15, W17 | Establish localization baseline |
| **vR.P.1** | Dataset fix + GT mask auto-detection | RGB 384×384 | ResNet-34 (ImageNet, frozen) | — | Proper GT masks from sagnikkayalcse52 dataset |
| **vR.P.1.5** | Speed optimizations (AMP, pin_memory, prefetch) | RGB 384×384 | ResNet-34 (ImageNet, frozen) | — | Training speed only, no quality change |
| **vR.P.2** | Gradual unfreeze (last 2 blocks) | RGB 384×384 | ResNet-34 (partially unfrozen) | — | +2-5% F1 from domain adaptation |
| **vR.P.3** | ELA as input (replace RGB) | ELA 384×384 | ResNet-34 (frozen, BN unfrozen) | — | Test ELA with pretrained features |
| **vR.P.4** | 4-channel (RGB + ELA) | RGB+ELA 384×384 | ResNet-34 (frozen) | — | Test combined signal |
| **vR.P.5** | ResNet-50 encoder | RGB 384×384 | ResNet-50 (frozen) | — | Test deeper features |
| **vR.P.6** | EfficientNet-B0 encoder | RGB 384×384 | EfficientNet-B0 (frozen) | — | Test parameter efficiency |
| **vR.P.7** | Extended training (50 epochs, patience 10) | ELA 384×384 | ResNet-34 (frozen, BN unfrozen) | — | P.3 was still improving at epoch 25; test longer training |
| **vR.P.10** | Focal+Dice loss + CBAM attention in decoder | ELA 384×384 | ResNet-34 (frozen, BN unfrozen) | — | Test whether attention improves decoder feature focus |

### Pretrained Track Design Principles

1. **Same single-variable ablation methodology** as the ETASR track
2. **vR.P.0 is the initial anchor** — vR.P.1 fixes the dataset, subsequent versions change one variable from vR.P.1
3. **Framework: PyTorch + SMP** — native ResNet-34 support, built-in freeze/unfreeze, segmentation losses
4. **Same dataset** (CASIA v2.0), same seed (42), same 70/15/15 split
5. **Resolution: 384×384** — proven in v6.5 (3× more pixels than ETASR's 128×128)

### Pretrained Track Constants

| Parameter | Value | Reason |
|-----------|-------|--------|
| Dataset | CASIA v2.0 — vR.P.0: divg07 (no GT masks); vR.P.1+: sagnikkayalcse52 (with GT masks) | Dataset continuity; GT masks from vR.P.1 |
| Random seed | 42 | Reproducibility |
| Input size | 384×384 | v6.5 setting, fits T4 with batch=16 |
| Decoder | UNet (SMP default) | Skip connections from all 4 encoder stages |
| Loss | BCEDiceLoss | Proven in v6.5 |
| Batch size | 16 | Fits T4 at 384×384 resolution |
| Max epochs | 25 | v6.5 setting |
| Early stopping | patience=7, monitor=val_loss | More patient (pretrained converges slower) |
| Framework | PyTorch + SMP | Native ResNet-34, built-in tools |

### Pretrained Run Tracking Table

| Version | Change | Pixel-F1 | IoU | Pixel-AUC | Tam-F1 (cls) | Macro F1 (cls) | Test Acc | Epochs | Verdict |
|---------|--------|----------|-----|-----------|-------------|----------------|----------|--------|---------|
| **vR.P.0** | **ResNet-34 frozen, RGB (divg07, ELA pseudo-masks)** | **0.3749** | **0.2307** | **0.8486** | **0.5924** | **0.6814** | **70.63%** | **24 (17)** | **Baseline (no GT)** |
| **vR.P.1** | **Dataset fix + GT masks (sagnikkayalcse52)** | **0.4546** | **0.2942** | **0.8509** | **0.6185** | **0.6867** | **70.15%** | **25 (18)** | **Proper baseline ✅** |
| **vR.P.1.5** | **Speed optimizations (from P.1)** | **0.4227** | **0.2680** | **0.8560** | **0.6501** | **0.7016** | **71.05%** | **23 (16)** | **NEUTRAL (speed only)** |
| **vR.P.2** | **Gradual unfreeze (layer3+layer4)** | **0.5117** | **0.3439** | **0.8688** | **0.5796** | **0.6673** | **69.04%** | **14 (7)** | **POSITIVE ✅ (pixel)** |
| **vR.P.3** | **ELA input (replace RGB, BN unfrozen)** | **0.6920** | **0.5291** | **0.9528** | **0.8145** | **0.8560** | **86.79%** | **25 (25)** | **STRONG POSITIVE ✅✅** |
| **vR.P.4** | **4ch RGB+ELA (conv1+BN unfrozen)** | **0.7053** | **0.5447** | **0.9433** | **0.7873** | **0.8322** | **84.42%** | **25 (24)** | **NEUTRAL** |
| **vR.P.5** | **ResNet-50 encoder (frozen)** | **0.5137** | **0.3456** | **0.8828** | **0.6736** | **0.7143** | **72.00%** | **25 (19)** | **POSITIVE ✅** |
| **vR.P.6** | **EfficientNet-B0 encoder (frozen)** | **0.5217** | **0.3529** | **0.8708** | **0.6351** | **0.6950** | **70.68%** | **23 (16)** | **POSITIVE ✅** |
| vR.P.7 | Extended training (50 epochs, patience 10) | 0.7154 | 0.5569 | 0.9504 | 0.8637 | 0.8637 | 87.37% | 46 (36) | **POSITIVE (+2.34pp)** |
| **vR.P.8** | **Progressive unfreeze (layer4 only)** | **0.6985** | **0.5367** | **0.9541** | **0.8650** | **0.8650** | **87.59%** | **32 (23)** | **NEUTRAL (+0.65pp)** |
| vR.P.9 | Focal+Dice loss (replacing BCE+Dice) | 0.6923 | 0.5294 | 0.9323 | 0.8606 | 0.8606 | 87.16% | 25 (21) | NEUTRAL (+0.03pp) |
| vR.P.10 | Focal+Dice + CBAM attention | 0.7277 | 0.5719 | 0.9573 | 0.8615 | 0.8615 | 87.32% | 25 (24) | **POSITIVE (+3.57pp)** |
| vR.P.10 r02 | Reproducibility re-run | 0.7277 | 0.5719 | 0.9573 | 0.8615 | 0.8615 | 87.32% | 25 (24) | Reproducibility ✅ |
| **vR.P.12** | **Augmentation + Focal+Dice** | **0.6968** | **0.5347** | **0.9502** | **0.8756** | **0.8756** | **88.48%** | **45 (35)** | **NEUTRAL (+0.48pp)** |
| **vR.P.14** | **Test-Time Augmentation (TTA)** | **0.6388** | **0.4693** | **0.9618** | **0.8205** | **0.8619** | **87.43%** | **25 (25)** | **NEGATIVE (-5.32pp)** |
| **vR.P.14b** | **P.14 re-run (bug fix, complete eval)** | **0.6388** | **0.4693** | **0.9618** | **0.8205** | **0.8619** | **87.43%** | **25 (25)** | **Supersedes P.14 Run-01** |
| **vR.P.15** | **Multi-Quality ELA (Q=75/85/95)** | **0.7329** | **0.5785** | **0.9608** | **0.8660** | **0.8660** | **87.53%** | **25 (24)** | **POSITIVE (+4.09pp) — NEW SERIES BEST** |
| **vR.P.18** | **JPEG Compression Robustness** | **INVALID** | **—** | **—** | **—** | **—** | **—** | **eval-only** | **INVALID (checkpoint not found)** |

### How the Two Tracks Relate

```
Track 1 (ETASR Classification)             Track 2 (Pretrained Localization)
------------------------------              ----------------------------------
vR.1.0 -> vR.1.1 -> vR.1.2(X)
           |                                vR.P.0 (divg07, ELA pseudo-masks)
           v                                 |
         vR.1.3 -> vR.1.4 -> vR.1.5        vR.P.1 (sagnikkayalcse52, GT masks)
                               |              |          \              \
                               v            vR.P.1.5     vR.P.6 ✅     vR.P.5 ✅
                  vR.1.6 (BEST: 90.23%)       |         (EffNet-B0)   (ResNet-50)
                     |                      vR.P.2 (gradual unfreeze)
                     v                        |
                  vR.1.7 (NEUTRAL)          vR.P.3 ✅✅ (ELA input, F1=0.6920)
                                              |         \           \
                                            vR.P.4 ✅    vR.P.7 ✅   vR.P.10 ✅
                                            (4ch)      (extended)   (CBAM+Focal)
                                              |         F1=0.7154   F1=0.7277
                                         FINAL           \         /
                                        SUBMISSION      vR.P.15 ✅✅ (Multi-Q ELA)
                                        NOTEBOOK        SERIES BEST F1=0.7329
                                                           |
                                                        vR.P.19--P.28
                                                        (Phase 2 experiments)
```

- **Track 1** demonstrates ablation methodology, paper reproduction, and experimental rigor
- **Track 2** achieves the assignment requirements (localization, masks, overlays)
- The final submission uses Track 2's best model, with Track 1 documented as supporting analysis

---

## 4. Experiment Rules

### Constants (DO NOT CHANGE across versions)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Dataset | CASIA v2.0 (full, all formats) | Consistency |
| Random seed | 42 | Reproducibility |
| File loading order | `sorted(os.listdir())` | Determinism |
| ELA quality | 90 | Paper specification |
| Image size | 128×128 | Paper specification (except if explicitly ablated) |
| Framework | TensorFlow/Keras | Consistency |
| Evaluation set | Test set (from vR.1.1 onward) | Fair comparison |

### Variables (change exactly ONE per version)

The single change for each version is specified in the roadmap. All other parameters remain frozen at the previous version's values.

### Evaluation Protocol (from vR.1.1 onward)

Every version MUST report:
1. **Accuracy** on test set
2. **Per-class Precision** (Authentic + Tampered)
3. **Per-class Recall** (Authentic + Tampered)
4. **Per-class F1** (Authentic + Tampered)
5. **Macro-average** Precision, Recall, F1
6. **ROC-AUC** (probability of tampered class)
7. **Confusion matrix** (TN, FP, FN, TP)
8. **Training curves** (loss + accuracy, train vs val)
9. **Epochs trained** / best epoch
10. **Model weights saved** as `.keras` file

### Comparison Protocol

After each run:
- Compare against ALL previous versions in the results tracking table
- Determine if the change was **positive**, **neutral**, or **negative**
- If negative: the change is rejected and future versions branch from the last positive version

---

## 5. Run Tracking Table

### Template

| Version | Change | Test Acc | Au Prec | Au Rec | Au F1 | Tp Prec | Tp Rec | Tp F1 | Macro F1 | ROC-AUC | Epochs | Verdict |
|---------|--------|----------|---------|--------|-------|---------|--------|-------|----------|---------|--------|---------|
| vR.1.0 | Baseline | 89.89%* | 0.9607* | 0.8652* | 0.9104* | 0.8279* | 0.9483* | 0.8840* | 0.8972* | —* | 13 (8) | Baseline |
| **vR.1.1** | **Test split + metrics** | **88.38%** | **0.9170** | **0.8843** | **0.9004** | **0.8393** | **0.8830** | **0.8606** | **0.8805** | **0.9601** | **13 (8)** | **Honest baseline ✅** |
| **vR.1.2** | **Augmentation** | **85.53%** | **0.8843** | **0.8701** | **0.8771** | **0.8145** | **0.8336** | **0.8239** | **0.8505** | **0.9011** | **6 (1)** | **NEGATIVE — REJECTED** |
| **vR.1.3** | **Class weights** | **89.17%** | **0.9290** | **0.8852** | **0.9066** | **0.8431** | **0.9012** | **0.8712** | **0.8889** | **0.9580** | **14 (9)** | **POSITIVE ✅** |
| **vR.1.4** | **BatchNorm** | **88.75%** | **0.9401** | **0.8657** | **0.9013** | **0.8240** | **0.9194** | **0.8691** | **0.8852** | **0.9536** | **8 (3)** | **NEUTRAL** |
| **vR.1.5** | **LR scheduler** | **88.96%** | **0.9403** | **0.8692** | **0.9034** | **0.8279** | **0.9194** | **0.8712** | **0.8873** | **0.9560** | **10 (5)** | **NEUTRAL** |
| **vR.1.6** | **Deeper CNN** | **90.23%** | **0.9572** | **0.8746** | **0.9140** | **0.8372** | **0.9428** | **0.8869** | **0.9004** | **0.9657** | **18 (13)** | **POSITIVE ✅** |
| **vR.1.7** | **GAP replaces Flatten** | **89.17%** | **0.9590** | **0.8541** | **0.9035** | **0.8161** | **0.9467** | **0.8766** | **0.8901** | **0.9495** | **10 (5)** | **NEUTRAL (−1.06pp)\*** |
| vR.2.0 | ELA localization | — | — | — | — | — | — | — | — | — | — | Pending |

\* vR.1.0 metrics are on validation set (no test set). Not directly comparable to subsequent versions.

### Verdict Categories

- **POSITIVE** (+): Test accuracy or macro F1 improved by ≥ 0.5%
- **NEUTRAL** (=): Change within ±0.5% of previous version
- **NEGATIVE** (−): Test accuracy or macro F1 dropped by > 0.5%

---

## 6. Version File Structure

Each version produces:

```
New Research Approach/
├── vR.x.x Image Detection and Localisation.ipynb    ← Notebook to run on Kaggle
├── docs-vR.x.x/
│   └── version_notes.md                              ← Change description, rationale, config
└── Runs/
    └── vr-x-x-image-detection-and-localisation-run-01.ipynb  ← Executed output from Kaggle
```

---

## 7. Iteration Workflow

```
┌───────────────────────────────┐
│  1. Plan next version         │  ← You provide run results
│     Analyze previous run      │
│     Update tracking table     │
│     Decide: keep or reject    │
├───────────────────────────────┤
│  2. Generate notebook         │  ← I produce vR.x.x.ipynb
│     One change only           │
│     Freeze all other params   │
├───────────────────────────────┤
│  3. Generate documentation    │  ← I produce docs-vR.x.x/
│     Change description        │
│     Rationale + expected      │
│     Config diff from previous │
├───────────────────────────────┤
│  4. You run on Kaggle         │  ← You execute notebook
│     Save output to Runs/      │
│     Return results to me      │
├───────────────────────────────┤
│  5. I analyze results         │  ← Loop back to step 1
│     Update tracking table     │
│     Plan next version         │
└───────────────────────────────┘
```

**Abort conditions:**
- If 3 consecutive versions are NEUTRAL or NEGATIVE, reassess the roadmap
- If accuracy drops below 85%, rollback and investigate
- If a version crashes or fails to train, fix before proceeding

---

## 8. Insights from Earlier Experiments (vK.12.0)

### Source

The vK.12.0 notebook (151 cells, PyTorch/SMP) was the final iteration of the "Synthesis Era" (vK.11.x–12.0). It attempted a dual-head UNet+ResNet34 for simultaneous segmentation and classification. It failed (Tam-F1=0.1321, crashed at cell 77). Full analysis:

- `Notes from earlier Notebook/useful_ideas_summary.md` — Categorized inventory (42 ideas)
- `Notes from earlier Notebook/adoptable_improvements.md` — 21 ideas ready for integration
- `Notes from earlier Notebook/ablation_candidates.md` — 5 ideas for future ablation versions
- `Notes from earlier Notebook/discarded_elements.md` — 12 elements NOT to reuse

### Key Lessons Affecting This Roadmap

| # | Lesson | Impact |
|---|--------|--------|
| 1 | ReduceLROnPlateau prevented collapse in vK.12.0 despite broken architecture | **Confirms vR.1.5 priority** |
| 2 | Gradient clipping (clipnorm=1.0) stabilized early training | **New candidate: vR.1.8** |
| 3 | JPEG compression augmentation is signal-preserving for ELA (unlike geometric) | **Augmentation retry: vR.1.9 (after GAP)** |
| 4 | Dual-head architectures fail without extensive loss-weight tuning | **vR.P.0 correctly uses single-task UNet** |
| 5 | Tampered-only metric filtering reveals true forensic performance | **Add to all notebooks (non-variable)** |
| 6 | Per-forgery-type evaluation (copy-move vs splicing) reveals attack-specific weaknesses | **Add to evaluation protocol** |

### Evaluation Protocol Enhancement

The following non-variable additions are recommended for all versions from vR.1.3 onward (~66 lines total):

| Addition | Lines | Value |
|----------|-------|-------|
| Data leakage assertion (set intersection) | 3 | Critical safety |
| Best-epoch marker on training curves | 2 | Visual clarity |
| Tampered-only metric highlight | 5 | Forensic focus |
| Threshold sweep on validation set | 15 | Better calibration |
| Per-forgery-type breakdown | 10 | Attack-type analysis |
| Worst-10 failure case grid | 20 | Error diagnosis |
| Environment info cell | 5 | Reproducibility |
| Seed verification | 3 | Reproducibility |
| Split hash verification | 3 | Determinism check |

These do NOT change any ablation variable — they are evaluation infrastructure only.

### Tentative Roadmap Extension

If vR.1.3 through vR.1.7 complete and time permits:

| Version | Change | Source |
|---------|--------|--------|
| vR.1.8 | Gradient clipping (clipnorm=1.0) | vK.12.0 cell 69 |
| vR.1.9 | JPEG compression augmentation (QF 50–90) | vK.12.0 cell 44 |
| vR.1.10 | Alternative ELA (cv2 vs PIL) | vK.12.0 cell 43 |

These are tentative and subject to results from vR.1.3–1.7.

---

## 9. Additional Experiments from Research Analysis and Forensic Feature Exploration

### Overview

The following 10 experiments extend the pretrained localization track (vR.P.x) with new forensic feature inputs, architectural improvements, and training enhancements identified through research analysis and internal experimentation. These start at vR.P.19 (continuing from the existing P.18 DCT robustness experiment).

### Duplicate Detection Summary

Before adding new experiments, each proposed technique was checked against ALL existing experiments (P.0--P.18):

| Proposed Technique | Existing Coverage | Decision |
|---|---|---|
| Multichannel ELA for RGB | P.3 (ELA RGB 3ch) | SKIP -- duplicate |
| RGB ELA (color artifact) | P.3 (same) | SKIP -- duplicate |
| **Multi-quality RGB ELA** | P.15 uses grayscale only | **ADD as P.19** -- RGB variant is distinct |
| **ELA magnitude channel** | Not covered | **ADD as P.20** |
| **ELA residual learning** | Not covered | **ADD as P.21** |
| **Noiseprint features** | Not covered | **ADD as P.24** |
| **SRM noise maps** | Not covered | **ADD as P.22** |
| **Chrominance analysis** | Not covered | **ADD as P.23** |
| Frequency-domain (DCT) | P.16 (DCT spatial maps) | SKIP -- duplicate |
| RGB + ELA fusion | P.4 (4ch RGB+ELA) | SKIP -- duplicate |
| **Edge supervision loss** | Not covered | **ADD as P.25** |
| Lightweight attention (SE/CBAM) | P.10 (CBAM) | SKIP -- covered |
| **Dual-task seg+classification** | Not covered | **ADD as P.26** |
| **JPEG compression augmentation** | P.18 tests but doesn't train | **ADD as P.27** -- training-time aug is distinct |
| Photometric augmentation | P.12 (partly covered) | SKIP -- sufficiently covered |
| **Cosine annealing LR** | Not covered | **ADD as P.28** |
| Pixel-level AUC ROC | Already in all P.x notebooks | SKIP -- duplicate |

### New Experiment Roadmap (vR.P.19--P.28)

#### Group A: Feature Domain Experiments (ELA Variants)

| Version | Technique | Parent | Single Variable | Expected Impact |
|---------|-----------|--------|-----------------|-----------------|
| **vR.P.19** | Multi-Quality RGB ELA (9ch, Q=75/85/95) | P.3 | Input representation (9ch conv1) | +2-5pp Pixel F1 |
| **vR.P.20** | ELA Magnitude + Chrominance Decomposition | P.3 | Input representation (3ch) | +1-3pp Pixel F1 |
| **vR.P.21** | ELA Residual Learning (Laplacian high-pass) | P.3 | Input representation (3ch) | +2-4pp Pixel F1 |

#### Group B: Feature Domain Experiments (Noise-Based Features)

| Version | Technique | Parent | Single Variable | Expected Impact |
|---------|-----------|--------|-----------------|-----------------|
| **vR.P.22** | SRM Noise Maps (3 SRM filters) | P.3 | Input representation (3ch) | +1-4pp Pixel F1 |
| **vR.P.23** | Chrominance Channel Analysis (YCbCr) | P.3 | Input representation (3ch) | +0-3pp Pixel F1 |
| **vR.P.24** | Noiseprint Forensic Features (DnCNN residual) | P.3 | Input representation (3ch) | +2-6pp Pixel F1 |

#### Group C: Architecture Experiments

| Version | Technique | Parent | Single Variable | Expected Impact |
|---------|-----------|--------|-----------------|-----------------|
| **vR.P.25** | Edge Supervision Loss (Sobel edge BCE) | P.3 | Loss function (add edge term) | +1-3pp Pixel F1 |
| **vR.P.26** | Dual-Task Segmentation + Classification | P.3 | Architecture (add cls head) | +1-2pp Pixel F1 |

#### Group D: Training Experiments

| Version | Technique | Parent | Single Variable | Expected Impact |
|---------|-----------|--------|-----------------|-----------------|
| **vR.P.27** | JPEG Compression Augmentation (training-time) | P.3 | Augmentation (JPEG recompress) | +1-3pp standard, +5-10pp robustness |
| **vR.P.28** | Cosine Annealing LR Scheduler | P.3 | LR scheduler + 50 epochs | +1-2pp Pixel F1 |

### Execution Priority

1. **vR.P.21** (ELA residual) -- High-impact input experiment, simple implementation
2. **vR.P.22** (SRM noise) -- Novel forensic feature, orthogonal to ELA
3. **vR.P.19** (Multi-Q RGB ELA) -- Extends P.15's multi-quality idea with color
4. **vR.P.25** (Edge supervision) -- Improves boundary precision, loss-only change
5. **vR.P.28** (Cosine annealing) -- Training improvement, pairs well with 50-epoch budget
6. **vR.P.20** (ELA magnitude) -- Interesting decomposition, low implementation cost
7. **vR.P.26** (Dual-task) -- Unifies classification + segmentation
8. **vR.P.27** (JPEG compression aug) -- Robustness training
9. **vR.P.23** (YCbCr chrominance) -- Lower expected impact for CASIA
10. **vR.P.24** (Noiseprint) -- Highest potential but hardest implementation (needs DnCNN)

### Documentation

All 10 experiments have complete documentation in `Docs vR.P.x/docs-vR.P.{19-28}/`:
- `experiment_description.md` -- Hypothesis, motivation, pipeline, configuration
- `implementation_plan.md` -- Cell modification map, key code, verification checklist
- `expected_outcomes.md` -- Metric targets, success criteria, failure modes

### Relation to Existing Experiments

```
Feature Domain Experiments:
    ELA Variants:
        P.3  (ELA RGB baseline)
        P.15 (Multi-Q grayscale ELA)
        P.19 (Multi-Q RGB ELA)        ← NEW
        P.20 (ELA magnitude decomp)   ← NEW
        P.21 (ELA Laplacian residual)  ← NEW

    Noise-Based Features:
        P.22 (SRM noise maps)         ← NEW
        P.24 (Noiseprint DnCNN)       ← NEW

    Frequency-Domain Features:
        P.16 (DCT spatial maps)
        P.17 (ELA + DCT fusion)

    Color-Space Features:
        P.23 (YCbCr chrominance)       ← NEW
        P.4  (RGB + ELA fusion)

Architecture Experiments:
    Attention:
        P.10 (CBAM in decoder)
    Edge Supervision:
        P.25 (Sobel edge loss)         ← NEW
    Dual-Task Models:
        P.26 (Seg + Classification)    ← NEW

Training Experiments:
    Augmentation:
        P.12 (Albumentations geometric+photo)
        P.27 (JPEG compression aug)    ← NEW
    Compression Robustness:
        P.18 (robustness evaluation)
    LR Scheduling:
        P.28 (Cosine annealing)        ← NEW

Evaluation Experiments:
    P.14 (TTA -- negative result)
    P.18 (compression robustness)
```

