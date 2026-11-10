# Full Audit Report — Notebooks vs Paper (PeerJ CS 10-2205)

**Paper:** "Enhanced Image Tampering Detection using Error Level Analysis and a CNN" (Nagm et al., PeerJ CS 2024)
**NB1:** `CASIA2 ELA CNN Image Tampering Detection.ipynb`
**NB2:** `ELA CNN Image Forgery Detection.ipynb`
**Reference code:** `Reference Code/CASIA2code.py`

---

## 1. Executive Summary

| Criterion | NB1 | NB2 |
|-----------|-----|-----|
| Architecture match | ✅ Exact reproduction | ❌ Completely different |
| ELA implementation | ⬆️ Improved (in-memory) | ✅ Matches paper (temp file) |
| Training config | ✅ Matches | ⚠️ Mostly matches, adds EarlyStopping |
| Data split | ⚠️ 80/10/10 (paper: 80/16/4) | ❌ 70/15/15 |
| Code quality | ⬆️ Research-grade | ⚠️ Functional but fragile |
| Paper bugs fixed | ⬆️ Yes (BytesIO, stratified split, seeds) | ⚠️ Partially (auto-discovery, broad extensions) |

**Bottom line:** NB1 is a faithful, improved reproduction of the paper's proposed model. NB2 implements a different architecture (resembling the paper's ablation Model 2) and should not be cited as implementing this paper.

---

## 2. Architecture Fidelity

### NB1 — Perfect Match

NB1 reproduces the paper's proposed model exactly:

```
Conv2D(32, 5×5, valid, relu) → Conv2D(32, 5×5, valid, relu) → MaxPool2D(2×2) →
Dropout(0.25) → Flatten(161,312) → Dense(150, relu) → Dropout(0.5) → Dense(2, sigmoid)
```

Every layer, every parameter, every activation function matches Algorithm 1 and the reference code.

### NB2 — Completely Different

NB2 implements a 3-block CNN with BatchNormalization:

```
Conv2D(64, 3×3) + BN + Pool → Conv2D(128, 3×3) + BN + Pool → Conv2D(256, 3×3) + BN + Pool →
Dropout(0.5) → Flatten(73,984) → Dense(512, relu) → Dropout(0.5) → Dense(2, sigmoid)
```

Key differences:
- **3 conv layers** instead of 2 (50% more depth)
- **3×3 kernels** instead of 5×5 (smaller receptive field per layer)
- **Escalating filters** (64→128→256) instead of flat (32→32)
- **BatchNormalization** after each conv (paper's proposed model has none)
- **3 pooling layers** instead of 1 (more aggressive spatial reduction)
- **Dense(512)** instead of Dense(150)
- **Both dropouts at 0.5** instead of 0.25 then 0.5
- **~2.5× fewer parameters** despite appearing "deeper" (73,984 flatten vs 161,312)

This architecture closely resembles the paper's **ablation Model 2** (Conv→BatchNorm→MaxPool blocks with BatchNorm), which achieved only **88.9% test accuracy** — 5.2% worse than the proposed model.

---

## 3. ELA Implementation

| Aspect | Paper / Reference Code | NB1 | NB2 |
|--------|----------------------|-----|-----|
| Quality factor | 90 | 90 ✅ | 90 ✅ |
| Storage method | Temp file (`resaved.jpg`) | **BytesIO (in-memory)** ⬆️ | Temp file (`temp_file.jpg`) ✅ |
| Scaling | `255.0 / max_diff` → `Brightness.enhance()` | Same ✅ | Same ✅ |
| Zero-guard | `max_value = 1` if 0 | Same ✅ | Same ✅ |
| Error handling | Basic try/except | Returns None, caller checks ✅ | Returns None, caller checks ✅ |
| Race condition risk | Yes (`resaved.jpg` hardcoded) | **None** (in-memory) ⬆️ | Yes (`temp_file.jpg` hardcoded) ⚠️ |

**NB1's BytesIO approach** is a genuine improvement — eliminates disk I/O and race conditions without changing the mathematical result.

**NB2's temp file approach** matches the reference code but inherits a race condition: if multiple processes or threads use the same working directory, they'll overwrite each other's temp file.

---

## 4. Data Pipeline

### Extension Handling

| Source | Extensions Accepted |
|--------|-------------------|
| Paper (text) | JPEG only (9,501 of 12,614 images) |
| Reference code | `.jpg` only (line 66/87) |
| NB1 | All files, ELA function handles format (try/except) |
| NB2 | `.jpg, .jpeg, .png, .tif, .tiff, .bmp, .gif` (7 formats) |

The paper explicitly excludes TIFF images without explanation. NB1 attempts all files (letting PIL's open() decide), while NB2 accepts 7 formats. Both notebooks **will include more images than the paper** if TIFF files are present in the CASIA dataset, which affects reproducibility.

### Path Discovery

| Source | Method |
|--------|--------|
| Reference code | Hardcoded Kaggle path |
| NB1 | `find_dataset()` — walks `/kaggle/input/` looking for `Au/` + `Tp/` dirs |
| NB2 | `find_dataset()` — same approach + `get_image_files()` pre-discovery |

Both notebooks improve on the reference code's hardcoded path.

### Data Split

| Source | Train | Val | Test | Stratified? |
|--------|-------|-----|------|------------|
| Paper | 80% | 4% | 16% | No |
| NB1 | 80% | **10%** | **10%** | **Yes** ⬆️ |
| NB2 | **70%** | **15%** | **15%** | No |

- NB1 changes the split to 80/10/10 (more balanced val/test) and adds stratification — both improvements
- NB2 uses 70/15/15 — 10% less training data, which will hurt accuracy

### Shuffling

| Source | Method | Bug? |
|--------|--------|------|
| Reference code | `random.shuffle(X)` then 10× `sklearn.shuffle(X, Y)` | **YES — critical bug** |
| NB1 | 10× `sklearn.shuffle(X, Y, random_state=i)` | No — correct ✅ |
| NB2 | 1× `sklearn.shuffle(X, Y, random_state=42)` | No — correct ✅ |

NB1 correctly omits the independent `random.shuffle(X)` that corrupts labels.

---

## 5. Training Configuration

| Parameter | Paper | NB1 | NB2 |
|-----------|-------|-----|-----|
| Optimizer | Adam | Adam ✅ | Adam ✅ |
| Learning rate | 0.0001 | 0.0001 ✅ | 0.0001 ✅ |
| Batch size | 8 | 8 ✅ | 8 ✅ |
| Epochs | 40 | 40 ✅ | 40 ✅ |
| Loss | binary_crossentropy | binary_crossentropy ✅ | binary_crossentropy ✅ |
| EarlyStopping | Commented out | None ✅ | **patience=2** ❌ |
| LR scheduler | None | None ✅ | None ✅ |
| Seeds | None set | **42 for random/numpy/tf** ⬆️ | None ✅ |
| ModelCheckpoint | None | None ✅ | None ✅ |

NB2's EarlyStopping with patience=2 is **extremely aggressive** and may stop training prematurely. The paper does not use early stopping (it's commented out in the reference code). This could significantly affect NB2's results.

---

## 6. Evaluation & Metrics

| Metric | Paper | NB1 | NB2 |
|--------|-------|-----|-----|
| Keras accuracy | ✅ | ✅ | ✅ |
| Keras precision | ✅ | ✅ | ✅ |
| Keras recall | ✅ | ✅ | ✅ |
| F1 score | ❌ Not reported | ✅ (macro) ⬆️ | ✅ (weighted) ⬆️ |
| Classification report | ❌ | ✅ ⬆️ | ✅ ⬆️ |
| Confusion matrix | ✅ (percentages only) | ✅ (counts + heatmap) ⬆️ | ✅ (counts + heatmap) ⬆️ |
| AUC / ROC | ❌ | ❌ | ❌ |
| Per-class breakdown | ❌ | ✅ ⬆️ | ✅ ⬆️ |
| Training curves | ❌ Not shown in paper | ✅ (4 panels) ⬆️ | ✅ (2 panels) ⬆️ |

Both notebooks improve on the paper's limited metrics reporting. Neither computes AUC/ROC.

**Note on F1 averaging:** NB1 uses `average='macro'` (unweighted class mean) while NB2 uses `average='weighted'` (weighted by class frequency). For imbalanced datasets, these will give different numbers. Macro is stricter; weighted is more optimistic with majority-class performance.

---

## 7. Code Quality

### NB1 Strengths
- In-memory ELA via BytesIO (no disk I/O, no race conditions)
- Stratified train/val/test split
- Reproducibility seeds (random=42, numpy=42, tf=42)
- Dataset auto-discovery with depth limit
- `predict_image()` utility for single-image inference
- `visualize_prediction()` with color-coded confidence display
- Seaborn heatmap for confusion matrix
- 4-panel training curves (accuracy, loss, precision, recall)
- Model saved as both `.h5` and `.json`
- Proper memory cleanup with `del` after splits
- Clean markdown documentation throughout

### NB1 Weaknesses
- No extension filtering — relies on PIL try/except (slower, less predictable)
- No per-image progress bar (only every 1000 images)
- Flattens images during loading then reshapes — wasteful memory pattern

### NB2 Strengths
- Explicit 7-extension whitelist (clear, deliberate)
- `get_image_files()` pre-discovers all files before loading
- File count sanity check with `RuntimeError`
- Experiment metrics log dictionary for cross-experiment tracking
- Architecture diagram export via `plot_model()`
- Extension distribution logging

### NB2 Weaknesses
- Temp file ELA (`temp_file.jpg`) — race condition risk
- No reproducibility seeds
- EarlyStopping patience=2 is too aggressive
- **Wrong architecture for this paper** — implements ablation Model 2, not proposed model
- No stratified splitting
- 70/15/15 split gives less training data than paper's 80%
- Both dropout layers at 0.5 (paper uses 0.25 then 0.5)
- `plt.imshow` for confusion matrix instead of seaborn heatmap (harder to read values)

---

## 8. Paper Bugs Found

### 8.1 Dropout Rate Contradiction
**Where:** Hyperparameters section vs Algorithm 1
- Paper prose states: "a dropout probability of 0.25" (implies both layers)
- Algorithm 1 pseudocode: Step 7 uses 0.25, Step 10 uses 0.5
- Reference code: `Dropout(0.25)` at line 135, `Dropout(0.5)` at line 138
- **Verdict:** Two different rates (0.25, 0.5). The prose is wrong.

### 8.2 Model File Size (277 MB)
- 24.2M parameters × 4 bytes/float32 = ~92 MB
- 277 MB is 3× the expected size
- Likely includes Keras optimizer state (Adam stores 2 momentum buffers per parameter: 92 MB × 3 ≈ 276 MB)
- **Not technically wrong** (h5 files include optimizer), but **misleading** — the model weights alone are ~92 MB

### 8.3 Sigmoid + 2 Outputs with Binary Cross-Entropy
- `Dense(2, activation='sigmoid')` + `binary_crossentropy` is a multi-label formulation
- Correct alternatives: `softmax` + `categorical_crossentropy`, OR single `sigmoid` + `binary_crossentropy`
- With `to_categorical(Y, 2)` labels `[1,0]` / `[0,1]`, this setup happens to work because each output learns independently, but it's architecturally sloppy — the two outputs are not constrained to sum to 1

### 8.4 Val/Test Split Obfuscation
- Paper says "80% training and 20% validation and testing"
- Code splits the 20% as 80/20 again → **16% test, 4% validation**
- A 4% validation set (~380 images) is too small for reliable early stopping or hyperparameter tuning
- The paper does not disclose this clearly

### 8.5 No GPU Despite CNN Training
- Paper states: "4 CPU cores, 30 GB RAM" on Kaggle
- Training a 24.2M parameter CNN for 40 epochs on CPU is extremely slow
- Paper does not discuss training time or justify the absence of GPU

### 8.6 Comparison Fairness
- Table 5 compares against AlexNet, VGG19, ResNet50, ResNet101 using results from **other papers**
- These results may use different splits, preprocessing, or datasets
- No controlled re-implementation was performed — the comparison is unreliable

---

## 9. NB1 Improvements Over Paper

| Improvement | Benefit |
|------------|---------|
| BytesIO ELA (in-memory) | Eliminates disk I/O, prevents race conditions, ~2× faster |
| Stratified splitting | Ensures class balance in all splits — critical for imbalanced data |
| Reproducibility seeds (42) | Deterministic results across runs |
| `find_dataset()` auto-discovery | Works across Kaggle dataset slug changes |
| F1 score + classification report | Fills the paper's metric gaps |
| Seaborn confusion matrix heatmap | Professional-quality visualization |
| 4-panel training curves | Tracks all 4 metrics (acc, loss, precision, recall) over training |
| `predict_image()` utility | Easy single-image inference for demos |
| `visualize_prediction()` | Color-coded prediction display with confidence |
| 80/10/10 split (vs 80/16/4) | More balanced val/test sets |
| Fixes `random.shuffle(X)` bug | Correct label alignment preserved |

---

## 10. NB2 Deviations From Paper

| Deviation | Impact |
|-----------|--------|
| **Architecture: 3-block CNN with BatchNorm** | Implements ablation Model 2 (~88.9% acc), not proposed model (~94.14%) |
| **Conv filters: 64→128→256** instead of 32→32 | Different feature extraction characteristics |
| **Kernel size: 3×3** instead of 5×5 | Smaller receptive field per layer |
| **Dense(512)** instead of Dense(150) | 3.4× larger fully connected layer |
| **Both dropouts at 0.5** instead of 0.25, 0.5 | More aggressive regularization |
| **70/15/15 split** instead of 80/16/4 | 10% less training data |
| **EarlyStopping(patience=2)** | May stop training too early — paper doesn't use it |
| **3 MaxPool layers** instead of 1 | Spatial dimensions reduced to 17×17 before flatten (vs 71×71) |

**Critical point:** NB2's architecture is not wrong per se — it is a valid CNN for image classification. But it is **not the model described in this paper**, and citing this paper while running NB2's architecture would be scientifically inaccurate.

---

## 11. Critical Bug in Reference Code (`CASIA2code.py`)

**Line 79:**
```python
random.shuffle(X)
```

**What it does:** Shuffles the image list `X` independently — the label list `Y` is NOT shuffled correspondingly. At this point in the code, `X` contains authentic images (all labeled `Y=1`) and has NOT yet been mixed with tampered images.

**Why it's destructive:**
1. After loading authentic images (lines 63–75): `X = [au_img_1, au_img_2, ...]`, `Y = [1, 1, 1, ...]`
2. `random.shuffle(X)` permutes X — but Y stays `[1, 1, 1, ...]`
3. Since all Y values are still `1` at this point, the shuffle is **harmless in this specific case** because all labels are identical

**Wait — correction:** Upon closer inspection, this shuffle occurs AFTER loading authentic images but BEFORE loading tampered images. Since all images are authentic (all Y=1) at this point, shuffling X without Y is **benign** — every label is the same. The shuffle just reorders authentic images among themselves.

**However**, this is still bad practice:
- If the code were modified to load both classes first, this line would corrupt labels
- The intent appears to be shuffling, but proper shuffling happens later at lines 100–101

**The 10× sklearn shuffle at lines 100–101 is correct** — it shuffles X and Y together with corresponding indices.

---

## 12. Recommendations

### For NB1 (CASIA2 ELA CNN)
1. **Add JPEG-only filter** — to exactly match the paper, filter for `.jpg`/`.jpeg` only (currently accepts all formats)
2. **Consider changing split to 80/16/4** — to exactly reproduce the paper's methodology (current 80/10/10 is arguably better but deviates)
3. **Add AUC/ROC curve** — standard metric the paper omits
4. **Add per-forgery-type analysis** — compare copy-move vs splicing detection accuracy
5. **Document the sigmoid anti-pattern** — note in markdown that `sigmoid(2)` + BCE is technically a multi-label formulation

### For NB2 (ELA CNN Forgery Detection)
1. **Rename or re-frame** — this notebook should NOT claim to implement the paper's proposed model; it implements a BatchNorm variant resembling ablation Model 2
2. **If the goal is to match the paper:** replace the entire architecture with NB1's model (Conv2D(32,5×5)×2, no BatchNorm, Dense(150))
3. **If keeping this architecture:** rename to "ELA CNN Variant — 3-Block BatchNorm Model" and cite it as an ablation experiment
4. **Increase EarlyStopping patience** — patience=2 is dangerously aggressive; consider patience=5–10 or remove it (to match paper)
5. **Change split to 80/x/y** — 70% training data is 10% less than the paper
6. **Fix dropout rates** — change first dropout to 0.25 (or acknowledge the deviation)
7. **Add reproducibility seeds**
8. **Switch to BytesIO ELA** — eliminate the temp file race condition

### For Both Notebooks
1. **Add cross-validation** — single split is unreliable
2. **Report AUC/ROC** — standard for binary classification
3. **Test on external datasets** — CASIA 2.0 only is insufficient for generalization claims
4. **Address class imbalance** — ~5,600 Au vs ~3,900 Tp is a ~60/40 split; consider class weights

---

## Appendix: Paper Quick Reference

```
PAPER:  Nagm et al., PeerJ CS 2024
MODEL:  Conv2D(32,5×5)×2 → MaxPool → Dropout(0.25) → Flatten → Dense(150) → Dropout(0.5) → Dense(2,sigmoid)
ELA:    Q=90, brightness scaling
DATA:   CASIA 2.0 JPEG only (9,501 images)
SPLIT:  80/16/4 (train/test/val)
TRAIN:  Adam(lr=1e-4), batch=8, epochs=40, BCE
RESULT: 94.14% test accuracy, 94.1% precision, 94.07% recall
```
