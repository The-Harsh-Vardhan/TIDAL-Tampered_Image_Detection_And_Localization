# Code Audit — Reference Implementations

## Files Audited

1. `Reference Code/CASIA2code.py` — Primary reference implementation
2. `Reference Code/code.py` — Secondary reference with alternative architecture

---

## 1. CASIA2code.py — Detailed Audit

### 1.1 Correct Components

| Component | Status | Notes |
|-----------|--------|-------|
| ELA function (`image_to_ela`) | **Correct** | Proper ELA implementation: re-save, difference, scale |
| PIL-based image handling | **Correct** | Proper RGB conversion and JPEG re-save |
| Quality parameter (Q=90) | **Correct** | Matches paper specification |
| Brightness scaling in ELA | **Correct** | Scales by 255/max_diff to enhance visibility |
| Train/test split | **Partially correct** | Uses sklearn but splits into 3 sets unnecessarily |
| Optimizer choice (Adam) | **Correct** | Matches paper |
| Loss function (binary_crossentropy) | **Correct** | Appropriate for binary classification |
| Metrics (Precision, Recall) | **Correct** | Useful evaluation metrics |

### 1.2 Incorrect Implementations

#### Issue 1: Wrong Image Size (CRITICAL)
```python
# In CASIA2code.py:
image_size = (150, 150)  # Line 55
X = X.reshape(-1, 150, 150, 3)  # Line 109
input_shape = (150, 150, 3)  # Line 132
```
**Paper specifies:** 128 × 128 × 3
**Impact:** Architectural mismatch — the CNN was designed for 128×128 input

#### Issue 2: Wrong Dense Layer Size
```python
model.add(Dense(150, activation='relu'))  # Line 137
```
**Paper specifies:** Dense(256, activation='relu')
**Impact:** Reduced model capacity compared to paper architecture

#### Issue 3: Wrong Output Activation (CRITICAL)
```python
model.add(Dense(2, activation='sigmoid'))  # Line 139
```
**Paper specifies:** `softmax` activation for 2-class output
**Impact:** Sigmoid on 2 units does not produce a valid probability distribution. Each unit independently outputs [0,1] without summing to 1. This creates a fundamental mathematical inconsistency with `binary_crossentropy` loss applied to one-hot labels.

#### Issue 4: Shuffling X Without Y
```python
random.shuffle(X)  # Line 79
```
**Impact:** Shuffles feature list *before* tampered images are added, but Y is not shuffled in sync. This destroys the label-feature correspondence for authentic images. However, since X is a Python list at this point and all authentics have Y=1, the damage is contained — but it's still incorrect practice.

#### Issue 5: Flattening Then Reshaping
```python
def prepare_image(image_path):
    return np.array(image_to_ela(image_path, 90).resize(image_size)).flatten() / 255.0
# Later:
X = X.reshape(-1, 150, 150, 3)
```
**Impact:** Images are flattened to 1D vectors during loading, stored in a list, then reshaped back to 4D. This wastes memory and is error-prone — any image that fails to match the expected flattened size will cause a silent reshape error.

#### Issue 6: Only Loading .jpg Files
```python
if filename.endswith('jpg'):  # Lines 66, 88
```
**Impact:** Ignores `.png`, `.tif`, `.bmp` and other image formats present in CASIA v2.0. This discards a significant portion of the dataset.

#### Issue 7: Early Stopping Monitor is Wrong
```python
early_stopping = EarlyStopping(monitor='val_acc', ...)  # Line 166
```
**Impact:** In modern Keras, the metric is named `val_accuracy`, not `val_acc`. This silently fails to find the metric and early stopping never triggers.

#### Issue 8: Early Stopping is Commented Out
```python
# callbacks = [early_stopping]  # Line 178
```
**Impact:** Early stopping is defined but never used — training runs for all 40 epochs regardless of overfitting.

#### Issue 9: Confusion Matrix Tick Labels Wrong
```python
plt.xticks(np.arange(1), [str(i) for i in range(1)])  # Line 235
plt.yticks(np.arange(1), [str(i) for i in range(1)])  # Line 236
```
**Impact:** Only shows label "0" on each axis. Should show both "Authentic" and "Tampered" labels for a 2×2 matrix.

### 1.3 Missing Steps

- No data augmentation
- No F1 score computation
- No classification report
- No cross-validation
- No learning rate scheduling
- No model checkpointing (best model not saved)
- No reproducibility seeding

---

## 2. code.py — Detailed Audit

### 2.1 Correct Components

| Component | Status | Notes |
|-----------|--------|-------|
| ELA function (`convert_to_ela_image`) | **Correct** | Clean implementation |
| ELA function (`image_to_ela`) | **Correct** | Duplicate of CASIA2code.py version |

### 2.2 Incorrect Implementations

#### Issue 1: Completely Wrong Architecture (CRITICAL)
```python
model.add(Conv2D(filters=64, kernel_size=(3, 3), ...))    # 64 filters, 3×3
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), ...))   # 128 filters, 3×3
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), ...))   # 256 filters, 3×3
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
```
**Paper specifies:** 2× Conv2D with 32 filters, 5×5 kernels, single MaxPool2D
**Impact:** This is a completely different, much larger architecture. It does NOT match the paper at all.

#### Issue 2: Fatal Output Layer Bug (CRITICAL)
```python
model.add(Dense(1, activation='softmax'))  # Line 184
```
**Impact:** Softmax on a single unit always outputs 1.0. This model cannot learn anything — the output is a constant. This is a **fatal bug** that renders the entire model non-functional.

#### Issue 3: Contradictory Reshape Operations
```python
X = X.reshape(-1, 150, 150, 3)  # Line 137 — reshape to 150×150
# ...
X = X.reshape(-1, 128, 128, 3)  # Line 142 — reshape to 128×128
```
**Impact:** The data is resized to 150×150 during loading but then reshaped to 128×128 without actual re-interpolation. This would either crash (dimension mismatch) or corrupt the pixel data.

#### Issue 4: Double Encoding
```python
Y = to_categorical(Y, 2)  # Line 136
# ...
Y = to_categorical(Y, 2)  # Line 141
```
**Impact:** Labels are one-hot encoded twice. The second call tries to one-hot encode already one-hot encoded data, producing incorrect labels.

#### Issue 5: Dataset Artificially Limited
```python
if len(Y) == 100:   # Au capped at 100
    break
if len(Y) == 200:   # Tp capped at 100 (total 200)
    break
```
**Impact:** Only 200 total images used. CASIA v2.0 has ~12,000+ images. This produces an undertrained, unrepresentative model.

#### Issue 6: Confusion Matrix Has 10 Classes
```python
plt.xticks(np.arange(10), [str(i) for i in range(10)])  # Line 259
plt.yticks(np.arange(10), [str(i) for i in range(10)])  # Line 260
```
**Impact:** Displays 10 class labels for a 2-class problem. Clearly copy-pasted from a different project (likely MNIST/CIFAR-10).

#### Issue 7: Missing Adam Import
```python
from keras.callbacks import EarlyStopping
# Adam import is missing but model.compile uses Adam(...)
```
**Impact:** Will throw a NameError at compile time.

### 2.3 Outdated APIs

Both files use deprecated Keras imports:
```python
from keras.utils.np_utils import to_categorical  # Deprecated
from keras.models import Sequential               # Should use tensorflow.keras
from keras.layers import Dense, ...               # Should use tensorflow.keras
```

**Modern equivalent:** All imports should use `tensorflow.keras.*` or `keras.*` (Keras 3).

---

## 3. Summary Comparison

| Aspect                | Paper Spec          | CASIA2code.py       | code.py              |
|-----------------------|---------------------|----------------------|----------------------|
| Image size            | 128×128             | **150×150** ✗        | 150×150/128×128 ✗    |
| Conv filters          | 32, 32              | 32, 32 ✓             | **64, 128, 256** ✗   |
| Kernel size           | 5×5                 | 5×5 ✓                | **3×3** ✗            |
| Num Conv layers       | 2                   | 2 ✓                  | **3** ✗              |
| Batch Normalization   | Not used            | Not used ✓            | **Used** ✗           |
| Dense units           | 256                 | **150** ✗            | 512 ✗                |
| Output activation     | Softmax (2 units)   | **Sigmoid** ✗        | **Softmax (1 unit)** ✗✗ |
| MaxPool layers        | 1                   | 1 ✓                  | **3** ✗              |
| ELA quality           | 90                  | 90 ✓                 | 90 ✓                 |
| ELA implementation    | Correct             | Correct ✓            | Correct ✓            |
| Dataset loading       | All images          | JPG only ✗           | **100 only** ✗✗     |
| Runs without error    | —                   | Mostly ✓             | **No** (fatal bugs) ✗|

---

## 4. Recommendation

**Neither reference file can be used as-is.** The correct approach is:

1. **Use the ELA function** from `CASIA2code.py` (it's correctly implemented)
2. **Reconstruct the CNN** from the paper specification (128×128 input, 32-32 Conv, 256 Dense, Softmax output)
3. **Fix dataset loading** to include all image formats and the full dataset
4. **Use modern TensorFlow/Keras APIs**
5. **Implement proper evaluation** with all required metrics
