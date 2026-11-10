# Improvements Over Reference Code

This document details specific improvements made in the clean implementation (`vR.ETASR Image Detection and Localisation.ipynb`) compared to the raw reference scripts (`CASIA2code.py` and `code.py`).

---

## 1. Architecture Corrections

### 1.1 Correct Input Size
| Reference Code | Improvement |
|----------------|-------------|
| `image_size = (150, 150)` | `IMAGE_SIZE = (128, 128)` |

**Rationale:** The paper specifies 128×128 input. Using the correct size ensures architectural alignment.

### 1.2 Correct Dense Layer
| Reference Code | Improvement |
|----------------|-------------|
| `Dense(150, activation='relu')` | `Dense(256, activation='relu')` |

**Rationale:** Paper specifies 256 units in the fully connected layer.

### 1.3 Correct Output Activation
| Reference Code | Improvement |
|----------------|-------------|
| `Dense(2, activation='sigmoid')` (CASIA2code.py) | `Dense(2, activation='softmax')` |
| `Dense(1, activation='softmax')` (code.py) | `Dense(2, activation='softmax')` |

**Rationale:**
- `sigmoid` on 2 units: outputs don't sum to 1, inconsistent with categorical cross-entropy
- `softmax` on 1 unit: always outputs 1.0, model cannot learn
- `softmax` on 2 units: correct probability distribution over 2 classes

### 1.4 Correct Loss Function
| Reference Code | Improvement |
|----------------|-------------|
| `binary_crossentropy` with one-hot labels | `categorical_crossentropy` with one-hot labels |

**Rationale:** With 2-unit softmax output and one-hot encoded labels, `categorical_crossentropy` is the mathematically correct loss function.

---

## 2. Dataset Loading Improvements

### 2.1 All Image Formats Supported
| Reference Code | Improvement |
|----------------|-------------|
| Only `.jpg` files loaded | `.jpg`, `.jpeg`, `.png`, `.tif`, `.bmp` all supported |

**Rationale:** CASIA v2.0 contains mixed formats. Filtering to JPG only discards significant data.

### 2.2 Full Dataset Used
| Reference Code | Improvement |
|----------------|-------------|
| `code.py` caps at 200 images total | All available images loaded |

**Rationale:** 200 images is insufficient for meaningful training. The full dataset (~12,000+ images) is needed for reliable results.

### 2.3 Proper Shuffling
| Reference Code | Improvement |
|----------------|-------------|
| `random.shuffle(X)` applied to X only (CASIA2code.py line 79) | Paired shuffle using `sklearn.utils.shuffle(X, Y)` |

**Rationale:** Shuffling features without labels destroys the correspondence between images and their ground truth.

---

## 3. ELA Preprocessing Improvements

### 3.1 In-Memory Buffer
| Reference Code | Improvement |
|----------------|-------------|
| Saves temp file to disk: `image.save('resaved.jpg', ...)` | Uses `io.BytesIO()` in-memory buffer |

**Rationale:**
- Avoids filesystem I/O bottleneck
- Prevents race conditions in parallel processing
- No temp file cleanup needed
- Faster execution

### 3.2 Robust Error Handling
| Reference Code | Improvement |
|----------------|-------------|
| Silent `None` return on error | Explicit error logging with image path, skip counter |

**Rationale:** Silently returning `None` causes downstream `NoneType` errors during `resize()`. Explicit handling provides debugging information.

---

## 4. Training Pipeline Improvements

### 4.1 Early Stopping Actually Enabled
| Reference Code | Improvement |
|----------------|-------------|
| Early stopping defined but commented out | Early stopping enabled with `patience=5`, monitoring `val_accuracy` |

**Rationale:** Without early stopping, the model trains for all 40 epochs regardless of overfitting, wasting compute and degrading generalization.

### 4.2 Correct Monitor Name
| Reference Code | Improvement |
|----------------|-------------|
| `monitor='val_acc'` (deprecated) | `monitor='val_accuracy'` (modern Keras) |

**Rationale:** `val_acc` silently fails in TensorFlow 2.x+. The callback never triggers because the metric name doesn't match.

### 4.3 Learning Rate
| Reference Code | Improvement |
|----------------|-------------|
| `lr=0.0001` (both files) | `learning_rate=0.0001` with potential scheduling |

**Rationale:** Using the non-deprecated parameter name.

---

## 5. Evaluation Improvements

### 5.1 Complete Metrics Suite
| Reference Code | Improvement |
|----------------|-------------|
| Accuracy, Precision, Recall only | Accuracy, Precision, Recall, F1 Score, Classification Report |

**Rationale:** F1 score provides a balanced measure; classification report gives per-class breakdown.

### 5.2 Correct Confusion Matrix
| Reference Code | Improvement |
|----------------|-------------|
| `code.py`: 10-class tick labels for 2-class problem | Proper 2×2 matrix with "Authentic"/"Tampered" labels |
| `CASIA2code.py`: single tick label | Seaborn heatmap with annotations |

**Rationale:** Confusion matrix must visually match the actual classification task.

### 5.3 Training Curves
| Reference Code | Improvement |
|----------------|-------------|
| Basic matplotlib plots with typos (`accuarcy`) | Clean, publication-quality dual-panel plots |
| Deprecated `axes=` parameter | Standard matplotlib subplot API |

---

## 6. Code Quality Improvements

### 6.1 Modern Keras API
| Reference Code | Improvement |
|----------------|-------------|
| Mixed `keras.*` and `tensorflow.keras.*` imports | Consistent `tensorflow.keras.*` throughout |
| `from keras.utils.np_utils import to_categorical` | `from tensorflow.keras.utils import to_categorical` |

### 6.2 Reproducibility
| Reference Code | Improvement |
|----------------|-------------|
| No seed setting | `np.random.seed(42)`, `tf.random.set_seed(42)`, `random.seed(42)` |

### 6.3 Memory Management
| Reference Code | Improvement |
|----------------|-------------|
| Flatten-then-reshape approach | Direct array storage in correct shape |
| All data in RAM at once | Explicit `del` of unused large arrays after splitting |

### 6.4 Configuration Centralization
| Reference Code | Improvement |
|----------------|-------------|
| Hard-coded values scattered throughout | All hyperparameters in a single configuration cell |

```python
# Configuration
IMAGE_SIZE = (128, 128)
ELA_QUALITY = 90
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
```

---

## 7. Summary of Critical Fixes

| # | Bug | Severity | Fix |
|---|-----|----------|-----|
| 1 | Wrong image size (150 vs 128) | High | Set IMAGE_SIZE = (128, 128) |
| 2 | Sigmoid output on 2 units | Critical | Changed to softmax |
| 3 | Softmax on 1 unit (code.py) | Fatal | Changed to 2 units + softmax |
| 4 | binary_crossentropy with one-hot | High | Changed to categorical_crossentropy |
| 5 | Dense(150) vs Dense(256) | Medium | Changed to 256 |
| 6 | val_acc monitor name | Medium | Changed to val_accuracy |
| 7 | Early stopping disabled | Medium | Enabled with proper config |
| 8 | JPG-only loading | Medium | All formats supported |
| 9 | 200-image cap | Critical | Full dataset used |
| 10 | X shuffled without Y | Medium | Paired shuffle |
