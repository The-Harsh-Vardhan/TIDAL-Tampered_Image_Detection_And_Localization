"""Build vR.1.2 notebook from vR.1.1 source with data augmentation changes."""
import json
import copy

# Load vR.1.1 notebook
with open(r'c:\D Drive\Projects\BigVision Assignment\New Research Approach\vR.1.1 Image Detection and Localisation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def get_source(cell):
    return ''.join(cell['source'])

def set_source(cell, text):
    cell['source'] = text.split('\n')
    # Convert to proper notebook format (each line ends with \n except last)
    lines = text.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]]
    if lines:
        cell['source'].append(lines[-1])

# Clear all outputs and execution counts
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []

# ============================================================
# Cell 0 (markdown): Update title and introduction
# ============================================================
set_source(cells[0], """# vR.1.2 — ETASR Ablation Study: Data Augmentation

**Base paper:** Gorle, R. & Guttavelli, A. (2025). *Enhanced Image Tampering Detection using Error Level Analysis and a CNN.* ETASR, Vol. 15, No. 1.

**Parent version:** vR.1.1 (evaluation fix — honest baseline)

**Change in this version:** Add real-time data augmentation
- Horizontal flip (50% probability)
- Vertical flip (50% probability)
- Random rotation ±15 degrees
- Applied ONLY during training (val/test unchanged)

**Pipeline:** `Raw Image -> RGB -> ELA (Q=90) -> Resize 128x128 -> Normalize [0,1] -> Augment (train only) -> CNN -> Softmax -> {Authentic, Tampered}`

---

### Table of Contents

1. [Version Change Log](#1-version-change-log)
2. [Imports and Configuration](#2-imports-and-configuration)
3. [Dataset Preparation](#3-dataset-preparation)
4. [ELA Preprocessing](#4-ela-preprocessing)
5. [ELA Visualization](#5-ela-visualization)
6. [Data Splitting](#6-data-splitting)
7. [Model Architecture](#7-model-architecture)
8. [Training Pipeline](#8-training-pipeline)
9. [Test Set Evaluation](#9-test-set-evaluation)
10. [Results Visualization](#10-results-visualization)
11. [Ablation Comparison](#11-ablation-comparison)
12. [Discussion](#12-discussion)""")

# ============================================================
# Cell 1 (markdown): Update version change log
# ============================================================
set_source(cells[1], """---

## 1. Version Change Log

| Parameter | vR.1.1 (Parent) | vR.1.2 (This Version) |
|-----------|-----------------|----------------------|
| Data augmentation | None | **H-flip, V-flip, Rotation ±15°** |
| Training method | `model.fit(X_train, ...)` | **`model.fit(train_generator, ...)`** |
| Steps per epoch | Auto (len/batch) | **`len(X_train) // BATCH_SIZE`** |

### What DID NOT change (frozen from vR.1.1)
- ELA quality: 90
- Image size: 128x128
- CNN architecture: 2xConv2D(32) + Dense(256) + Dense(2)
- Optimizer: Adam(lr=0.0001)
- Loss: categorical_crossentropy
- Batch size: 32
- Early stopping: patience=5 on val_accuracy
- Seed: 42
- Data split: 70/15/15 train/val/test (stratified)
- Evaluation: per-class + macro metrics, ROC-AUC on test set

### Cumulative changes from baseline (vR.1.0)
1. **vR.1.1:** 70/15/15 split, per-class metrics, ROC-AUC, ELA viz, model save
2. **vR.1.2:** Data augmentation (this version)""")

# ============================================================
# Cell 2 (code): Update imports and configuration
# ============================================================
set_source(cells[2], """# ============================================================
# 2.1 — Imports and Configuration
# ============================================================

import os
import random
import warnings
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageChops, ImageEnhance
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # NEW in vR.1.2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- Version Info ----
VERSION = 'vR.1.2'
CHANGE = 'Data augmentation: horizontal flip, vertical flip, rotation +/-15 degrees'

# ---- Hyperparameters (FROZEN from baseline) ----
IMAGE_SIZE = (128, 128)
ELA_QUALITY = 90
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
EARLY_STOP_PATIENCE = 5

# ---- Split ratios (FROZEN from vR.1.1) ----
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

# ---- NEW in vR.1.2: Augmentation parameters ----
AUGMENTATION = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation_range': 15,
    'fill_mode': 'nearest'
}

print(f'Version: {VERSION}')
print(f'Change:  {CHANGE}')
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {len(tf.config.list_physical_devices("GPU")) > 0}')
if len(tf.config.list_physical_devices("GPU")) > 0:
    print(f'GPU: {tf.config.list_physical_devices("GPU")[0]}')
print(f'\\nConfiguration (frozen from baseline):')
print(f'  Image size:    {IMAGE_SIZE}')
print(f'  ELA quality:   {ELA_QUALITY}')
print(f'  Batch size:    {BATCH_SIZE}')
print(f'  Max epochs:    {EPOCHS}')
print(f'  Learning rate: {LEARNING_RATE}')
print(f'  Early stop:    patience={EARLY_STOP_PATIENCE}')
print(f'  Split:         70% train / 15% val / 15% test')
print(f'\\nNEW in this version:')
print(f'  Augmentation:  {AUGMENTATION}')""")

# ============================================================
# Cell 15 (markdown): Update training pipeline description
# ============================================================
set_source(cells[15], """---

## 8. Training Pipeline

**CHANGED in vR.1.2:** Data augmentation added to training.

- Optimizer: Adam (lr=0.0001) — frozen
- Loss: categorical_crossentropy — frozen
- Early stopping: val_accuracy, patience=5, restore_best_weights — frozen
- Batch size: 32 — frozen
- **NEW: ImageDataGenerator with horizontal flip, vertical flip, rotation ±15°**
- Augmentation applied ONLY to training batches. Validation data is NOT augmented.""")

# ============================================================
# Cell 17 (code): Replace training with augmented training
# ============================================================
set_source(cells[17], """# ============================================================
# 8.2 — Train the Model (CHANGED: with augmentation)
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

# NEW in vR.1.2: Data augmentation (training only)
train_datagen = ImageDataGenerator(
    horizontal_flip=AUGMENTATION['horizontal_flip'],
    vertical_flip=AUGMENTATION['vertical_flip'],
    rotation_range=AUGMENTATION['rotation_range'],
    fill_mode=AUGMENTATION['fill_mode']
)

train_generator = train_datagen.flow(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    seed=SEED
)

steps_per_epoch = len(X_train) // BATCH_SIZE

print(f'Training on {X_train.shape[0]} samples (with augmentation), validating on {X_val.shape[0]} samples')
print(f'Test set ({X_test.shape[0]} samples) held out — NOT used during training')
print(f'Batch size: {BATCH_SIZE}, Steps/epoch: {steps_per_epoch}, Max epochs: {EPOCHS}')
print(f'Augmentation: H-flip={AUGMENTATION["horizontal_flip"]}, V-flip={AUGMENTATION["vertical_flip"]}, Rotation=+/-{AUGMENTATION["rotation_range"]}deg')
print('=' * 60)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping],
    verbose=1
)

print(f'\\nTraining complete. Epochs run: {len(history.history["loss"])}')""")

# ============================================================
# Cell 28 (code): Update ablation tracking table
# ============================================================
set_source(cells[28], """# ============================================================
# 11.1 — Ablation Results Tracking
# ============================================================

print('=' * 100)
print(f'  ABLATION STUDY — RESULTS TRACKING TABLE')
print('=' * 100)
print()
print(f'  {"Version":<10} {"Change":<30} {"Test Acc":>9} {"Tp Prec":>8} {"Tp Rec":>7} {"Tp F1":>7} {"Macro F1":>9} {"AUC":>6} {"Epochs":>7}')
print(f'  {"-"*95}')
print(f'  {"vR.1.0":<10} {"Baseline (val metrics)":<30} {"89.89%*":>9} {"0.8279*":>8} {"0.9483*":>7} {"0.8840*":>7} {"0.8972*":>9} {"  ---":>6} {"13 (8)":>7}')
print(f'  {"vR.1.1":<10} {"Eval fix":<30} {"88.38%":>9} {"0.8393":>8} {"0.8830":>7} {"0.8606":>7} {"0.8805":>9} {"0.9601":>6} {"13 (8)":>7}')
print(f'  {"vR.1.2":<10} {"Augmentation (this run)":<30} {test_acc:>8.2%} {test_prec_per[1]:>8.4f} {test_rec_per[1]:>7.4f} {test_f1_per[1]:>7.4f} {test_f1_macro:>9.4f} {test_roc_auc:>6.4f} {str(len(history.history["loss"])):>7}')
print()
print('  * vR.1.0 metrics are on validation set (biased). Not directly comparable.')
print('=' * 100)""")

# ============================================================
# Cell 29 (markdown): Update discussion
# ============================================================
set_source(cells[29], """---

## 12. Discussion

### Change in This Version

**Data augmentation** was added to the training pipeline using `ImageDataGenerator`:
- Horizontal flip (50% probability)
- Vertical flip (50% probability)
- Random rotation ±15 degrees
- Fill mode: nearest

Augmentation is applied ONLY during training. Validation and test sets are NOT augmented, ensuring unbiased evaluation.

### Why This Change

The vR.1.1 audit identified severe overfitting:
- Train-val accuracy gap: 4.25pp at best epoch, expanding to 13pp by epoch 13
- Val loss collapse at epochs 12–13 (loss nearly tripled)
- FN rate doubled from ~5% to ~12% with the reduced training set

Data augmentation creates diverse training views without increasing dataset size or RAM usage, targeting these specific overfitting symptoms.

### Expected Impact

- Test accuracy: +0.5–2.5pp improvement
- Reduced overfitting gap (train-val accuracy closer together)
- FN rate improvement (better tampered recall)
- Potentially longer training (augmentation slows convergence = more exploration)
- Stabilized late-epoch training (reduced val collapse)

### Next Version: vR.1.3

The next ablation will add **class weights** (inversely proportional to class frequency) to address the 1.46:1 class imbalance between Authentic and Tampered images.""")

# ============================================================
# Cell 30 (code): Update model save filename
# ============================================================
set_source(cells[30], """# ============================================================
# 12.1 — Save Model Weights
# ============================================================

model.save(f'{VERSION}_ela_cnn_model.keras')
print(f'Model saved: {VERSION}_ela_cnn_model.keras')
print(f'Model parameters: {model.count_params():,}')""")

# Save the notebook
output_path = r'c:\D Drive\Projects\BigVision Assignment\New Research Approach\vR.1.2 — ETASR Run-01 Image Detection and Localisation.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Notebook saved to: {output_path}')
print(f'Total cells: {len(cells)}')
print(f'Code cells: {sum(1 for c in cells if c["cell_type"] == "code")}')
print(f'Markdown cells: {sum(1 for c in cells if c["cell_type"] == "markdown")}')
