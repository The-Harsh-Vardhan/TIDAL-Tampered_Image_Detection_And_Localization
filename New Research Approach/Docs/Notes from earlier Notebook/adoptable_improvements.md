# Adoptable Improvements from vK.12.0

| Field | Value |
|-------|-------|
| Source | vK.12.0 (cell references below) |
| Target | vR.1.x (Track 1, TensorFlow/Keras) and vR.P.x (Track 2, PyTorch) |
| Constraint | Must NOT change any ablation variable (model, optimizer, loss, data, preprocessing) |
| Date | 2026-03-15 |

---

## Integration Rules

1. These improvements add evaluation, visualization, or reproducibility cells ONLY
2. They do NOT modify training configuration, architecture, or data pipeline
3. They can be added to any version without creating a new ablation version
4. TF/Keras implementations provided; PyTorch equivalents noted where relevant

---

## Tier 1 — Add to Next Notebook (High Value, Low Effort)

### 1. Data Leakage Verification

| Field | Value |
|-------|-------|
| vK.12.0 cell | 38 |
| Why useful | Asserts zero file overlap between train/val/test splits. Catches split errors automatically. |
| Where to integrate | After train/val/test split cell |
| TF/Keras implementation | 3 lines |

```python
assert len(set(train_paths) & set(val_paths)) == 0, "LEAK: train ∩ val"
assert len(set(train_paths) & set(test_paths)) == 0, "LEAK: train ∩ test"
assert len(set(val_paths) & set(test_paths)) == 0, "LEAK: val ∩ test"
print("Data leakage check: PASSED (zero overlap)")
```

---

### 2. Tampered-Only Metric Highlighting

| Field | Value |
|-------|-------|
| vK.12.0 cell | 64 |
| Why useful | Authentic images are trivially correct (no forgery to detect). The tampered-class metrics are the true measure of forensic ability. Current notebooks show per-class metrics but don't emphasize this distinction. |
| Where to integrate | After classification report, as a "Forensic Performance" subsection |
| TF/Keras implementation | 5 lines |

```python
tp_mask = y_true == 1
tp_prec = precision_score(y_true, y_pred, pos_label=1)
tp_rec = recall_score(y_true, y_pred, pos_label=1)
tp_f1 = f1_score(y_true, y_pred, pos_label=1)
print(f"FORENSIC DETECTION:  Precision={tp_prec:.4f}  Recall={tp_rec:.4f}  F1={tp_f1:.4f}")
```

---

### 3. Training Curves with Best-Epoch Marker

| Field | Value |
|-------|-------|
| vK.12.0 cell | 12, 77 |
| Why useful | A vertical line at the best epoch (where early stopping restored weights) makes it immediately clear which epoch produced the reported metrics. |
| Where to integrate | Replace existing training curve cell |
| TF/Keras implementation | 2 extra lines per subplot |

```python
best_epoch = np.argmin(history.history['val_loss'])
ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')
ax.scatter(best_epoch, history.history['val_loss'][best_epoch], color='red', s=100, zorder=5)
```

---

### 4. Seed Verification Cell

| Field | Value |
|-------|-------|
| vK.12.0 cell | 133 |
| Why useful | Verifies seeds are actually set. Catches environments where seed setting silently fails. |
| Where to integrate | After configuration cell, before data loading |
| TF/Keras implementation | 3 lines |

```python
tf.random.set_seed(SEED)
test_tensor = tf.random.uniform([5], seed=SEED)
print(f"Seed verification: SEED={SEED}, sample={test_tensor.numpy()[:3]}")
```

---

### 5. Split Determinism Verification

| Field | Value |
|-------|-------|
| vK.12.0 cell | 135 |
| Why useful | Hashes file paths per split — if hashes match across runs, the split is deterministic. Catches non-determinism from OS file listing order. |
| Where to integrate | After train/val/test split |
| TF/Keras implementation | 4 lines |

```python
import hashlib
for name, paths in [("Train", train_paths), ("Val", val_paths), ("Test", test_paths)]:
    h = hashlib.md5(str(sorted(paths)).encode()).hexdigest()[:8]
    print(f"  {name} hash: {h} ({len(paths)} samples)")
```

---

### 6. Environment Info Logger

| Field | Value |
|-------|-------|
| vK.12.0 cell | 141 |
| Why useful | Records TF version, GPU name, Python version, etc. Critical for reproducing results across Kaggle sessions with different Docker images. |
| Where to integrate | First code cell, after imports |
| TF/Keras implementation | Already partially done (TF version + GPU). Enhancement: |

```python
import platform
print(f"Python:     {platform.python_version()}")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy:      {np.__version__}")
print(f"Sklearn:    {sklearn.__version__}")
print(f"GPU:        {tf.config.list_physical_devices('GPU')}")
print(f"Platform:   {platform.platform()}")
```

---

### 7. Dataset Summary Table

| Field | Value |
|-------|-------|
| vK.12.0 cell | 36 |
| Why useful | Formatted table of class counts and ratios per split. More informative than raw count prints. |
| Where to integrate | After data loading and splitting |
| TF/Keras implementation | 6 lines |

```python
print(f"{'Split':<8} {'Total':>6} {'Au':>6} {'Tp':>6} {'Ratio':>8}")
print("-" * 38)
for name, y in [("Train", Y_train), ("Val", Y_val), ("Test", Y_test)]:
    au = np.sum(y == 0); tp = np.sum(y == 1)
    print(f"{name:<8} {len(y):>6} {au:>6} {tp:>6} {au/max(tp,1):>8.2f}")
```

---

## Tier 2 — Evaluation Enhancements (Medium Effort, High Diagnostic Value)

### 8. Threshold Sweep Optimization

| Field | Value |
|-------|-------|
| vK.12.0 cell | 79 |
| Why useful | The default 0.5 threshold may not be optimal. Sweeping thresholds on the validation set and applying the best to the test set gives a more calibrated decision boundary. |
| Where to integrate | After ROC curve, before final metrics table |
| Critical rule | Find threshold on VALIDATION set, report on TEST set. Never optimize on test. |

```python
thresholds = np.arange(0.10, 0.91, 0.01)
val_f1s = [f1_score(y_val_true, (val_probs[:, 1] >= t).astype(int)) for t in thresholds]
optimal_t = thresholds[np.argmax(val_f1s)]
print(f"Optimal threshold (val): {optimal_t:.2f} (F1={max(val_f1s):.4f})")

# Apply to test set
y_pred_opt = (test_probs[:, 1] >= optimal_t).astype(int)
print(f"Test F1 @ 0.50: {f1_score(y_test, y_pred_default):.4f}")
print(f"Test F1 @ {optimal_t:.2f}: {f1_score(y_test, y_pred_opt):.4f}")
```

---

### 9. Per-Forgery-Type Evaluation

| Field | Value |
|-------|-------|
| vK.12.0 cell | 87 |
| Why useful | CASIA v2.0 filenames encode forgery type: `Tp_D_` = different-source splicing, `Tp_S_` = same-source copy-move. Breaking down reveals which attack type the model handles better. |
| Where to integrate | New subsection after confusion matrix |
| Prerequisite | Must retain original filenames alongside predictions |

```python
tp_test_mask = y_true == 1
tp_fnames = [test_paths[i] for i in range(len(y_true)) if y_true[i] == 1]
tp_preds = y_pred[tp_test_mask]
tp_true = y_true[tp_test_mask]

splice = [os.path.basename(f).startswith('Tp_D_') for f in tp_fnames]
copymove = [os.path.basename(f).startswith('Tp_S_') for f in tp_fnames]

for name, mask in [("Splicing", splice), ("Copy-move", copymove)]:
    if sum(mask) > 0:
        m = np.array(mask)
        acc = accuracy_score(tp_true[m], tp_preds[m])
        print(f"  {name}: {sum(mask)} images, accuracy={acc:.4f}")
```

---

### 10. Worst-N Failure Case Analysis

| Field | Value |
|-------|-------|
| vK.12.0 cell | 110 |
| Why useful | Shows the N images where the model was most confidently wrong. Reveals systematic failure patterns (e.g., all failures are TIF files, or all are small tampered regions). |
| Where to integrate | New cell after confusion matrix |

```python
wrong_mask = y_pred != y_true
wrong_conf = y_pred_probs[wrong_mask, y_pred[wrong_mask]]
worst_idx = np.argsort(wrong_conf)[-10:]  # Top 10 most confident mistakes

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Top 10 Most Confident Mistakes', fontweight='bold')
for ax, idx in zip(axes.flat, worst_idx):
    ax.imshow(X_test[idx])
    ax.set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}\nConf: {wrong_conf[idx]:.2%}", fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

### 11. FP/FN Error Analysis

| Field | Value |
|-------|-------|
| vK.12.0 cell | 112 |
| Why useful | Separately visualize false positives (authentic classified as tampered) and false negatives (tampered classified as authentic). Diagnoses whether failures are biased toward one error type. |
| Where to integrate | After worst-N analysis |

```python
fp_mask = (y_true == 0) & (y_pred == 1)  # Authentic → Tampered
fn_mask = (y_true == 1) & (y_pred == 0)  # Tampered → Authentic
print(f"False Positives: {fp_mask.sum()} | False Negatives: {fn_mask.sum()}")
# Display top-5 of each with ELA maps and confidence scores
```

---

### 12. Color-Coded Prediction Borders

| Field | Value |
|-------|-------|
| vK.12.0 cell | 14 |
| Why useful | Green border for correct predictions, red border for incorrect. Instantly shows correctness in prediction grids. |
| Where to integrate | Enhance existing sample predictions visualization |

```python
color = 'green' if y_true[i] == y_pred[i] else 'red'
for spine in ax.spines.values():
    spine.set_edgecolor(color)
    spine.set_linewidth(3)
```

---

## Tier 3 — Final Notebook Sections (Higher Effort, Research Value)

### 13. Robustness Testing Suite

| Field | Value |
|-------|-------|
| vK.12.0 cell | 116 |
| Why useful | Tests model on degraded images: JPEG re-compression at QF 70 and 50, Gaussian noise (sigma=10, 25), Gaussian blur (3×3, 5×5), downscale-upscale (0.75×). Reveals how fragile ELA-based detection is to post-tampering processing. |
| Where to integrate | Final section of notebook |
| Estimated effort | 30 lines |

Degradation conditions:

| Condition | Implementation |
|-----------|---------------|
| Clean | No modification (baseline) |
| JPEG QF 70 | `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])` → decode |
| JPEG QF 50 | Same with quality=50 |
| Gaussian noise σ=10 | `img + np.random.normal(0, 10/255, img.shape)` |
| Gaussian noise σ=25 | Same with sigma=25 |
| Gaussian blur 3×3 | `cv2.GaussianBlur(img, (3,3), 0)` |
| Gaussian blur 5×5 | `cv2.GaussianBlur(img, (5,5), 0)` |
| Resize 0.75× | Downscale then upscale back |

---

### 14. Inference Speed Benchmarking

| Field | Value |
|-------|-------|
| vK.12.0 cell | 120 |
| Why useful | Reports images/second throughput. Useful for comparing model complexity across ablation versions. |
| Where to integrate | After robustness testing |

```python
import time
times = []
for _ in range(3):  # 3 runs
    start = time.time()
    model.predict(X_test[:100], verbose=0)
    times.append(time.time() - start)
mean_t = np.mean(times)
print(f"Throughput: {100/mean_t:.1f} images/sec ({mean_t/100*1000:.1f} ms/image)")
```

---

### 15. Shortcut Learning Detection

| Field | Value |
|-------|-------|
| vK.12.0 cell | 91 |
| Why useful | Tests whether the model relies on actual image content vs dataset artifacts. Shuffle labels and verify accuracy drops to ~50%. |
| Where to integrate | Diagnostic section |

```python
shuffled = np.random.permutation(y_true)
random_acc = np.mean(y_pred == shuffled)
print(f"Accuracy on shuffled labels: {random_acc:.4f} (expect ~0.50)")
print(f"Accuracy on real labels:     {acc:.4f}")
print(f"Delta: {acc - random_acc:.4f} → Model uses real features" if acc - random_acc > 0.3 else "WARNING: possible shortcut")
```

---

### 16. ELA Heatmap with Hot Colormap Overlay

| Field | Value |
|-------|-------|
| vK.12.0 cell | 103 |
| Why useful | Overlays ELA signal as a hot-colormap on the original image. More visually informative than raw ELA for presentations. |
| Where to integrate | Enhance existing ELA visualization section |

```python
import cv2
ela_gray = np.mean(ela_img, axis=2)  # Average channels
heatmap = cv2.applyColorMap((ela_gray * 255).astype(np.uint8), cv2.COLORMAP_HOT)
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
overlay = 0.6 * original + 0.4 * heatmap_rgb
```

---

### 17–21. Additional Infrastructure

| # | Improvement | vK.12.0 Cell | Integration Point |
|---|------------|-------------|-------------------|
| 17 | Three-file checkpoint (last, best, periodic) | 66 | Training callbacks |
| 18 | Save training history as JSON | 66 | After training |
| 19 | Contour overlays on predictions | 108 | Track 2 visualization |
| 20 | Cross-version automated comparison | 93 | Final cell |
| 21 | Mask coverage histogram + CDF | 51 | Track 2 dataset analysis |

---

## Total Effort Estimate

| Tier | Items | Lines of Code | Priority |
|------|-------|---------------|----------|
| Tier 1 | 7 | ~30 | Add to vR.1.3+ immediately |
| Tier 2 | 5 | ~50 | Add to evaluation section |
| Tier 3 | 9 | ~80 | Add as time permits |
| **Total** | **21** | **~160** | |

All additions are non-variable evaluation/infrastructure cells that do not affect the ablation study.
