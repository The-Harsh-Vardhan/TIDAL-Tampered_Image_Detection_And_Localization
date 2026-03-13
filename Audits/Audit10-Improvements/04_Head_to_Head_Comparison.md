# Head-to-Head Comparison

Side-by-side comparison of all notebook runs and vK.10.3 (code-only, not yet run).

---

## Metrics Comparison

| Metric | v8 Run | vK.3 Run | vK.7.5 Run | vK.10.3 (code) |
|--------|--------|----------|------------|----------------|
| **Epochs trained** | 27 (early stopped) | 50 (full) | 2 | Up to 50 (early stop) |
| **Image-Level Accuracy** | 0.7190 | **0.8986** | 0.5526 | TBD |
| **AUC-ROC** | **0.8170** | Not computed | Not computed | Implemented |
| **Dice (all)** | N/A | 0.5760 | 0.5935 (inflated) | TBD |
| **Tampered-Only F1** | 0.2949 | Not reported | Not reported | Implemented |
| **Tampered-Only Dice** | N/A | N/A | N/A | Implemented (checkpoint criterion) |
| **IoU (all)** | 0.4926 | 0.5528 | 0.5935 (inflated) | TBD |
| **Optimal Threshold** | 0.7500 | 0.5 (fixed) | 0.5 (fixed) | 0.5 (fixed) |

---

## Architecture Comparison

| Aspect | v8 | vK.3 | vK.7.5 | vK.10.3 |
|--------|-----|------|--------|---------|
| **Model** | SMP U-Net ResNet34 | Custom UNetWithClassifier | Custom UNetWithClassifier | Custom UNetWithClassifier |
| **Pretrained** | Yes (ImageNet) | No | No | No |
| **Parameters** | ~24.4M | ~31M | ~31M | ~31M |
| **Image size** | 384x384 | 256x256 | 256x256 | 256x256 |
| **Dual-head** | No (seg only, cls via threshold) | Yes (cls + seg) | Yes (cls + seg) | Yes (cls + seg) |

---

## Training Setup Comparison

| Feature | v8 | vK.3 | vK.7.5 | vK.10.3 |
|---------|-----|------|--------|---------|
| **AMP** | Yes | No | No | Yes |
| **Gradient clipping** | Yes (5.0) | Yes (1.0) | Yes (1.0) | Yes (5.0) |
| **Effective batch size** | 256 (64 x 4 accum) | 8 | 8 | 8-32 (VRAM auto) |
| **Weight decay** | 0 | 0 | 0 | 1e-4 |
| **Scheduler** | ReduceLROnPlateau | CosineAnnealing(T=10) | CosineAnnealing(T=10) | CosineAnnealing(T=50) |
| **Early stopping** | Yes (patience=10) | No | No | Yes (patience=10) |
| **Checkpoint resume** | Yes | No | No | Yes (with history) |
| **Loss** | BCE+Dice (seg only) | Focal+BCE+Dice (dual) | Focal+BCE+Dice (dual) | Focal+BCE+Dice (dual) |
| **pos_weight** | 30.01 (pixel ratio) | Balanced class weights | Balanced class weights | Balanced class weights |
| **Differential LR** | Yes (enc/dec) | No | No | No |
| **Gradient accumulation** | Yes (4 steps) | No | No | No |

---

## Evaluation Features Comparison

| Feature | v8 | vK.3 | vK.7.5 | vK.10.3 |
|---------|-----|------|--------|---------|
| **Tampered-only metrics** | Yes | No | No | Yes |
| **ROC-AUC** | Yes | No | No | Yes |
| **Threshold sweep** | Yes (15 points) | No | No | No |
| **Mask-size stratification** | Yes (4 buckets) | No | No | No |
| **Forgery-type breakdown** | Yes (splice/copy-move) | No | No | No |
| **Robustness testing** | Yes (8 conditions) | No | No | No |
| **Grad-CAM explainability** | Yes | No | No | No |
| **Shortcut learning checks** | Yes (2 tests) | No | No | No |
| **Data leakage verification** | Yes | No | No | No |
| **Failure case analysis** | Yes (worst 10) | No | No | No |
| **Confusion matrix** | No | No | No | No |
| **Artifact inventory** | Yes | No | No | No |
| **4-panel visualization** | Yes | Yes | Yes | Yes |
| **Training curves** | Yes (2x2) | Yes | Yes (2 pts) | Yes (2x2 + train_dice) |

---

## Documentation Comparison

| Aspect | v8 | vK.3 | vK.7.5 | vK.10.3 |
|--------|-----|------|--------|---------|
| **CONFIG dict** | Yes | No (scattered) | No (scattered) | Yes |
| **TOC** | No | No | Yes | Yes |
| **Assignment alignment notes** | Partial | No | Yes (every section) | Yes |
| **Docstrings** | Minimal | Structured | Structured | Minimal |
| **Collapsible sections** | No | No | No | Yes |
| **Version changelog** | Yes (v6.5→v8) | No | No | Partial (title only) |

---

## Overall Ranking

| Rank | Notebook | Score | Strength | Weakness |
|------|----------|-------|----------|----------|
| 1 | **v8 Run** | 82/100 | Best evaluation methodology | Mediocre localization |
| 2 | **vK.10.3** (code) | 75/100 | Best engineering foundation | Not yet run, no advanced eval |
| 3 | **vK.3 Run** | 65/100 | Best classification results | No engineering refinements |
| 4 | **vK.7.5 Run** | 30/100 | Best documentation polish | Untrained, broken metrics |
