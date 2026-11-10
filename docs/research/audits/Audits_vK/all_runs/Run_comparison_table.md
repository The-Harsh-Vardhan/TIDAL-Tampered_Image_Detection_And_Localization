# Run Comparison Table

**All Experiment Runs: vK.1 through vK.10.6**
**Date:** 2026-03-14

---

## Master Metrics Comparison

| Run | Img Acc | AUC-ROC | Mixed F1 | Tam F1 | Tam IoU | Epochs | Status |
|---|---|---|---|---|---|---|---|
| vK.1 | — | — | — | — | — | — | No run output |
| vK.2 | — | — | — | — | — | — | No run output |
| vK.3 (run-01) | 0.8986 | — | 0.5761¹ | ~0.15–0.25² | — | 50 | Completed |
| **v6.5 (run-01)** | **0.8246** | **0.8703** | **0.7208** | **0.4101** | **0.3563** | **25 (ES)** | Completed |
| **v8 (run-01)** | **0.7190** | **0.8170** | **0.5181** | **0.2949** | **0.2321** | **27 (ES)** | Completed |
| vK.7.1 (run-01) | 0.8986 | — | 0.5761¹ | ~0.15–0.25² | — | 50 | Completed |
| vK.7.5 | — | — | — | — | — | — | **Incomplete** |
| vK.10.3b (run-01) | 0.5061 | 0.6069 | 0.5781¹ | 0.0004 | 0.0002 | ~10 (ES) | Completed |
| vK.10.4 (run-01) | 0.4675 | 0.6534 | 0.5938¹ | 0.0000 | 0.0000 | 10 (ES) | Completed |
| vK.10.5 (run-01) | 0.4791 | 0.6201 | 0.5724¹ | 0.0006 | 0.0003 | ~10 (ES) | Completed |
| vK.10.3b (run-02) | 0.5061 | 0.6069 | 0.5781¹ | 0.0004 | 0.0002 | ~10 (ES) | **Duplicate of run-01** |
| **vK.10.6 (run-01)** | **0.8357** | **0.9057** | **0.4853¹** | **0.2213³** | **0.1554³** | **100** | **Completed** |

¹ Mixed-set F1/Dice inflated by authentic images scoring 1.0 (59.4% of test set).
² Tampered-only metrics not explicitly reported. Estimated from inflation pattern.
³ After threshold optimization (optimal=0.15). Default threshold (0.50) gives Tam-F1=0.1946.

ES = Early Stopped.

---

## Architecture Comparison

| Run | Model | Params | Pretrained | Image Size |
|---|---|---|---|---|
| vK.1–vK.3 | Custom `UNetWithClassifier` | ~15.7M | **No** | 256×256 |
| **v6.5** | **`smp.Unet` (ResNet34)** | **24.4M** | **Yes (ImageNet)** | **384×384** |
| **v8** | **`smp.Unet` (ResNet34)** | **24.4M** | **Yes (ImageNet)** | **384×384** |
| vK.7.1 | Custom `UNetWithClassifier` | ~15.7M | **No** | 256×256 |
| vK.7.5 | Custom `UNetWithClassifier` | ~15.7M | **No** | 256×256 |
| vK.10.3b–10.5 | Custom `UNetWithClassifier` | **31.6M** | **No** | 256×256 |
| **vK.10.6** | Custom `UNetWithClassifier` | **31.6M** | **No** | 256×256 |

**Two architecture tracks exist:** v6.5/v8 use SMP with pretrained ImageNet encoder. The vK.x series uses a custom U-Net trained from scratch.

---

## Training Configuration Comparison

| Run | Optimizer | LR | Eff. Batch | Epochs | AMP | Early Stop | DataParallel | Loss |
|---|---|---|---|---|---|---|---|---|
| vK.3 Blk1 | Adam | 1e-4 | 8 | 30 | No | No | No | CE + BCE |
| vK.3 Blk2 | Adam | 1e-4 | 8 | 50 | No | No | No | Focal + BCE+Dice |
| **v6.5** | **AdamW** | **diff(1e-4/1e-3)** | **16** | **25(ES)** | **Yes** | **Yes(p=10)** | **Yes(2GPU)** | **BCEDice** |
| **v8** | **AdamW** | **diff(1e-4/1e-3)** | **256** | **27(ES)** | **Yes** | **Yes(p=10)** | **Yes(2GPU)** | **BCEDice(pw=30)** |
| vK.7.1 | Adam | 1e-4 | 8 | 50 | No | No | No | Focal + BCE+Dice |
| vK.10.3b | Adam | 1e-4 | 32 | 50 | **Yes** | **Yes(p=10)** | No | BCE + BCE+Dice |
| vK.10.4 | Adam | 1e-4 | 32 | 50 | **Yes** | **Yes(p=10)** | No | BCE + BCE+Dice |
| vK.10.5 | Adam | 1e-4 | 32 | 50 | **Yes** | **Yes(p=10)** | **Yes** | BCE + BCE+Dice |
| **vK.10.6** | Adam | 1e-4 | 32 | **100** | **Yes** | **Yes(p=30)** | **Yes** | Focal + BCE+Dice |

---

## Engineering Quality Comparison

| Feature | vK.1–3 | v6.5 | v8 | vK.7.1 | vK.7.5 | vK.10.3b | vK.10.4 | vK.10.5 | vK.10.6 |
|---|---|---|---|---|---|---|---|---|---|
| CONFIG dict | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** | **Yes** |
| Reproducibility seed | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** | **Yes** |
| AMP | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** | **Yes** |
| Early stopping | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** | **Yes(p=30)** |
| DataParallel | — | **Yes** | **Yes** | — | — | — | — | **Yes** | **Yes** |
| Gradient accumulation | — | **Yes** | **Yes** | — | — | — | — | — | — |
| Differential LR | — | **Yes** | **Yes** | — | — | — | — | — | — |
| LR Scheduler | — | — | **Yes** | — | — | — | — | — | — |
| `get_base_model()` | — | — | — | — | — | — | — | **Yes** | **Yes** |
| Checkpoint (3-file) | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** | **Yes** |
| VRAM auto-scaling | — | — | — | — | — | **Yes** | **Yes** | **Yes** | **Yes** |
| W&B integration | — | **Yes** | **Yes** | Partial | Partial | **Yes** | **Yes** | **Yes** | **Yes** |
| Data leakage check | — | **Yes** | **Yes** | — | — | — | — | — | **Yes** |
| Data Viz section | — | — | — | — | — | — | **Yes** | **Yes** | **Yes** |
| Block 1 data leak | **BUG** | — | — | **BUG** | **BUG** | Fixed | Fixed | Fixed | Fixed |

---

## Evaluation Methodology Comparison

| Feature | vK.3 | v6.5 | v8 | vK.7.1 | vK.10.3b | vK.10.4 | vK.10.5 | vK.10.6 |
|---|---|---|---|---|---|---|---|---|
| Tampered-only metrics | — | **Yes** | **Yes** | — | **Yes** | **Yes** | **Yes** | **Yes** |
| Threshold optimization | — | **Yes** | **Yes** | — | — | — | — | **Yes** |
| Forgery-type breakdown | — | **Yes** | **Yes** | — | — | — | — | **Yes** |
| Mask-size stratification | — | Partial | **Yes** | — | — | — | — | **Yes** |
| Shortcut detection | — | — | **Yes** | — | — | — | — | **Yes** |
| Robustness testing | — | **Yes** | **Yes** | — | — | — | — | **Yes** |
| Grad-CAM | — | **Yes** | **Yes** | — | — | — | — | **Yes** |
| Failure case analysis | — | **Yes** | **Yes** | — | — | — | — | **Yes** |
| Pixel-level AUC | — | — | — | — | — | — | — | **Yes** |
| Confusion matrix | — | — | — | — | — | — | — | **Yes** |
| ROC/PR curves (plotted) | — | — | — | — | — | — | — | **Yes** |

**vK.10.6 is the first run to implement ALL evaluation features**, including confusion matrix, PR curves, and pixel-level AUC — surpassing even v6.5/v8.

---

## Bug Tracker Across Versions

| Bug | vK.1 | vK.2 | vK.3 | v6.5 | v8 | vK.7.1 | vK.10.3b | vK.10.4 | vK.10.5 | vK.10.6 |
|---|---|---|---|---|---|---|---|---|---|---|
| Block 1 data leakage | X | X | X | — | — | X | — | — | — | — |
| Dice inflation (all-sample) | X | X | X | X¹ | X¹ | X | X | X | X | X¹ |
| No pretrained encoder | X | X | X | — | — | X | X | X | X | X |
| No seeding | X | X | X | — | — | X | — | — | — | — |
| Checkpoint on acc not F1/Dice | X | X | X | — | — | X | — | — | — | — |
| No LR scheduler | X | X | X | **X** | — | X | X | X | X | X |
| pos_weight=30 regression | — | — | — | — | **X** | — | — | — | — | — |
| Robustness eval bug (identical F1) | — | — | — | **X** | — | — | — | — | — | — |
| CONFIG/docs mismatch | — | — | — | — | — | — | — | — | — | **X** |
| CosineAnnealing double-cycle | — | — | — | — | — | — | — | — | — | **X** |

¹ v6.5/v8 report both mixed-set AND tampered-only metrics, so the inflation is visible but the honest metric is also available.

**"No pretrained encoder" persists across the entire vK.x series. v6.5/v8 solved this.**

---

## The Story in One Chart

```
Tampered-Only F1 Score Across Runs
(higher is better, 1.0 = perfect)

v6.5   ████████████████████ 0.41  ← BEST RUN (pretrained ResNet34)
v8     ██████████████       0.29  ← regression (pos_weight broke it)
10.6   ███████████          0.22  ← best vK.x (100 epochs, from scratch)
vK.3   ██████████           ~0.20 ← estimated (from scratch)
vK.7.1 ██████████           ~0.20 ← estimated (from scratch)
10.3b  ▏                    0.0004 ← from scratch, early stopped
10.4   ▏                    0.0000 ← ZERO
10.5   ▏                    0.0006 ← from scratch + DataParallel

vK.7.5  [INCOMPLETE - NO DATA]
10.3b-r2 [DUPLICATE OF RUN-01]
```

**The pretrained encoder track (v6.5/v8) still produces ~2× better segmentation than even the best from-scratch run (vK.10.6).**
