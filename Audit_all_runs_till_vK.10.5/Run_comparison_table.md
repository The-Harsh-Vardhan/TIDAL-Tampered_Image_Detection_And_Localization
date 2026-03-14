# Run Comparison Table

**All Experiment Runs: vK.1 through vK.10.5**
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

¹ Mixed-set F1/Dice inflated by authentic images scoring 1.0 (59.4% of test set).
² Tampered-only metrics not explicitly reported. Estimated from inflation pattern.

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

---

## Engineering Quality Comparison

| Feature | vK.1–3 | v6.5 | v8 | vK.7.1 | vK.7.5 | vK.10.3b | vK.10.4 | vK.10.5 |
|---|---|---|---|---|---|---|---|---|
| CONFIG dict | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** |
| Reproducibility seed | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** |
| AMP | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** |
| Early stopping | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** |
| DataParallel | — | **Yes** | **Yes** | — | — | — | — | **Yes** |
| Gradient accumulation | — | **Yes** | **Yes** | — | — | — | — | — |
| Differential LR | — | **Yes** | **Yes** | — | — | — | — | — |
| LR Scheduler | — | — | **Yes** | — | — | — | — | — |
| `get_base_model()` | — | — | — | — | — | — | — | **Yes** |
| Checkpoint (3-file) | — | **Yes** | **Yes** | — | — | **Yes** | **Yes** | **Yes** |
| VRAM auto-scaling | — | — | — | — | — | **Yes** | **Yes** | **Yes** |
| W&B integration | — | **Yes** | **Yes** | Partial | Partial | **Yes** | **Yes** | **Yes** |
| Data leakage check | — | **Yes** | **Yes** | — | — | — | — | — |
| Data Viz section | — | — | — | — | — | — | **Yes** | **Yes** |
| Block 1 data leak | **BUG** | — | — | **BUG** | **BUG** | Fixed | Fixed | Fixed |

---

## Evaluation Methodology Comparison

| Feature | vK.3 | v6.5 | v8 | vK.7.1 | vK.10.3b | vK.10.4 | vK.10.5 |
|---|---|---|---|---|---|---|---|
| Tampered-only metrics | — | **Yes** | **Yes** | — | **Yes** | **Yes** | **Yes** |
| Threshold optimization | — | **Yes** | **Yes** | — | — | — | — |
| Forgery-type breakdown | — | **Yes** | **Yes** | — | — | — | — |
| Mask-size stratification | — | Partial | **Yes** | — | — | — | — |
| Shortcut detection | — | — | **Yes** | — | — | — | — |
| Robustness testing | — | **Yes** | **Yes** | — | — | — | — |
| Grad-CAM | — | **Yes** | **Yes** | — | — | — | — |
| Failure case analysis | — | **Yes** | **Yes** | — | — | — | — |
| Confusion matrix | — | — | — | — | — | — | — |
| ROC/PR curves (plotted) | — | — | — | — | — | — | — |

**v6.5 and v8 have the most comprehensive evaluation suites.** Confusion matrix and PR curves are missing from all runs.

---

## Bug Tracker Across Versions

| Bug | vK.1 | vK.2 | vK.3 | v6.5 | v8 | vK.7.1 | vK.10.3b | vK.10.4 | vK.10.5 |
|---|---|---|---|---|---|---|---|---|---|
| Block 1 data leakage | X | X | X | — | — | X | — | — | — |
| Dice inflation (all-sample) | X | X | X | X¹ | X¹ | X | X | X | X |
| No pretrained encoder | X | X | X | — | — | X | X | X | X |
| No seeding | X | X | X | — | — | X | — | — | — |
| Checkpoint on acc not F1/Dice | X | X | X | — | — | X | — | — | — |
| No LR scheduler | X | X | X | **X** | — | X | X | X | X |
| pos_weight=30 regression | — | — | — | — | **X** | — | — | — | — |
| Robustness eval bug (identical F1) | — | — | — | **X** | — | — | — | — | — |

¹ v6.5/v8 report both mixed-set AND tampered-only metrics, so the inflation is visible but the honest metric is also available.

**"No pretrained encoder" persists across the entire vK.x series. v6.5/v8 solved this.**

---

## The Story in One Chart

```
Tampered-Only F1 Score Across Runs
(higher is better, 1.0 = perfect)

v6.5   ████████████████████ 0.41  ← BEST RUN (pretrained ResNet34)
v8     ██████████████       0.29  ← regression (pos_weight broke it)
vK.3   ██████████           ~0.20 ← estimated (from scratch)
vK.7.1 ██████████           ~0.20 ← estimated (from scratch)
10.3b  ▏                    0.0004 ← from scratch, 31.6M params
10.4   ▏                    0.0000 ← ZERO
10.5   ▏                    0.0006 ← from scratch + DataParallel

vK.7.5  [INCOMPLETE - NO DATA]
```

**The pretrained encoder track (v6.5/v8) produces 100–680× better segmentation than the from-scratch track (vK.10.x).**
