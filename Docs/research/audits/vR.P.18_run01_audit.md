# vR.P.18 Run-01 Audit Report — JPEG Compression Robustness Testing

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Notebook** | `Runs/vr-p-18-jpeg-compression-robustness-testing-run-01.ipynb` |
| **Parent** | vR.P.3 (ELA Q=90 RGB, Pixel F1 = 0.6920) |
| **Type** | Evaluation-only (no training) |
| **Verdict** | **INVALID — P.3 checkpoint not found, all results from untrained model** |

---

## PART 1 — Experiment Summary

### What This Experiment Does
P.18 is a **pure evaluation notebook** — no training occurs. It loads the pre-trained P.3 model checkpoint and evaluates it under 5 test conditions to measure robustness to JPEG recompression:

1. **Original** — no recompression
2. **Q=95** — mild compression
3. **Q=90** — moderate (same Q used for ELA computation)
4. **Q=80** — significant
5. **Q=70** — heavy compression

For each condition: recompress test image → compute ELA at Q=90 → normalize → inference → metrics.

### Research Hypothesis
JPEG recompression at progressively lower quality factors will degrade ELA-based tamper detection, revealing how robust the P.3 model is to real-world image sharing (social media typically recompresses at Q=75–85).

### CRITICAL FAILURE: Checkpoint Not Found

```
WARNING: No P.3 checkpoint found!
  Falling back to ImageNet pretrained weights + frozen body + BN unfrozen.
  Results will NOT represent P.3 performance.
```

The P.3 model `.pth` file was **not attached as a Kaggle dataset input**. The notebook fell back to a freshly initialized model with ImageNet encoder weights and random decoder weights. **ALL results are from an untrained model.**

---

## PART 2 — Pipeline Audit

### Dataset Handling
| Aspect | Status |
|--------|--------|
| Dataset | CASIA v2.0 sagnikkayalcse52 — PASS |
| Split | 70/15/15 stratified (Train: 8829, Val: 1892, Test: 1893) — PASS |
| GT masks | 5123/5123 matched — PASS |
| Data leakage | None detected — PASS |

### Preprocessing
| Aspect | Status |
|--------|--------|
| ELA Q=90 | Correctly applied after recompression — PASS |
| ELA normalization | Uses training-set stats (mean=[0.0497, 0.0418, 0.0590]) — PASS |
| JPEG recompression | PIL save to BytesIO, reopen — PASS |
| Quality levels | [95, 90, 80, 70] — reasonable range |

### Model Architecture
| Aspect | Status |
|--------|--------|
| Model | UNet + ResNet-34 | PASS |
| Total params | 24,436,369 | PASS |
| Mode | eval (no training) | PASS |
| **Checkpoint** | **NOT FOUND — CRITICAL FAILURE** |

### Evaluation
| Aspect | Status |
|--------|--------|
| Pixel metrics | F1, IoU, AUC, Precision, Recall — all computed — PASS |
| Image metrics | Accuracy, Macro F1, ROC-AUC — all computed — PASS |
| Confusion matrices | All 5 conditions — PASS |
| Visualizations | Degradation curves, prediction grids — PASS |

---

## PART 3 — Code Quality Roast

### Strengths
1. **Clean experiment design** — Pure evaluation with no training, clear condition loop
2. **Complete metrics** — Per-condition pixel + image metrics, confusion matrices, visualizations
3. **Good robustness framework** — Degradation curves, delta table, cross-track comparison
4. **All cells executed cleanly** — Zero errors

### Critical Issues

| Severity | Issue | Impact |
|----------|-------|--------|
| **FATAL** | P.3 checkpoint not found, fell back to ImageNet weights | **All results invalid — untrained model** |
| **HIGH** | No validation that checkpoint loaded successfully before proceeding | Notebook silently continues with garbage model |
| **MEDIUM** | Warning message is just `print()`, not `raise RuntimeError()` | Should abort execution when checkpoint missing |

### Why the Results Are Garbage
- Pixel F1 = 0.0362 (vs expected ~0.69 from P.3) — **19× too low**
- Image Accuracy = 40.62% = 769/1893 — exactly the tampered class proportion
- Confusion matrix: TN=0, FP=1124, FN=0, TP=769 — **predicts EVERY image as tampered**
- Pixel AUC ~0.50 — pure random chance
- Zero variation in image-level metrics across compression conditions

---

## PART 4 — Ablation Study Analysis

### Hypothesis Verification
**Cannot be evaluated.** The experiment intended to measure P.3's compression robustness, but the P.3 model was never loaded. The results measure ImageNet-weight performance, which is scientifically meaningless.

### Expected vs Actual

| Condition | Expected F1 (from P.3) | Actual F1 (ImageNet) | |
|-----------|----------------------|---------|---|
| Original | ~0.6920 | 0.0362 | INVALID |
| Q=95 | ~0.67–0.69 | 0.0344 | INVALID |
| Q=90 | ~0.63–0.67 | 0.0292 | INVALID |
| Q=80 | ~0.55–0.63 | 0.0389 | INVALID |
| Q=70 | ~0.45–0.55 | 0.0400 | INVALID |

---

## PART 5 — Results Extraction

### Reported Metrics (ALL INVALID — untrained model)

| Condition | Pixel F1 | IoU | Pixel AUC | Img Acc | Img F1 | Img AUC |
|-----------|----------|-----|-----------|---------|--------|---------|
| Original | 0.0362 | 0.0185 | 0.5009 | 40.62% | 0.2889 | 0.1630 |
| Q=95 | 0.0344 | 0.0175 | 0.4639 | 40.62% | 0.2889 | 0.2749 |
| Q=90 | 0.0292 | 0.0148 | 0.5003 | 40.62% | 0.2889 | 0.3184 |
| Q=80 | 0.0389 | 0.0198 | 0.5044 | 40.62% | 0.2889 | 0.3474 |
| Q=70 | 0.0400 | 0.0204 | 0.5100 | 40.62% | 0.2889 | 0.3395 |

---

## PART 7 — Suggested Improvements

1. **Upload P.3 checkpoint as Kaggle dataset** — The `.pth` file must be uploaded as a dataset input before running
2. **Add checkpoint validation gate** — Abort execution with `assert` or `raise` if checkpoint not found
3. **Validate "Original" baseline matches P.3** — After loading checkpoint, verify Original condition reproduces P.3's Pixel F1 (0.6920 ± 0.001)
4. **Add more quality levels** — Include Q=60, Q=50 for extreme cases, and Q=85 for social media sweet spot

---

## PART 8 — Final Verdict

### Scores

| Category | Score | Notes |
|----------|-------|-------|
| Research value | **2/10** | Good experimental design, but zero usable data |
| Implementation quality | **5/10** | Clean code, good metrics framework, fatal checkpoint failure |
| Experimental validity | **0/10** | All results from untrained model |
| **Overall** | **2.3/10** | |

### Verdict: **INVALID — Must be re-run with P.3 checkpoint**

The notebook framework is well-designed and ready to produce valuable results. The only issue is the missing checkpoint. **Re-run with the P.3 `.pth` file uploaded as a Kaggle dataset input.** All code is correct — only the data dependency is missing.

### Required Fix for Re-run
1. Save `vR.P.3_unet_resnet34_model.pth` as a Kaggle dataset
2. Attach it as input to the notebook
3. Update the checkpoint search path to include the attached dataset directory
4. Re-run — all cells should work without modification
