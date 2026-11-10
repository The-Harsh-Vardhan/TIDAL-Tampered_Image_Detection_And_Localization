# 08 — Top 10 Critical Risks

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

| # | Risk | Severity | Document | Impact |
|---|---|---|---|---|
| 1 | **Mixed-set F1 includes authentic samples scoring 1.0**, inflating the primary reported metric | 🔴 CRITICAL | [02](02_Metric_Inflation_Check.md) | Results appear better than they are. Tampered-only F1 could be 20-30 points lower than "all" F1. |
| 2 | **No pretrained encoder** — 31M parameters trained from scratch on ~7K images | 🔴 CRITICAL | [05](05_Model_Architecture_Critique.md) | High overfitting risk, poor feature quality, inferior to SMP+ResNet34. |
| 3 | **Notebook targets Kaggle, assignment requires Colab** — hardcoded `/kaggle/` paths, no Drive mount | 🔴 CRITICAL | [00](00_Assignment_Requirement_Coverage.md) | Fails assignment requirement #9. |
| 4 | **No cross-dataset validation** — model only tested on CASIA v2 | 🟡 HIGH | [03](03_Shortcut_Learning_Analysis.md) | Cannot distinguish between tamper detection and CASIA-specific artifact detection. |
| 5 | **JPEG compression shortcut untested** — model may detect JPEG artifacts, not semantic tampering | 🟡 HIGH | [03](03_Shortcut_Learning_Analysis.md) | Model may fail on non-JPEG or same-quality images. |
| 6 | **Stderr completely suppressed** (`os.dup2` to devnull) — critical errors silenced | 🟡 HIGH | [07](07_Engineering_Quality_Audit.md) | CUDA errors, OOM, and tracebacks will be invisible. |
| 7 | **7 unused imports/functions** — `roc_auc_score`, `is_valid_image()`, `pandas`, etc. | 🟠 MEDIUM | [07](07_Engineering_Quality_Audit.md) | Suggests hastily assembled code. Dead validation code means images are not actually validated. |
| 8 | **pos_weight computed on unresized masks** — effective pos_weight mismatches training scale | 🟠 MEDIUM | [01](01_Data_Leakage_Check.md) | BCE loss is slightly miscalibrated. |
| 9 | **TF32 enabled contradicts deterministic mode** — results are not truly reproducible | 🟠 MEDIUM | [06](06_Training_Pipeline_Robustness.md) | Reproducing exact numbers on re-run is not guaranteed. |
| 10 | **No `cv2.imread` null-check on images** — corrupt images will crash with NoneType error | 🟠 MEDIUM | [07](07_Engineering_Quality_Audit.md) | Pipeline is fragile; one bad image kills the run. |

---

## Risk Distribution

- 🔴 CRITICAL: 3
- 🟡 HIGH: 3
- 🟠 MEDIUM: 4

## Actionable Fixes (Priority Order)

1. Report **tampered-only F1 as the primary metric**; label mixed-set F1 clearly as "including authentic"
2. Replace custom UNet with SMP + pretrained ResNet34 (or add it as a comparison)
3. Add dual Kaggle/Colab support or convert to Colab
4. Add cross-dataset test (even on a small external set)
5. Remove stderr suppression; use targeted `warnings.filterwarnings` instead
6. Delete dead code and unused imports
7. Compute pos_weight on resized masks
8. Remove TF32 enable or remove deterministic flag
9. Add `cv2.imread` null check for images
10. Add resume-from-checkpoint for Kaggle disconnections
