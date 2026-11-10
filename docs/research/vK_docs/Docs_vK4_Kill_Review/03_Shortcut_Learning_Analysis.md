# 03 — Shortcut Learning Analysis

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Built-in Shortcut Test — Mask Randomization

Cell 41 implements a mask randomization test: predictions are compared against **random** binary masks. A high F1 would indicate the model predicts dataset-correlated patterns. This test is correctly designed and is a genuine strength.

However, the test has a **fundamental flaw**: comparing against *random 50/50 binary masks* will always yield low F1 regardless, because random masks have ~50% foreground while forgery masks are typically <15%. A more useful test would use **shuffled ground-truth masks** (permute across samples) to test whether the model's predictions correlate with specific GT masks.

---

## Unaddressed Shortcut Risks

### 1. JPEG Compression Artifact Exploitation — 🔴 HIGH RISK

CASIA v2 tampered images are often saved with different JPEG quality than authentic images. The splicing boundaries create detectable compression inconsistencies. The model may learn to detect **JPEG grid boundary mismatues** rather than semantic tampering content.

The augmentation pipeline includes `ImageCompression(quality_lower=50, quality_upper=95, p=0.3)` which partially mitigates this, but only at 30% probability and only during training. **There is no test** to verify the model's predictions change when JPEG artifacts are eliminated (e.g., converting all inputs to lossless PNG first).

### 2. Image Naming Convention Leakage — ⚠️ MEDIUM RISK

The `detect_forgery_type()` function (Cell 10) extracts forgery type from **filenames** (`_cm` for copy-move, `_sp` for splicing). While this information is only used for metadata, it reveals that filenames encode label information. If any preprocessing step inadvertently reads filenames, this would be direct leakage. Currently safe, but fragile.

### 3. Authentic Mask = All Zeros — ⚠️ MEDIUM RISK

When `mask_path is None` for authentic images, a zero mask is generated (Cell 14 line 451). The model receives a strong signal: "if the mask supervision is all zeros, the classification label is always 0." This is correct ground truth, but it means the model can achieve perfect segmentation on authentic images by simply outputting zeros, which inflates mixed metrics (see doc 02).

### 4. Missing Boundary Sensitivity Test — ⚠️ MEDIUM RISK

v8 includes a boundary sensitivity test (erosion/dilation impact on F1). vK.4 does **not** include this. Without it, we cannot determine if the model is detecting forgery boundaries specifically or just outputting blob-like predictions that happen to overlap.

### 5. No Cross-Dataset Validation — 🔴 HIGH RISK

The model is trained and tested exclusively on CASIA v2. There is no validation on a second dataset (e.g., Columbia, COVERAGE, or DSO-1). CASIA-specific artifacts (naming, compression, resolution patterns) could be the primary learned features. Without cross-dataset testing, we cannot distinguish between "good tampering detector" and "good CASIA memorizer."

---

## Verdict

The mask randomization test is a positive step but is insufficient. Critical shortcut vectors (JPEG artifacts, CASIA-specific patterns) are not tested. The model could be a **CASIA artifact detector** rather than a **tamper detector**.

**Severity: HIGH**
