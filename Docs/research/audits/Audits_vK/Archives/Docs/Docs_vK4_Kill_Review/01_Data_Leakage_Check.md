# 01 — Data Leakage Check

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Findings

### 1. Split Strategy — ✅ No Obvious Leakage

The split uses `train_test_split` with `stratify` on label and a fixed `random_state=42`. The split is performed on image-path-level entries, so the same image cannot appear in both train and val/test sets — **assuming all filenames are unique**, which is a reasonable assumption for CASIA.

### 2. Mask-Name Cross-Extension Search — ⚠️ LOW RISK

Cell 10, lines 302–308: when a mask isn't found with the exact filename, the code tries alternative extensions (`.png`, `.jpg`, `.tif`, `.bmp`). This could theoretically match an **incorrect** mask if naming conventions aren't strict. This is a weak point but unlikely to cause systematic leakage.

### 3. `is_valid_image()` Defined But Never Called — ⚠️ DEAD CODE

The function `is_valid_image()` is defined (Cell 10 line 263) but **never invoked**. This means corrupted or truncated images silently enter the pipeline. If a corrupted image consistently produces a zero-tensor, the model gets free "authentic" signal. Not leakage per se, but an integrity gap.

### 4. pos_weight Computed on Raw (Unreized) Masks — ⚠️ MEDIUM RISK

Cell 20: `pos_weight` is computed by reading raw masks from disk via `cv2.imread`, without resizing to 256×256. The actual training masks ARE resized (via albumentations `A.Resize`). If the raw mask dimensions differ from the resized dimensions, the foreground/background ratio will be **different** from what the model actually sees during training. This is a subtle bug that could cause a slightly miscalibrated `pos_weight`.

### 5. No Image Hash Deduplication — ⚠️ MEDIUM RISK

CASIA v2 is known to contain near-duplicate images (same base image with different tampered regions). The split is filename-based, not content-hash-based. If two filenames share the same base image — one authentic, one tampered — they could land in different splits, constituting **soft leakage**. This is a dataset-level issue common to CASIA work, but the notebook does not acknowledge or mitigate it.

---

## Verdict

**No catastrophic leakage**, but two medium-risk issues:
1. `pos_weight` computed on unreized masks (measurement error)
2. No hash-based deduplication (soft leakage from near-duplicate base images)

**Severity: LOW-MEDIUM**
