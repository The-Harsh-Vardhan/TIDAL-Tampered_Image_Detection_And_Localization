# 04 — Visualization Verification

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Prediction Grid (Cell 34) — ✅ with caveats

**Strengths:**
- Shows Best 2 / Median 2 / Worst 2 tampered + 2 authentic — this is a reasonable sampling strategy that avoids pure cherry-picking
- 4-column format (Original / GT Mask / Predicted Mask / Overlay) is standard and informative
- Each row annotates label, prediction, and F1 score

**Weaknesses:**
- **Sorting by F1 only** — worst cases are shown, but there's no explicit **failure analysis** explaining WHY they failed (mask too small? wrong forgery type? unusual image?)
- **No forgery-type annotation on the grid** — the `forgery_type` field is collected but never displayed. A viewer cannot tell if worst cases are systematically copy-move or splicing
- **Only 8 samples total** from potentially hundreds — the grid is useful but limited

## Grad-CAM (Cell 37) — ⚠️ Partially Misleading

**Concern 1:** Grad-CAM is computed by backpropagating `seg_logits.sum()`. This sums ALL pixel logits, meaning the CAM shows where the model activates in general — not where it specifically detects tampering. For authentic images, the model should output near-zero everywhere, yet the CAM would still show activations. The Grad-CAM target should be **tampered-region logits only** (e.g., masked by GT or by positive predictions).

**Concern 2:** Only the **top 4 best-performing** tampered samples are shown. This is **textbook cherry-picking** for Grad-CAM. The most informative Grad-CAMs would be on **failure cases** — where is the model looking when it gets the answer wrong?

**Concern 3:** The Grad-CAM hook attaches to `model.down4.conv.block`, which is the deepest encoder layer. This is semantically high-level but spatially very coarse (16×16 at 256 input). The upsampled CAM will look "blobby" regardless of what the model learned.

## F1 vs Threshold (Cell 35) — ✅ Good

Clean, informative, correctly computed on validation set.

## Training Curves (Cell 27) — ✅ Good

4-subplot layout (loss, F1, Dice/IoU, LR) is comprehensive.

## Missing Visualizations

- **Confusion matrix** for image-level classification — critical omission
- **Per-forgery-type F1 distribution** (box plots or histograms)
- **Error distribution** — histogram of per-sample F1 scores
- **Failure panel** — dedicated worst-case analysis with annotations

---

## Verdict

Visualizations are better than average for an assignment notebook, but Grad-CAM cherry-picks best cases and uses a misleading backprop target. Failure analysis is absent.

**Severity: MEDIUM**
