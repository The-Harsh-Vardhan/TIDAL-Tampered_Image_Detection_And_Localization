# Ablation Candidates from vK.12.0

| Field | Value |
|-------|-------|
| Source | vK.12.0 (failed dual-head UNet+ResNet34) |
| Target Roadmap | DocsR1/ablation_master_plan.md |
| Constraint | Each candidate = exactly one variable change from parent version |
| Date | 2026-03-15 |

---

## Already Planned (Validated by vK.12.0 Experience)

### 1. ReduceLROnPlateau — vR.1.5 (CONFIRMED)

| Field | Value |
|-------|-------|
| Roadmap version | vR.1.5 |
| vK.12.0 cell | 62 |
| vK.12.0 config | factor=0.5, patience=3, min_lr=1e-7 |
| Current plan config | factor=0.5, patience=3, monitor=val_loss |
| Insight from vK.12.0 | Used successfully across vK.11.x–12.0 runs. The scheduler prevented complete training collapse even when the architecture was fundamentally broken. This confirms it is a stabilizing force worth adding. |
| Recommendation | Proceed as planned. Consider adding min_lr=1e-6 (vK.12.0 used 1e-7 which may be too low for the ETASR learning rate of 1e-4). |

### 2. Encoder Freeze Warmup — vR.P.1 (CONFIRMED)

| Field | Value |
|-------|-------|
| Roadmap version | vR.P.1 |
| vK.12.0 cell | 72 |
| Insight | vK.12.0 used a 2-epoch warmup where only the decoder trained before unfreezing encoder layers. Despite the overall failure, the warmup prevented early divergence. |
| Recommendation | Proceed as planned for Track 2. |

---

## New Candidates for Track 1 (ETASR, TF/Keras)

### Candidate A: Gradient Clipping — HIGH PRIORITY

| Field | Value |
|-------|-------|
| vK.12.0 cell | 69 |
| vK.12.0 implementation | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| TF/Keras equivalent | `optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)` |
| What it tests | Whether gradient clipping prevents the training instability observed in vR.1.0 (val acc dropped 4% in one epoch around epoch 11). |
| Recommended insertion | After vR.1.5 (LR scheduler). The scheduler already addresses instability; clipping tests a complementary approach. Could be **vR.1.8**. |
| Parent version | vR.1.5 (inherits LR scheduler) |
| Single-variable compliance | **YES** — only changes the optimizer's gradient processing. All other parameters frozen. |
| Expected impact | Stabilized training; prevents val-loss spikes. Unlikely to change final accuracy significantly but improves training reliability. |
| Risk | Low. Clipping is purely protective — it only activates when gradients exceed the norm threshold. No impact during normal training. |

---

### Candidate B: JPEG Compression Augmentation — HIGH PRIORITY

| Field | Value |
|-------|-------|
| vK.12.0 cell | 44 |
| vK.12.0 implementation | Albumentations `JpegCompression(quality_lower=50, quality_upper=90)` |
| TF/Keras equivalent | Custom preprocessing using `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, qf])` with random QF ∈ [50, 90] before ELA computation. |
| What it tests | Whether **signal-preserving** augmentation (JPEG re-compression at varying quality) succeeds where **geometric** augmentation (flip/rotate) failed in vR.1.2. |
| Hypothesis | ELA maps are sensitive to compression artifacts, so augmenting with varying compression levels adds meaningful diversity without destroying the ELA signal. Geometric transforms destroy pixel-exact ELA patterns that the Flatten→Dense layer memorizes. |
| Recommended insertion | After vR.1.7 (GAP replaces Flatten). The Flatten→Dense architecture cannot handle any augmentation due to pixel-exact memorization. GAP provides spatial invariance, enabling augmentation. Could be **vR.1.9**. |
| Parent version | vR.1.7 (inherits GAP architecture) |
| Single-variable compliance | **YES** — only changes augmentation (same type of change as vR.1.2). Would NOT combine with geometric augmentation. |
| Expected impact | +1–3% accuracy. JPEG augmentation makes the model robust to varying compression, which is directly relevant to ELA analysis. |
| Risk | Medium. Still an augmentation change, and vR.1.2 showed the original architecture is augmentation-sensitive. Must wait for GAP. |

---

### Candidate C: Alternative ELA Implementation (cv2-based) — LOW PRIORITY

| Field | Value |
|-------|-------|
| vK.12.0 cell | 43 |
| vK.12.0 implementation | `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])` followed by `cv2.imdecode()` |
| Current implementation | PIL `Image.save(BytesIO(), 'JPEG', quality=q)` with `ImageChops.difference()` |
| What it tests | Whether the JPEG codec (OpenCV vs PIL/Pillow) affects ELA map quality. Different libraries use different DCT implementations and quantization tables. |
| Recommended insertion | Low priority. Only worth testing if ELA signal quality is suspected as a bottleneck. Could be **vR.1.10** or appended to the very end of the roadmap. |
| Parent version | Latest best-performing Track 1 version |
| Single-variable compliance | **YES** — only changes the ELA computation method. |
| Expected impact | Likely neutral (±0.5%). The difference between codecs is minimal at quality=90. |
| Risk | Very low. |

---

## New Candidates for Track 2 (Pretrained, PyTorch)

### Candidate D: Differential Learning Rates — MEDIUM PRIORITY

| Field | Value |
|-------|-------|
| vK.12.0 cell | 62 |
| What it tests | Separate learning rates for encoder (low, e.g. 1e-5) and decoder (high, e.g. 1e-3), preserving pretrained features while decoder learns domain-specific reconstruction. |
| Recommended insertion | After vR.P.1 (gradual unfreeze). |
| Implementation | `optimizer = torch.optim.Adam([{'params': model.encoder.parameters(), 'lr': 1e-5}, {'params': model.decoder.parameters(), 'lr': 1e-3}])` |
| Single-variable compliance | **YES** — only changes LR assignment, not architecture or loss. |

---

### Candidate E: Three-Phase Training — LOW PRIORITY

| Field | Value |
|-------|-------|
| vK.12.0 cell | 72 |
| What it tests | Extension of vR.P.1: (1) frozen encoder for 5 epochs, (2) last 2 encoder blocks unfrozen with low LR for 10 epochs, (3) full unfreeze with very low LR for remaining epochs. |
| Recommended insertion | Could follow or replace vR.P.1. |
| Single-variable compliance | **YES** — only changes training schedule. |

---

## Priority Summary

| Priority | Candidate | Track | Insert After | Variable Changed | Effort |
|----------|-----------|-------|-------------|-----------------|--------|
| HIGH | A: Gradient clipping | Track 1 | vR.1.5 | Optimizer clipnorm | Low (1 param) |
| HIGH | B: JPEG compression augmentation | Track 1 | vR.1.7 (GAP) | Augmentation | Medium |
| MEDIUM | D: Differential learning rates | Track 2 | vR.P.1 | LR per param group | Low |
| LOW | C: Alternative ELA (cv2) | Track 1 | End of roadmap | ELA implementation | Low |
| LOW | E: Three-phase training | Track 2 | After vR.P.1 | Training schedule | Medium |

---

## Tentative Roadmap Extension

If the current roadmap (vR.1.3 through vR.1.7) completes and time permits:

| Version | Change | Source | Parent |
|---------|--------|--------|--------|
| vR.1.8 | Gradient clipping (clipnorm=1.0) | vK.12.0 cell 69 | vR.1.5 |
| vR.1.9 | JPEG compression augmentation (QF 50–90) | vK.12.0 cell 44 | vR.1.7 |
| vR.1.10 | Alternative ELA (cv2 vs PIL) | vK.12.0 cell 43 | Best Track 1 |

These are tentative and subject to results from vR.1.3–1.7.
