# Docs11: High Impact Improvements

Ranked improvements with research-backed justifications. Each improvement is assigned to a priority tier:

- **P0 (Must-Do):** Required for a competitive submission
- **P1 (High Impact):** Implement if time permits — significantly improves evaluation depth
- **P2 (Future Work):** Deferred — valuable but not needed for this submission

---

## P0 Tier — Required for Competitive Submission

### I1: Pretrained ResNet34 Encoder

**Addresses:** W1 (no pretrained encoder)

**What:** Replace the custom from-scratch encoder with an ImageNet-pretrained ResNet34 backbone via `segmentation_models_pytorch` (SMP). Add a custom classification head to the bottleneck features.

**Research justification:** v8 achieved AUC=0.817 with pretrained ResNet34. Every surveyed research paper (P1-P21) uses pretrained encoders. Training 31M params from scratch on ~5,500 images is data-inefficient.

**Implementation:**
```python
import segmentation_models_pytorch as smp

class TamperDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.segmentor = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=config['in_channels'],
            classes=1,
        )
        encoder_out = self.segmentor.encoder.out_channels[-1]  # 512
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout']),
            nn.Linear(256, config['n_labels']),
        )

    def forward(self, x):
        features = self.segmentor.encoder(x)
        cls_logits = self.classifier(features[-1])
        decoder_output = self.segmentor.decoder(*features)
        seg_logits = self.segmentor.segmentation_head(decoder_output)
        return cls_logits, seg_logits
```

| Attribute | Value |
|---|---|
| Difficulty | Medium (model class refactoring) |
| Expected impact | +0.10-0.15 image AUC, +0.05-0.10 tampered F1 |
| Parameters | ~24.5M (down from 31M) |
| GPU memory | Neutral |
| Dependencies | None |

---

### I2: ELA as 4th Input Channel

**Addresses:** W2 (RGB-only input)

**What:** Compute Error Level Analysis map for every image (JPEG re-save at QF=90, compute absolute pixel difference). Feed 4-channel input (RGB + ELA grayscale) to the encoder.

**Research justification:** Papers P1 and P7 achieved 96.21% accuracy on CASIA v2.0 using ELA + lightweight CNN. ELA amplifies JPEG compression inconsistencies between authentic and tampered regions — a signal invisible in RGB. This was the top-priority approved improvement in Docs9 (04_Improvement_Decision_Log.md item #2).

**Implementation:**
```python
def compute_ela(image_bgr, quality=90):
    """Compute Error Level Analysis map."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image_bgr, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image_bgr, decoded)
    return cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
```

Add to dataset `__getitem__`: compute ELA, stack as 4th channel, pass through augmentation via Albumentations `additional_targets`. SMP handles first-layer weight adaptation for 4-channel input automatically by averaging pretrained RGB weights.

| Attribute | Value |
|---|---|
| Difficulty | Easy-Medium |
| Expected impact | +0.05-0.10 tampered F1, especially on copy-move |
| GPU memory | Negligible (+1 channel) |
| Dependencies | Pairs well with I1 but independent |

---

### I3: Threshold Sweep / Optimization

**Addresses:** W5 (fixed threshold at 0.5)

**What:** After training, sweep segmentation threshold from 0.05 to 0.80 in 0.05 steps on the validation set. Select the threshold that maximizes tampered-only F1, then apply it to test evaluation.

**Research justification:** v8 found optimal threshold at 0.75 vs default 0.50. This is a free metric improvement with zero retraining. No model is perfectly calibrated at 0.5.

**Implementation:** ~30 lines: loop over thresholds, compute F1 at each, select best, plot F1-vs-threshold curve.

| Attribute | Value |
|---|---|
| Difficulty | Very Low |
| Expected impact | +0.05-0.15 on all segmentation metrics |
| GPU memory | Zero (evaluation-time only) |
| Dependencies | Trained model from any architecture |

---

### I4: Robustness Testing Suite

**Addresses:** W6 (no robustness testing)

**What:** Test the trained model against 8 degradation conditions applied to test images at inference time:

1. JPEG compression QF=70
2. JPEG compression QF=50
3. Gaussian noise σ=10
4. Gaussian noise σ=25
5. Gaussian blur kernel=3
6. Gaussian blur kernel=5
7. Resize 0.75× then back to original
8. Resize 0.5× then back to original

Report metrics per condition and delta from clean baseline. Produce a grouped bar chart.

**Research justification:** This is assignment bonus B1. v8 revealed JPEG robustness is good (0.9% drop) but Gaussian noise causes 13% drop. Papers P4, P13, P17 all include post-processing robustness evaluation.

| Attribute | Value |
|---|---|
| Difficulty | Medium (~60 lines + bar chart) |
| Expected impact | No metric improvement, HIGH grade impact (explicit bonus) |
| GPU memory | Inference only |
| Dependencies | Trained model |

---

### I5: Grad-CAM Explainability

**Addresses:** W7 (no Grad-CAM)

**What:** Generate Grad-CAM heatmaps from the encoder's deepest layer. Overlay on images with diagnostic coloring to show what the model attends to.

**Research justification:** Standard explainability technique. Demonstrates "thoughtful architecture choices" (Assignment Section 2). Shows reviewers the model learned meaningful forensic features, not dataset shortcuts.

**Implementation:** Hook-based gradient extraction from encoder bottleneck, compute weighted activation map, overlay with colormap.

| Attribute | Value |
|---|---|
| Difficulty | Medium (~80 lines) |
| Expected impact | No metric improvement, HIGH credibility |
| GPU memory | Inference only |
| Dependencies | Trained model; for SMP model, hook into encoder.layer4 |

---

### I6: Data Leakage Verification

**Addresses:** W14 (data leakage unverified)

**What:** Explicit verification cell that asserts zero overlap between train/val/test image paths. Include pHash near-duplicate detection.

**Research justification:** The data leakage CSV bug appeared in vK.3 and vK.7.5 (training on test set). Audit8 Pro flagged missing content-level leak checks. This is cheap credibility.

| Attribute | Value |
|---|---|
| Difficulty | Very Low (~15 lines) |
| Expected impact | Credibility, catches silent bugs |
| GPU memory | CPU only |
| Dependencies | None (run before training) |

---

## P1 Tier — High Impact, Implement If Time Permits

### I7: Edge Supervision Loss

**Addresses:** W3 (no edge supervision)

**What:** Add an auxiliary loss that supervises predicted mask boundaries against ground truth mask boundaries. Compute edges with Sobel operator, apply BCE loss.

```python
total_loss = α * FocalLoss(cls) + β * (w_bce*BCE + w_dice*Dice)(seg) + γ * EdgeLoss(seg, masks)
```

Where γ = 0.3 (tunable via CONFIG).

**Research justification:** EMT-Net (AUC=0.987) uses Edge Artifact Enhancement; ME-Net (F1=0.905) uses Edge Enhancement Path Aggregation. Both attribute boundary supervision as critical for localization under post-processing.

| Attribute | Value |
|---|---|
| Difficulty | Easy-Medium (~20 lines) |
| Expected impact | +0.01-0.03 tampered F1, improved boundary quality |
| GPU memory | Negligible |
| Dependencies | None |

---

### I8: Forgery-Type Evaluation Breakdown

**Addresses:** W8 (no forgery-type breakdown)

**What:** Parse CASIA filenames (`Tp_D_*` = copy-move, `Tp_S_*` = splicing) to report per-type metrics separately.

| Attribute | Value |
|---|---|
| Difficulty | Low (~30 lines) |
| Expected impact | Reveals copy-move weakness (addresses B2) |
| Dependencies | Trained model |

---

### I9: Mask-Size Stratified Evaluation

**Addresses:** W9 (no mask-size stratification)

**What:** Bucket test images by tampered region percentage: tiny (<2%), small (2-5%), medium (5-15%), large (>15%). Report metrics per bucket.

| Attribute | Value |
|---|---|
| Difficulty | Low (~40 lines) |
| Expected impact | Reveals size-dependent failure modes |
| Dependencies | Trained model |

---

### I10: Failure Case Analysis

**What:** Display the 10 worst predictions (lowest per-sample F1) with GT mask, predicted mask, and metadata (mask size %, forgery type). Annotate failure modes.

| Attribute | Value |
|---|---|
| Difficulty | Low (~40 lines) |
| Expected impact | Demonstrates self-awareness |
| Dependencies | Trained model + I8 + I9 |

---

### I11: Confusion Matrix + ROC/PR Curves

**What:** Plot 2×2 confusion matrix heatmap for image-level classification. Plot ROC curve and Precision-Recall curve.

| Attribute | Value |
|---|---|
| Difficulty | Very Low (~25 lines) |
| Expected impact | Standard deliverable |
| Dependencies | Trained model |

---

### I12: Shortcut Learning Validation

**What:** Two tests:
1. **Mask randomization:** Shuffle masks across images. If F1 drops → model uses image content, not shortcuts.
2. **Boundary sensitivity:** Erode/dilate predicted masks by 1px. If metrics barely change → predictions aren't just edge artifacts.

| Attribute | Value |
|---|---|
| Difficulty | Low (~40 lines) |
| Expected impact | Proves scientific rigor |
| Dependencies | Trained model |

---

### I13: Gradient Accumulation

**Addresses:** W12 (no gradient accumulation)

**What:** Accumulate gradients over N mini-batches before optimizer step. With batch_size=16 and accumulation_steps=4, effective batch=64.

| Attribute | Value |
|---|---|
| Difficulty | Medium (training loop modification) |
| Expected impact | Stabilized training, better convergence |
| Dependencies | None (training infrastructure change) |

---

## P2 Tier — Future Work / Deferred

### I14: SRM Noise Residuals

**Addresses:** W2 (RGB-only input, complementary to ELA)

**What:** Apply SRM high-pass filter bank to extract noise residuals as additional input channels (6-channel: RGB + ELA + SRM).

**Research justification:** EMT-Net (P13) and ME-Net (P17) use SRM as core component.

| Attribute | Value |
|---|---|
| Difficulty | Medium-Hard |
| Expected impact | Potentially large if ELA alone is insufficient |
| Dependencies | I2 (try ELA first) |

---

### I15: Attention Mechanisms (SE/CBAM)

**Addresses:** W4 (no attention)

**What:** Add channel attention (SE blocks) or spatial+channel attention (CBAM) in the decoder or skip connections.

**Research justification:** TransU2-Net (P21) showed +14.2% F-measure from attention.

| Attribute | Value |
|---|---|
| Difficulty | Medium |
| Expected impact | +0.05-0.14 F-measure (uncertain on this architecture) |

---

### I16: Multi-Scale Training / Test-Time Augmentation

**What:** Train at multiple resolutions or apply TTA (flip, multi-scale) at inference.

| Attribute | Value |
|---|---|
| Difficulty | Medium |
| Expected impact | +0.01-0.03 |

---

### I17: Cross-Dataset Evaluation

**What:** Test on Coverage, CoMoFoD, or NIST'16 for generalization evidence.

| Attribute | Value |
|---|---|
| Difficulty | Medium (dataset preparation) |
| Expected impact | Demonstrates generalization (highly valued) |

---

### I18: Pixel-Level AUC-ROC

**Addresses:** W10 (no pixel-level AUC)

**What:** Compute `sklearn.metrics.roc_auc_score(gt.flatten(), pred_prob.flatten())` as threshold-independent localization quality metric.

| Attribute | Value |
|---|---|
| Difficulty | Low (~5 lines) |
| Expected impact | Better evaluation metric |

---

## Priority Summary

| Tier | Items | Total Code | Grade Impact |
|---|---|---|---|
| **P0** | I1-I6 | ~250 lines + model refactor | HIGH — addresses all CRITICAL weaknesses |
| **P1** | I7-I13 | ~300 lines | MEDIUM — deepens evaluation, improves training |
| **P2** | I14-I18 | Variable | LOW for grade, HIGH for research value |

**Recommended implementation order within P0:** I6 (leakage check) → I1 (pretrained encoder) → I2 (ELA) → I3 (threshold sweep) → I4 (robustness) → I5 (Grad-CAM)

**Rationale:** Start with verification (I6), then make the biggest architectural change (I1+I2), then extract maximum metrics from the model (I3), then earn bonus points (I4, I5).
