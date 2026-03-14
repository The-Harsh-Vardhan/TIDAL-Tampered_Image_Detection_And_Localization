# Experiment Plan

| Field | Value |
|-------|-------|
| **Notebook** | vR.0 Image Detection and Localisation.ipynb |
| **Dataset** | CASIA v2.0 (~12,614 images) |
| **Platform** | Kaggle (T4 GPU) or Google Colab |

---

## 1. Primary Experiment: ETASR Paper Reproduction

### Objective
Reproduce the results of ETASR_9593 paper on CASIA v2.0 dataset using the ELA + CNN approach.

### Paper-Reported Targets
| Metric | Target |
|--------|--------|
| Accuracy | 96.21% |
| Precision | 98.58% |
| Recall | 92.36% |
| F1 Score | 95.37% |

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| Image size | 128x128 |
| ELA quality | 90 |
| CNN architecture | Conv2D(32)-Conv2D(32)-MaxPool-FC(256)-Softmax(2) |
| Optimizer | Adam (lr=0.0001) |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | patience=5, monitor=val_accuracy |
| Split | 70/15/15 stratified |
| Seed | 42 |

### Success Criteria
- Accuracy >= 90% on test set (reasonable threshold for reproduction)
- F1 Score >= 85%
- ROC-AUC >= 0.90
- Training completes without errors
- All metrics computed and visualized

---

## 2. Sanity Checks

### Check 1: Data Integrity
- Verify Au and Tp directories found and populated
- Verify class ratio is approximately 1.46:1 (Au:Tp)
- Verify no images skipped during ELA processing (or minimal skips)
- Verify split sizes are approximately 70/15/15

### Check 2: ELA Preprocessing
- Visualize ELA maps for authentic vs tampered images
- Confirm tampered images show brighter/non-uniform ELA patterns
- Confirm authentic images show dimmer/uniform ELA patterns

### Check 3: Model Architecture
- Verify model.summary() matches paper Table III
- Verify total parameters ~29.5M
- Verify input shape (128, 128, 3) and output shape (2)

### Check 4: Training Convergence
- Training loss should decrease over epochs
- Validation accuracy should plateau near 90%+
- No mode collapse (accuracy stuck at ~50%)
- Early stopping should trigger (not running full 50 epochs)

### Check 5: Evaluation Validity
- Test set was never used during training
- Confusion matrix values sum to test set size
- Both classes represented in predictions (not all one class)

---

## 3. Expected Results

Based on the paper and similar ELA+CNN implementations:

| Metric | Expected Range | Rationale |
|--------|---------------|-----------|
| Accuracy | 88-96% | Paper reports 96.21%; reproduction typically ~90%+ |
| Precision | 90-98% | High precision expected (few false positives) |
| Recall | 85-93% | Some tampered images harder to detect |
| F1 Score | 88-95% | Balanced metric |
| ROC-AUC | 0.92-0.98 | Strong discriminative ability expected |
| Training epochs | 15-30 | Early stopping with patience=5 |

### Known Factors Affecting Reproduction
1. **File format filtering**: Paper may have used different subset of images
2. **Split randomization**: Different random split may yield +/-2% variation
3. **TensorFlow version**: Minor numerical differences across versions
4. **GPU type**: T4 vs other GPUs may cause slight training differences

---

## 4. Ablation Studies (If Time Permits)

### Ablation A: ELA Quality
Test Q=70, Q=80, Q=90, Q=95 to measure sensitivity to ELA quality parameter.

### Ablation B: Image Size
Test 64x64, 128x128, 256x256 to measure resolution impact on accuracy.

### Ablation C: Data Augmentation
Add horizontal/vertical flips and test impact on validation accuracy.

### Ablation D: Deeper Model
Add a third Conv2D layer and test if additional depth improves results.

---

## 5. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Accuracy below 85% | Low | ELA+CNN is well-proven on CASIA v2.0 |
| Dataset not found on Kaggle | Low | Auto-detect path; fallback instructions provided |
| OOM on GPU | Very Low | Model is tiny (~29.5M params); batch=32 fits easily |
| ELA fails on non-JPEG images | Medium | Graceful skip + count reported |
| Training doesn't converge | Low | Early stopping + Adam optimizer handles well |

---

## 6. Track 2: Pretrained Encoder-Decoder Experiment Plan (vR.P.x)

### Objective

Achieve pixel-level tampered region localization using a pretrained ResNet-34 encoder with a UNet decoder — fulfilling the assignment's core requirement for spatial masks.

### Why a Second Track?

The ETASR CNN (Track 1) outputs only a binary classification label. The assignment explicitly requires "tampered region masks" and "Original / Ground Truth / Predicted / Overlay" visualizations. A pretrained encoder-decoder is the only viable path to localization without building a model from scratch (which has failed in this project — vK.11-12 synthesis runs).

### Pretrained Experiment Roadmap

| Version | Change | Input | Encoder State | Expected Impact |
|---------|--------|-------|--------------|-----------------|
| **vR.P.0** | ResNet-34 + UNet, RGB, frozen encoder | RGB 384x384 | Fully frozen | Establish localization baseline |
| **vR.P.1** | Dataset fix + GT mask auto-detection | RGB 384x384 | Fully frozen | Proper GT masks from sagnikkayalcse52 |
| **vR.P.1.5** | Speed optimizations (AMP, workers) | RGB 384x384 | Fully frozen | Training speed only |
| **vR.P.2** | Gradual unfreeze (last 2 encoder blocks) | RGB 384x384 | Partially unfrozen | +2-5% F1 from domain adaptation |
| **vR.P.3** | ELA as input (replace RGB) | ELA 384x384 | Frozen (BN unfrozen) | Test ELA + pretrained compatibility |
| **vR.P.4** | 4-channel (RGB + ELA) | RGB+ELA 384x384 | Frozen (conv1 unfrozen) | Test combined signal |
| **vR.P.5** | ResNet-50 encoder | RGB 384x384 | Frozen | Test deeper features |
| **vR.P.6** | EfficientNet-B0 encoder | RGB 384x384 | Frozen | Test parameter efficiency |

### vR.P.0 Configuration (First Experiment)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Encoder | ResNet-34 | Proven in v6.5 (Tam-F1 = 0.41) |
| Encoder weights | ImageNet | Standard transfer learning |
| Encoder frozen | Yes (fully) | Protect pretrained features, only ~500K trainable |
| Decoder | UNet (SMP default) | Skip connections from 4 encoder stages |
| Input | RGB, 384×384 | Clean ImageNet transfer, 3× more pixels than ETASR |
| Loss | BCEDiceLoss | Combination loss proven in v6.5 |
| Optimizer | Adam (decoder lr=1e-3) | Decoder only |
| Batch size | 16 | Fits T4 at 384×384 |
| Epochs | 25 | v6.5 setting |
| Framework | PyTorch + SMP | Native ResNet-34, built-in freeze/unfreeze |
| Augmentation | None | Clean baseline first |

### Success Criteria (vR.P.0)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Tampered Pixel F1 | ≥ 0.35 | Within range of v6.5's 0.41 |
| Pixel IoU | ≥ 0.25 | Reasonable segmentation quality |
| Pixel AUC | ≥ 0.80 | Good discrimination at pixel level |
| Training convergence | Best epoch > 1 | Must actually learn (unlike vR.1.2) |

### Evaluation Metrics (New for Localization)

In addition to the ETASR classification metrics, the pretrained track adds:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Pixel-level F1 | 2×TP / (2×TP + FP + FN) per pixel | Primary localization metric |
| IoU / Jaccard | TP / (TP + FP + FN) | Segmentation quality |
| Dice Score | Same as pixel F1 | Standard segmentation metric |
| Pixel AUC | ROC-AUC on per-pixel probabilities | Threshold-independent |

### Comparison Table Format (Unified Both Tracks)

| Version | Track | Encoder | Input | Test Acc | Macro F1 | AUC | Tam-F1 | Pixel-F1 | IoU |
|---------|-------|---------|-------|----------|----------|-----|--------|----------|-----|
| vR.1.1 | ETASR | 2-layer CNN | ELA 128² | 88.38% | 0.8805 | 0.9601 | 0.8606 | N/A | N/A |
| vR.P.0 | Pretrained | ResNet-34 | RGB 384² | — | — | — | — | — | — |

### Ground Truth Mask Strategy

CASIA v2.0 has inconsistent pixel-level mask coverage. Options:

1. **Use available CASIA masks** — Real GT where available
2. **ELA-thresholded pseudo-masks** — Available for all images but noisy
3. **Classification-first approach** — Start with pretrained classification (no masks needed), then tackle localization after confirming mask availability

**Recommended:** Start with classification using the pretrained encoder as a feature extractor (addresses the ETASR comparison immediately), then investigate mask availability for the localization version.

### Risk Assessment (Pretrained Track)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Pretrained features don't transfer to forensics | Very low (disproven by v6.5) | High | Start with proven ResNet-34 |
| T4 GPU memory overflow at 384×384 | Low | Medium | Use batch_size=8 as fallback |
| ELA destroys pretrained features | Medium | Medium | Test ELA only after RGB baseline (vR.P.2) |
| Overfitting with unfrozen encoder | Medium | Medium | Freeze first, gradual unfreeze with low LR |
| No pixel-level GT masks available | Medium | High | Use pseudo-masks or classification-only approach |
