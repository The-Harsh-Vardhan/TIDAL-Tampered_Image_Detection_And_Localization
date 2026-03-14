# Architecture Analysis Report

| Field | Value |
|-------|-------|
| **Date** | 2026-03-14 |
| **Scope** | Architecture selection for tampered image detection on CASIA v2.0 |
| **Selected** | ETASR_9593: ELA + CNN (Gorle & Guttavelli, 2025) |

---

## 1. Candidate Architecture Comparison

| # | Paper | Method | Task | CASIA v2.0 Result | Params | T4 Fit | Code Available | Verdict |
|---|-------|--------|------|-------------------|--------|--------|---------------|---------|
| P7 | **ETASR_9593** | ELA + 2-layer CNN | Classification | **96.21% acc** | ~29.5M | Yes | Partial (reconstructed) | **SELECTED** |
| P1 | ELA-CNN Hybrid (IEEE) | ELA + CNN | Classification | 87.75% acc | Unknown | Yes | No | Rejected (lower accuracy) |
| P4 | U-Net Mixed Tampering | U-Net + FENet fusion | Localization | ~95% acc | Large | Tight | No | Rejected (no code, complex) |
| P13 | EMT-Net | Swin + ResNet + SRM | Localization | AUC=0.987 (NIST) | Very Large | No | No | Rejected (GPU-heavy) |
| P17 | ME-Net | ConvNeXt + ResNet-50 | Localization | F1=0.905 (NIST) | Very Large | No | No | Rejected (GPU-heavy) |
| P21 | TransU2-Net | U2-Net + attention | Localization | F-meas=0.735 | Large | Tight | No | Rejected (no code, complex) |
| P16 | Weber Descriptors | YCbCr + SVM | Classification | 96.52% acc | Small | Yes | No | Rejected (traditional ML, no DL) |
| P20 | Hybrid DCCAE | VGGNet + Capsule AE | Classification | 99.23% (CASIA V1) | Large | No | No | Rejected (different dataset, complex) |

---

## 2. Selection Rationale: ETASR_9593

### Why ETASR Was Selected

1. **Reproducibility**: The paper describes a simple, fully-specified architecture (Table III). Every layer, activation, and hyperparameter is documented. No ambiguity.

2. **CASIA v2.0 Compatibility**: The paper was benchmarked directly on CASIA v2.0 -- the exact dataset required by the assignment.

3. **High Accuracy**: 96.21% reported accuracy outperforms VGG16 (90.32%), VGG19 (88.92%), and ResNet101 (74.75%) on the same dataset.

4. **Lightweight**: A 2-layer CNN trains in minutes on a T4 GPU. No pretrained weights needed. No large encoder downloads.

5. **Code Availability**: Two reference implementations (`CASIA2code.py`, `code.py`) were found. Both had critical bugs but provided enough code to reconstruct a correct implementation.

6. **ELA Preprocessing**: Error Level Analysis is a proven forensic technique that transforms tamper detection from a generic vision problem into a forensic feature classification problem. The preprocessing does the heavy lifting; the CNN is simple.

### Why Others Were Rejected

| Architecture | Rejection Reason |
|---|---|
| EMT-Net (P13) | Requires Swin Transformer + ResNet + SRM. No code available. Would not fit on T4 GPU. |
| ME-Net (P17) | Dual-branch ConvNeXt + ResNet-50. No code available. GPU requirements exceed T4. |
| TransU2-Net (P21) | No code available. U2-Net base requires significant implementation effort. |
| U-Net Mixed (P4) | FENet spatial-frequency fusion is complex. No code or clear architecture spec. |
| Hybrid DCCAE (P20) | Benchmarked on CASIA V1 (not V2). Extreme preprocessing pipeline. |
| Weber Descriptors (P16) | Traditional ML (SVM). Does not satisfy the deep learning requirement. |

---

## 3. Architecture Details

### ETASR CNN Architecture (Table III)

```
Input: 128x128x3 (ELA image)
  |
  v
Conv2D(32, 5x5, valid, ReLU)     -> (124, 124, 32)   [2,432 params]
Conv2D(32, 5x5, valid, ReLU)     -> (120, 120, 32)   [25,632 params]
MaxPooling2D(2x2)                 -> (60, 60, 32)     [0 params]
Dropout(0.25)                     -> (60, 60, 32)     [0 params]
Flatten                           -> (115,200)        [0 params]
Dense(256, ReLU)                  -> (256)            [29,491,456 params]
Dropout(0.5)                      -> (256)            [0 params]
Dense(2, Softmax)                 -> (2)              [514 params]
                                                       ─────────────
                                    Total:             ~29.5M params
```

### Training Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | Adam | Paper |
| Learning rate | 0.0001 | Paper |
| Loss | Categorical cross-entropy | Paper |
| Batch size | 32 | Paper |
| Max epochs | 50 | Paper |
| Early stopping | val_accuracy, patience=5 | Paper (implied) |
| Input size | 128x128x3 | Paper Table III |
| ELA quality | 90 | Paper |

### ELA Preprocessing Pipeline

```
Raw Image (any format)
  -> Convert to RGB
  -> Re-save as JPEG at Q=90 (in-memory via BytesIO)
  -> Compute |Original - Resaved| pixel-wise
  -> Scale brightness to [0, 255] range
  -> Resize to 128x128
  -> Normalize to [0, 1]
  -> Feed to CNN
```

---

## 4. Reference Code Audit Summary

Two reference implementations were audited. **11 bugs were identified and fixed:**

| # | Bug | Severity | Fix |
|---|-----|----------|-----|
| 1 | Image size 150x150 (paper says 128) | Critical | Use 128x128 |
| 2 | Dense(150) not Dense(256) | Critical | Use Dense(256) |
| 3 | Sigmoid output not Softmax | Critical | Use Softmax |
| 4 | random.shuffle(X) without Y | Fatal | sklearn paired shuffle |
| 5 | Dense(1, softmax) always outputs 1.0 | Fatal | Dense(2, softmax) |
| 6 | Double reshape crash (150->128) | Fatal | Single consistent size |
| 7 | Temp file ELA (not thread-safe) | Medium | In-memory BytesIO |
| 8 | binary_crossentropy with one-hot | Medium | categorical_crossentropy |
| 9 | Early stopping commented out | Medium | Enabled |
| 10 | val_acc (deprecated) | Low | val_accuracy |
| 11 | Only loads .jpg files | Low | All formats supported |

---

## 5. Comparison with Prior Project Architectures

| Version | Architecture | Best Tam-F1 | Accuracy | Status |
|---------|-------------|------------|----------|--------|
| v6.5 | SMP UNet + ResNet34 | 0.4101 | 0.8246 | Project best (localization) |
| vK.10.6 | Custom UNet (scratch) | 0.2213 | 0.8357 | Best from-scratch |
| vK.11.4-12.0b | Synthesis (SMP+ELA+Edge+CLS) | 0.1321 | 0.4062 | **FAILED** |
| **vR.0** | **ELA + CNN (ETASR)** | **N/A (classification)** | **~96% (expected)** | **New approach** |

The ETASR approach trades localization capability for classification reliability. Given that 6 consecutive synthesis runs failed catastrophically, a simpler, proven approach was the pragmatic starting choice.

---

## 6. Pretrained Model Track: The Path to Localization

### Why Pretrained Models Are Now Being Pursued

The ETASR CNN has a **hard ceiling**: it produces a single binary label (Authentic/Tampered) and **cannot generate pixel-level localization masks** — which is the assignment's core requirement. The ablation study (vR.1.x) has also stalled after vR.1.2 augmentation regressed by -2.85pp and was rejected.

Three lines of evidence support the pretrained approach:

1. **Project history:** v6.5 (SMP UNet + ResNet-34, ImageNet pretrained) achieved Tam-F1 = 0.41 for pixel-level segmentation — nearly **2× better** than the best from-scratch model (Tam-F1 = 0.22).
2. **Literature:** 19 of 21 surveyed papers use or reference pretrained encoders. ME-Net (ResNet-50 backbone) achieves F1 = 0.905.
3. **Assignment:** "The choice of architecture is entirely up to you." No restriction on pretraining. Only constraint is T4 GPU compatibility.

### Pretrained Candidate Architectures

| # | Architecture | Type | Params (Total) | Trainable (frozen) | ImageNet Top-1 | T4 Memory | Project Evidence | Verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | **ResNet-34 + UNet** | Encoder-Decoder | 21.3M | ~500K | 73.3% | ~3 GB | **v6.5: Tam-F1=0.41** | **PRIMARY** |
| 2 | ResNet-50 + UNet | Encoder-Decoder | 23.5M | ~600K | 76.1% | ~4 GB | Untested | Secondary |
| 3 | EfficientNet-B0 + UNet | Encoder-Decoder | 5.3M | ~400K | 77.1% | ~2.5 GB | Untested | Experimental |

### Decision Matrix (Weighted Scoring)

| Criterion (Weight) | ResNet-34 | ResNet-50 | EfficientNet-B0 |
|---------------------|-----------|-----------|-----------------|
| Project evidence (30%) | **10** (v6.5 proven) | 3 (untested) | 3 (untested) |
| Literature support (20%) | 8 | **9** (ME-Net) | 6 (survey mention) |
| Parameter efficiency (10%) | 6 | 5 | **10** |
| T4 compatibility (10%) | 9 | 8 | **10** |
| Feature quality (15%) | 7 | **9** | 8 |
| Implementation risk (15%) | **10** (SMP native) | **10** (SMP native) | 8 (SMP native) |
| **Weighted Score** | **8.4** | **7.0** | **6.5** |

**Ranking:** ResNet-34 first (proven), ResNet-50 second (deeper features), EfficientNet-B0 third (efficiency).

### Data Efficiency Advantage

| Approach | Training Images | Trainable Params | Data:Param Ratio |
|----------|----------------|------------------|------------------|
| ETASR CNN (current) | 8,829 | 29,520,034 | **1 : 3,343** |
| ResNet-34 (frozen) + head | 8,829 | ~500,000 | **1 : 57** |

The pretrained approach achieves a **60× better data efficiency ratio** by freezing the backbone and only training the lightweight decoder.

### ELA Compatibility

ImageNet pretrained models expect natural RGB images; ELA maps have fundamentally different statistics. Three input strategies are viable:

| Strategy | Input | Transfer Quality | Risk | Project Evidence |
|----------|-------|-----------------|------|-----------------|
| **A: RGB only** | Raw RGB images | Excellent | Very low | v6.5: Tam-F1=0.41 |
| B: ELA only | ELA maps (3ch) | Partial (early layers) | Medium | Untested |
| C: RGB + ELA | 4-channel (RGB+ELA) | Good (RGB), uncertain (ELA) | Medium-High | vK.11-12 failed (but confounded) |

**Recommendation:** Start with Strategy A (RGB only) where transfer is guaranteed, then test ELA as an ablation.

### Pretrained vs ETASR: Side-by-Side

| Aspect | ETASR CNN (vR.1.x track) | Pretrained ResNet-34 (vR.P.x track) |
|--------|--------------------------|--------------------------------------|
| Output | Binary label (Au/Tp) | **Pixel-level mask** |
| Localization | ❌ Cannot localize | ✅ Full pixel-level localization |
| Assignment alignment | Classification only | **Satisfies all requirements** |
| Input | ELA maps (128×128) | RGB images (384×384) |
| Trainable params | 29.5M (all) | ~500K (decoder only) |
| Data efficiency | 1 : 3,343 | 1 : 57 |
| Framework | TensorFlow/Keras | PyTorch + SMP |
| Best result | 88.38% test acc (vR.1.1) | Tam-F1 = 0.41 (v6.5) |

### Two-Track Strategy

Both tracks continue in parallel:

- **Track 1 (ETASR, vR.1.x):** Continues the ablation study for classification. Documents experimental methodology and understanding of the paper. Academic value.
- **Track 2 (Pretrained, vR.P.x):** Achieves the assignment's localization requirement with competitive results. Final submission notebook will use this track's best model.

See `Pretrained Models/` folder for full analysis:
- `01_Feasibility_Analysis.md` — Evidence and recommendation
- `02_Architecture_Comparison.md` — Deep architecture dive
- `03_Implementation_Strategy.md` — Phased rollout plan and code
- `04_ELA_Compatibility_Analysis.md` — BatchNorm domain shift analysis
