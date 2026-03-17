# Expected Outcomes -- vR.P.5: ResNet-50 Encoder

| Field | Value |
|-------|-------|
| **Version** | vR.P.5 |
| **Parent** | vR.P.1.5 (ResNet-34, RGB, frozen encoder) |
| **Change** | ResNet-50 encoder (ImageNet pretrained, frozen) |
| **Risk Level** | LOW-MODERATE |

---

## 1. Predictions

### Scenario A: POSITIVE (45% probability)

**Pixel F1 improves by >= 2pp over vR.P.1.5**

ResNet-50's bottleneck blocks produce 2048-dimensional features at the deepest level (vs 512 for ResNet-34) and wider skip connections at every resolution. This richer representation helps the decoder:
- Detect finer tampering boundaries with more precise feature channels
- Exploit higher-dimensional skip connections for better spatial detail
- Distinguish subtle artifacts that lower-dimensional features miss

Expected result: Pixel F1 improves by 2-5pp.

### Scenario B: NEUTRAL (35% probability)

**Pixel F1 within +/- 2pp of vR.P.1.5**

Frozen ImageNet features may perform similarly regardless of encoder depth for this specific domain. The low-level features (edges, textures, gradients) that matter most for forensic localization are similar between ResNet-34 and ResNet-50. The deeper features provide diminishing returns when the encoder is not fine-tuned.

Expected result: Pixel F1 within 2pp of vR.P.1.5.

### Scenario C: NEGATIVE (20% probability)

**Pixel F1 drops by > 2pp from vR.P.1.5**

The wider decoder (~8.7M trainable params vs ~3.1M) may overfit on the relatively small CASIA dataset (~8.8K training images). Additionally, the T4 may struggle with VRAM, potentially requiring a batch size reduction that affects training dynamics.

Risk factors:
- Decoder has ~2.8x more trainable params -> higher overfitting risk
- VRAM pressure at 384x384 with wider activations
- Bottleneck block features may be too abstract for pixel-level forensic tasks

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| **POSITIVE** | Pixel F1 >= vR.P.1.5 + 2pp |
| **NEUTRAL** | Pixel F1 within +/- 2pp of vR.P.1.5 |
| **NEGATIVE** | Pixel F1 < vR.P.1.5 - 2pp |

Note: vR.P.1.5 results are not yet available (pending Kaggle run). The comparison will be made once both versions have been run.

---

## 3. What to Watch For

1. **VRAM usage:** Monitor GPU memory. If OOM occurs, reduce BATCH_SIZE to 8 and note the change.
2. **Convergence speed:** ResNet-50's wider decoder has more params to train -- it may need more epochs to converge. Watch if early stopping triggers too early.
3. **Overfitting signal:** Compare train loss vs val loss gap. If the gap is significantly larger than vR.P.1.5, the wider decoder is overfitting.
4. **Per-epoch time:** Expect ~30-40% longer per epoch compared to ResNet-34 due to wider activations.
5. **Pixel F1 trajectory:** Plot pixel F1 over epochs. A rapid initial rise followed by plateau suggests the features are useful; a slow crawl suggests ResNet-50's features are not adding value over ResNet-34.

---

## 4. Comparison with Prior Versions

| Version | Encoder | Encoder State | Decoder Params | Status |
|---------|---------|---------------|----------------|--------|
| vR.P.1 | ResNet-34 | Frozen | ~3.1M | Pending |
| vR.P.1.5 | ResNet-34 | Frozen + speed opts | ~3.1M | Pending |
| **vR.P.5** | **ResNet-50** | **Frozen** | **~8.7M** | **This experiment** |
| vR.P.6 (planned) | EfficientNet-B0 | Frozen | TBD | Future |

### Why ResNet-50 Specifically?

- **Minimal code change:** Only the encoder name string changes. SMP handles everything else automatically.
- **Well-understood architecture:** ResNet-50 is heavily benchmarked on ImageNet and downstream tasks.
- **Natural progression:** ResNet-34 -> ResNet-50 is the standard depth scaling within the ResNet family.
- **Bottleneck vs basic blocks:** This is the key architectural difference -- testing whether bottleneck features generalize better to forensic localization.

---

## 5. If NEGATIVE -- Next Steps

- **Check decoder overfitting:** If train-val gap is large, consider adding dropout to decoder or reducing LR.
- **Try with partial unfreeze:** Combine ResNet-50 with gradual unfreeze (vR.P.2 approach) to adapt deeper features.
- **Move to EfficientNet-B0 (vR.P.6):** A parameter-efficient encoder may perform better than brute-force depth scaling.
- **Conclusion:** If both ResNet-34 and ResNet-50 frozen produce similar results, encoder depth is not the bottleneck -- the domain gap (ImageNet vs forensic images) matters more than depth.
