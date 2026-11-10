# 03 — Model Architecture Roast

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Summary

The v9 architecture is more ambitious than v8 and introduces two real design improvements: a learned classification head and an ELA input channel. Both improvements are unverified. One of them has a silent correctness risk that could invalidate the "pretrained encoder" claim entirely.

---

## 1. Base Architecture: SMP U-Net / ResNet34

### Choice rationale (stated)
Standard segmentation baseline; reliable, well-supported in SMP, fits on T4.

### Critique

This is still a convenience selection dressed up as a design decision. The assignment asks for "thoughtful architecture choices." What has been done is:

- Choose the most commonly demonstrated architecture in SMP documentation.
- Justify it by pointing at parameter count and runtime fit.
- Add progressively more complexity on top of it in hopes of earning the "thoughtful" label.

That is not thoughtfulness. That is incremental patch accumulation on top of a default choice.

The real question that has never been answered is: **why does a standard RGB semantic segmentation backbone make sense for image forensics?**

Image forensics is an anomaly detection problem. The signal is subtle re-compression artifacts, DCT coefficient inconsistencies, and noise pattern discontinuities — not semantic content. ResNet34 was trained to classify dogs and cars. Its skip connections preserve semantic features. The features most useful for forgery detection are often exactly the kind of fine-grained, low-level statistical artifacts that deep semantic encoders destroy when they pool and downsample.

The assignment does not require a forensic-grade solution, so this can be a "practical baseline" claim. Make that claim explicitly. Do not keep implying the architecture is "well-reasoned."

---

## 2. ELA 4-Channel Input — The Silent Correctness Risk

This is the most dangerous part of the v9 architecture.

### The problem

ResNet34 has a first convolutional layer with weight shape `[64, 3, 7, 7]` — 64 output filters, 3 input channels, 7×7 kernel. When `in_channels = 4`, SMP must adapt this layer to accept 4 channels.

The notebook sets:
```python
CONFIG["in_channels"] = 4 if CONFIG["use_ela"] else 3
```

And passes `in_channels=4` to `smp.Unet(...)`. SMP handles this one of two ways depending on the version:

1. **Random initialisation of the new weights** — The 3-channel pretrained weights are retained and a random 4th channel weight is appended. The ELA channel gets random gradients. The "ImageNet pretrained" justification is partially invalidated.
2. **Initialisation by averaging** — Some SMP versions initialise the 4th channel by averaging the 3-channel weights. This preserves some pretraining signal but is not documented in this notebook.

**The notebook never clarifies which strategy SMP is using.**  
There is no `print(model.encoder.layer0[0].weight.shape)` or equivalent. There is no mention of this issue in the architecture description or the comments. A reviewer trying to verify the notebook cannot confirm the pretrained claim.

This is not a hypothetical risk. It is an unknown that will affect training stability and any comparison with the v8 RGB-only baseline.

---

## 3. Dual-Task Classification Head

### Design

```
GlobalAveragePool(bottleneck features) → Linear(512, 128) → ReLU → Linear(128, 1) → sigmoid
```

This is attached to the encoder bottleneck. BCE classification loss weighted at 0.5.

### What is correct here

Adding a learned classification head is the right correction. The assignment requires image-level tamper detection. Calling `mask_pred.max()` as a heuristic is not satisfying that requirement. A jointly trained classifier learns from both the segmentation objective AND the explicit binary supervision — this is design improvement, full stop.

### Problems with the implementation

**Problem 1 — Why 128 hidden units?**  
No justification. The bottleneck has 512 channels. Going directly 512 → 128 → 1 with a single hidden layer is arbitrary. This is not a critical flaw, but undocumented hyperparameters are exactly what a design review flags.

**Problem 2 — No gradient conflict analysis.**  
A classification BCE loss with weight 0.5 on the encoder bottleneck means the encoder gradients are pulled simultaneously toward "produce a good segmentation map" and "produce a good classification signal." These objectives can conflict. When the segmentation says a region is slightly tampered but the classification says "authentic," the encoder receives contradictory gradient signals. There is no mention of this tension.

**Problem 3 — The classification head output is reported but never used for threshold selection.**  
Looking at the evaluation pipeline, `best_threshold` is still determined by scanning the segmentation mask probabilities, not the classification head output. So the dual-task head feeds into metrics but does not change the primary tamper detection decision path in an obvious way.

---

## 4. Edge Loss

```python
edge_loss = F.binary_cross_entropy_with_logits(
    seg_logits,
    gt_mask,
    weight=1.0 + edge_loss_weight * edge_mask
)
```

### What this does
Up-weights the BCE loss at boundary pixels by a factor of (1 + `edge_loss_weight`). This is a reasonable boundary emphasis technique.

### Problems

**Problem 1 — `edge_loss_weight = 2.0` and `edge_loss_lambda = 0.3` are arbitrary.**  
The combined loss is `seg_total + 0.5 * cls_loss + 0.3 * edge_loss`. The edge loss itself is a weighted BCE loss that uses weights up to 3.0 (1 + 2.0) on boundary pixels. No ablation determines whether 0.3 is too high or too low. These numbers were picked by intuition and put into config.

**Problem 2 — The edge mask is computed from the ground truth mask using `find_boundaries`.**  
If the ground truth masks in CASIA have soft anti-aliased edges or imprecise boundaries (which they often do — JPEG-block-aligned mask edges are common), then the "edge" pixels being up-weighted are themselves noisy. You are training the model to be extra confident at the most unreliable annotation boundaries.

---

## 5. Architecture Comparison (DeepLabV3+)

The notebook defines `run_architecture_comparison()` but it is **disabled by default** (`"run_architecture_comparison": False`). Since the notebook has never been executed, this comparison has never produced any result. The architectural justification that "U-Net ResNet34 is better than DeepLabV3+" has zero supporting evidence.

---

## Architecture Comparison Table

| Feature | v8 | v9 | Status |
|---------|----|----|--------|
| Segmentation backbone | U-Net + ResNet34 | U-Net + ResNet34 | Same |
| Input channels | 3 (RGB) | 4 (RGB + ELA) | Untested |
| Classification head | Heuristic max | Learned FC | Better design, untested |
| Loss terms | 2 (BCE + Dice) | 4 (BCE + Dice + edge + cls) | More complex, untested |
| Pretrained claim accuracy | Clear (3-ch ImageNet) | Ambiguous (4-ch adaptation) | Risk |
| Architecture justified | Weakly | Weakly | No change |
| Any comparison run | No | No (disabled) | No improvement |

The architecture got more complex. The justification did not improve. The risk profile increased. No execution confirmed any of it.
