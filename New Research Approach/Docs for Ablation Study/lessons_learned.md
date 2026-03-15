# Lessons Learned -- ETASR Ablation Study

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Retrospective insights from the complete ETASR and Pretrained ablation studies |
| **Paper** | ETASR_9593 -- "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Versions Covered** | ETASR: vR.1.0--vR.1.7 / Pretrained: vR.P.0--vR.P.6 |

---

## 1. Methodology Lessons

### Lesson 1: Honest baselines are the most important experiment

vR.1.1 (eval fix) revealed that the paper's 89.89% "accuracy" was inflated by validating on a subset with no held-out test set. The honest test accuracy was 88.38% -- 1.51pp lower. Without this correction, all subsequent experiments would have been measured against a biased reference. **Every ablation study must start with an honest evaluation.**

### Lesson 2: Single-variable control is essential but ordering matters

The study correctly maintained single-variable control. However, the order of experiments was suboptimal. Training-trick experiments (class weights, BN, LR scheduler) on a fundamentally limited architecture produced marginal improvements. The architecture change (vR.1.6) had 2-10x more impact. In hindsight:

- **Better order:** eval fix → deeper CNN → GAP → then training tricks
- **Why it matters:** testing training tricks on a broken architecture wastes experiments and creates a misleading narrative of diminishing returns

### Lesson 3: Explicit verdict criteria prevent post-hoc rationalisation

The study defined clear thresholds: POSITIVE (>+0.5pp), NEUTRAL (+/-0.5pp), NEGATIVE (<-0.5pp). This forced honest assessment of vR.1.4 (NEUTRAL despite BN being a "standard" practice) and vR.1.2 (REJECTED despite augmentation being "best practice"). Without these criteria, there would be temptation to claim all changes as improvements.

### Lesson 4: Rejected experiments are valuable data

vR.1.2 (augmentation REJECTED) provided the most actionable insight of the early series: the Flatten->Dense architecture memorises spatial positions. This diagnosis directly motivated vR.1.6 (deeper CNN) and vR.1.7 (GAP). Negative results are not failures -- they are diagnostic tools.

---

## 2. Architecture Lessons

### Lesson 5: Parameter location matters more than parameter count

| Architecture | Total Params | Test Accuracy | Key Difference |
|-------------|-------------|---------------|----------------|
| vR.1.1 (Flatten) | 29.5M | 88.38% | 99.9% in Dense |
| vR.1.7 (GAP) | 64K | 89.17% | 72.8% in Conv |

The 64K-parameter model outperforms the 29.5M-parameter model because parameters in convolution layers (feature extraction) are more valuable than parameters in dense layers (spatial memorisation). **Where parameters live matters more than how many there are.**

### Lesson 6: The Flatten->Dense bottleneck is the defining weakness of ETASR

The original architecture placed 99.9% of parameters in `Flatten(115,200) -> Dense(256)`. This made the model:
- Vulnerable to augmentation (spatial positions change)
- Prone to overfitting (29.5M params for ~8.8K training images)
- Resistant to training improvements (tricks can't fix a structural problem)
- Incapable of transfer (memorised spatial positions don't generalise)

Every architectural improvement (vR.1.6 reducing it, vR.1.7 eliminating it) produced meaningful gains.

### Lesson 7: Spatial information matters for ELA detection

vR.1.6 (Flatten, 90.23%) outperforms vR.1.7 (GAP, 89.17%) despite having 216x more parameters. This proves that WHERE tampering artifacts appear in the ELA map is informative. The ideal architecture would preserve spatial information without the Flatten->Dense bottleneck -- perhaps through spatial attention mechanisms or dilated convolutions.

### Lesson 8: The overfitting-capacity sweet spot

| Model | Params | Train-Val Gap | Test Acc |
|-------|--------|---------------|----------|
| vR.1.5 | 29.5M | ~6pp | 88.96% |
| vR.1.6 | 13.8M | ~5pp | 90.23% |
| vR.1.7 | 64K | ~1.3pp | 89.17% |

vR.1.6 slightly overfits but achieves the best accuracy. vR.1.7 barely overfits but has lower accuracy. The optimal model has enough capacity to learn complex patterns but not so much that it memorises training data.

---

## 3. Training Lessons

### Lesson 9: Class weights help accuracy but not AUC

Class weights (vR.1.3) improved test accuracy by +0.79pp but AUC actually dropped 0.0021. Class weights shift the decision threshold to account for class imbalance, but they don't improve the model's ability to separate the two distributions. For forensic detection where threshold calibration matters, class weights are useful. For fundamental feature quality, they are neutral.

### Lesson 10: BatchNorm can hurt when the architecture is the bottleneck

BN (vR.1.4) was expected to improve training stability and generalisation. Instead, it introduced a catastrophic epoch 1 spike (val_loss=16.13) and shortened productive training from 14 to 3 epochs. BN normalises activations, but when the primary source of variance is the 29.5M Dense layer, BN cannot stabilise what is fundamentally an overfitting problem. BN was useful in vR.1.6+ where the architecture was more balanced.

### Lesson 11: LR schedulers provide infrastructure, not improvement

ReduceLROnPlateau (vR.1.5) gave only +0.21pp -- marginal. However, it enabled vR.1.6 to train for 18 epochs (the LR was reduced twice: 1e-4 → 5e-5 → 2.5e-5). The scheduler's value was realised one version later, when the architecture could actually benefit from extended training. **Training infrastructure should be judged by its potential, not its immediate impact.**

---

## 4. Data Lessons

### Lesson 12: Augmentation failure is diagnostic, not just negative

vR.1.2 failed because the Flatten layer treats pixel position as a feature. This diagnosis was the most important finding of the early series:
- It explained why the model overfit (memorising positions, not learning features)
- It motivated the architectural changes in vR.1.6/1.7
- It predicted that GAP would reduce overfitting (by removing positional dependence)

### Lesson 13: ELA preprocessing is robust but limited

All 8 versions used identical ELA preprocessing (Q=90, brightness scaling, /255.0). This pipeline was never the limiting factor. However, ELA's limitations are real:
- Non-JPEG images produce empty/noisy ELA maps
- Brightness scaling normalises away magnitude information
- 128x128 resolution loses fine-grained compression artifacts

---

## 5. Evaluation Lessons

### Lesson 14: ROC-AUC is the best single metric

Accuracy can be gamed by threshold shifting (class weights improve accuracy without improving features). Per-class F1 can be inflated by biasing toward one class. ROC-AUC measures threshold-independent discriminatory power -- how well the model separates authentic and tampered distributions.

**Only vR.1.6 improved AUC from baseline.** This means only vR.1.6 actually improved the model's feature quality. Everything else was threshold manipulation.

### Lesson 15: Per-class metrics reveal what averages hide

Weighted accuracy masks the precision-recall tradeoff. From vR.1.1 to vR.1.7:
- Tp Recall improved: 0.8830 → 0.9467 (+6.37pp)
- Au Recall barely changed: 0.8577 → 0.8541 (-0.36pp)
- Tp Precision dropped: 0.8393 → 0.8161 (-2.32pp)

The model became a "tampered image detector" rather than a balanced classifier. This is only visible in per-class metrics.

### Lesson 16: Confusion matrix analysis > aggregate metrics

The FN rate trajectory (11.7% → 5.3%) shows steady improvement in catching tampered images. The FP rate trajectory (14.2% → 14.6%) shows no real progress in avoiding false accusations. These trends are invisible in aggregate accuracy but critical for real-world deployment.

---

## 6. What Would Be Done Differently

### If Starting the Ablation Study Over

| Step | Original Order | Better Order | Why |
|------|---------------|-------------|-----|
| 1 | vR.1.1: Eval fix | vR.1.1: Eval fix | Same (essential first step) |
| 2 | vR.1.2: Augmentation | vR.1.6: Deeper CNN | Architecture before training tricks |
| 3 | vR.1.3: Class weights | vR.1.7: GAP | Test both pooling strategies early |
| 4 | vR.1.4: BatchNorm | vR.1.3: Class weights | Now test on the better architecture |
| 5 | vR.1.5: LR Scheduler | vR.1.4: BatchNorm | BN on balanced architecture |
| 6 | vR.1.6: Deeper CNN | vR.1.2: Augmentation on GAP | Re-test augmentation without spatial memorisation |
| 7 | vR.1.7: GAP | vR.1.5: LR Scheduler | Infrastructure last |

### Key Changes

1. **Architecture first.** The deeper CNN and GAP experiments would have revealed the Flatten bottleneck immediately, avoiding 3 marginal experiments.
2. **Re-test augmentation on GAP.** vR.1.2 was rejected because the architecture memorises positions. With GAP, augmentation might succeed -- this experiment was never run.
3. **BN after architecture.** BN's epoch 1 catastrophe was amplified by the 29.5M-param architecture. On a 64K-param architecture, BN might behave differently.

---

## 7. Transferable Principles

### For Future Ablation Studies

1. **Always start with an honest baseline.** Validate your evaluation before testing anything else.
2. **ROC-AUC reveals feature quality.** If AUC doesn't improve, the model isn't learning better features -- it's just shifting thresholds.
3. **Architecture changes beat training tricks.** For CNN models with structural bottlenecks, fix the architecture first.
4. **Rejected experiments are diagnostic.** Understand WHY something failed before moving on.
5. **Parameter location > parameter count.** A model with 64K well-placed params can outperform one with 29.5M poorly-placed params.
6. **Training tricks have infrastructure value.** LR scheduling didn't help v1.5, but it enabled vR.1.6's longer training.
7. **BN can hurt on broken architectures.** Don't assume "standard practices" are always improvements.

### For ELA-Based Forensic Detection

1. **Spatial information matters.** ELA maps have spatial structure that contains forensic signal.
2. **Don't augment with geometric transforms** when the architecture is spatially sensitive.
3. **ELA at 128x128 has a ceiling.** Higher resolution may unlock additional performance.
4. **Classification is not localization.** Even at 90%+ accuracy, a classifier cannot tell WHERE tampering occurred.

---

## 8. Open Questions

### Unanswered by This Study

1. **Would augmentation work on the GAP architecture?** The spatial-memorisation argument suggests it might. This experiment was never run.

2. **Would higher resolution help?** 128x128 loses significant detail. 256x256 or 384x384 ELA maps might push accuracy past 92%.

3. **Is the 96.21% paper claim achievable?** The 5.98pp gap persists despite 7 ablations. It may be unreproducible due to evaluation methodology differences.

4. **Can attention mechanisms replace GAP?** Spatial attention would preserve positional information while reducing the Dense parameter bottleneck.

5. **How does pretrained localization compare?** The vR.P.x track uses ResNet-34/50 encoders that dwarf the ETASR CNN in representational power. Direct comparison of detection accuracy would reveal whether the ETASR architecture is fundamentally limited.

6. **Would ensemble methods help?** Combining vR.1.6 (best accuracy, spatial features) with vR.1.7 (best Tp recall, channel features) could exploit both representations.

---

## 9. Pretrained Track Lessons

### Lesson 17: Input representation dominates encoder architecture

The single most impactful variable in the pretrained series was switching from RGB to ELA input (P.3: +23.74pp Pixel F1). By comparison, swapping from ResNet-34 to ResNet-50 (P.5: +5.91pp) or EfficientNet-B0 (P.6: +6.71pp) produced 3-4x less improvement. **What the model sees matters more than how it sees it.**

### Lesson 18: BN unfreeze enables domain adaptation without overfitting

P.3's "frozen body + BN unfrozen" strategy unfreezes only 17K BatchNorm parameters while keeping 21.3M conv weights frozen. This allows the encoder's running statistics to adapt to ELA's distribution (mean ~0.05, very different from ImageNet's ~0.45) without the overfitting risk of unfreezing conv weights. P.2's aggressive unfreeze (layer3+layer4, 23M trainable) produced worse results than P.3's conservative approach (3.17M trainable).

### Lesson 19: Larger decoders don't proportionally improve results

ResNet-50 (P.5) forced SMP to create a 9M-parameter decoder (vs ResNet-34's 3.15M) due to 4x wider skip connections [256,512,1024,2048]. Despite 2.86x more trainable parameters, P.5's Pixel F1 (0.5137) barely exceeded P.1's (0.4546). The data:param ratio (1:1,021 vs 1:357) makes the larger decoder a net negative for generalization.

### Lesson 20: ELA-specific normalization is critical

P.3 computed the ELA distribution from 500 training samples: mean=[0.0497, 0.0418, 0.0590], std=[0.0663, 0.0570, 0.0756]. These values are ~10x smaller than ImageNet stats. Using ImageNet normalization on ELA images would push activations far outside the encoder's expected range, degrading feature quality. Always compute domain-specific normalization when input distribution differs from pretraining.

### Lesson 21: 4-channel fusion provides marginal benefit at high complexity cost

P.4 concatenated RGB (3ch) + ELA grayscale (1ch) into a 4-channel input with dual normalization. Despite this complexity (conv1 unfreeze, dual stats, training instability), the Pixel F1 gain over ELA-only (P.3) was only +1.33pp — below the ±2pp significance threshold. The FP rate more than doubled (2.7% → 6.4%). **Simpler is better: ELA-only provides 98% of the benefit.**

### Lesson 22: Copy-paste bugs undermine trust in experimental notebooks

P.5's model save filename says "resnet34" instead of "resnet50", and the comparison table prints the wrong encoder name. These cosmetic bugs don't affect training results but they signal insufficient review. In a series where single-variable control is paramount, output errors raise doubt about whether the right variable was actually changed. **Always diff your notebook against the parent before execution.**

### Lesson 23: Methodology consistency matters for cross-experiment comparison

P.6 omits AMP, TF32, and drop_last (branching from P.1), while P.5 includes them (branching from P.1.5). This confounds the ResNet-50 vs EfficientNet-B0 comparison. A rigorous encoder ablation would hold infrastructure constant. **When designing parallel experiments, ensure they share the same baseline infrastructure.**

### Lesson 24: Test your visualization code before long training runs

P.3's NameError (`denormalize` vs `denormalize_ela`) is a trivial fix but it prevented saving the best model in the series. A 1-minute pre-run test of visualization cells would have caught this. **The cost of a test cell is negligible; the cost of losing a trained model is enormous.**

---

## 10. Cross-Track Synthesis

### The Hierarchy of Impact (Both Tracks Combined)

```
1. INPUT REPRESENTATION (what the model sees)
   - ETASR: ELA at 128x128 is the foundation (never changed)
   - Pretrained: ELA input → +23.7pp Pixel F1 over RGB
   → Largest single-variable impact in either track

2. ARCHITECTURE (how the model processes input)
   - ETASR: Deeper CNN → +1.27pp accuracy
   - Pretrained: Encoder swap → +6.7pp Pixel F1
   → Second most impactful

3. TRAINING STRATEGY (how the model learns)
   - ETASR: Class weights → +0.79pp, LR scheduler → +0.21pp
   - Pretrained: BN unfreeze → enables P.3 breakthrough, gradual unfreeze → +5.7pp
   → Variable impact, depends on architecture quality

4. INFRASTRUCTURE (speed/precision optimizations)
   - ETASR: Not tested
   - Pretrained: AMP/TF32 → NEUTRAL (no quality impact)
   → Negligible quality impact, but saves time
```

### The Central Lesson

**Both tracks converge on the same conclusion: fix the signal before fixing the model.** In the ETASR track, the model was stuck at ~89% until the architecture was restructured (vR.1.6). In the pretrained track, the model was stuck at ~0.45 Pixel F1 until the input was changed to ELA (P.3). In both cases, training tricks produced diminishing returns on a fundamentally constrained setup.

### What We'd Do Differently (Both Tracks)

1. **Start with input experiments.** Test ELA vs RGB vs ELA+RGB before any encoder or training experiments.
2. **Hold infrastructure constant.** All experiments should share the same AMP/TF32/workers settings.
3. **Increase max_epochs conservatively.** P.3 was still improving at epoch 25. Use max_epochs=50 or even 100 with patience=10.
4. **Test visualization code before training.** A 1-minute smoke test of all cells (with a single batch) catches bugs like P.3's NameError.
5. **Run encoder experiments on the best input.** P.5/P.6 tested encoders on RGB. Testing them on ELA would provide more actionable insights.

---

## 9. Final Research Summary

### Best Configuration

**vR.1.6 (Deeper CNN)** is the best single model:
- 90.23% test accuracy
- 0.9004 Macro F1 (first to cross 0.90)
- 0.9657 ROC-AUC (best in series, only version to improve from baseline)
- 13.8M parameters (53% reduction from original)

### Major Architectural Improvements

1. **Deeper feature extraction** (vR.1.6): Adding a 3rd conv layer + MaxPool significantly improved all metrics by reducing the Flatten->Dense bottleneck.
2. **Global Average Pooling** (vR.1.7): Eliminated the bottleneck entirely. Despite lower accuracy, proves convolutional features alone carry forensic signal.

### Limitations of the ETASR Architecture

1. **No localization capability.** The classifier outputs a binary label, not a pixel-level mask. This fails the assignment's core requirement.
2. **Low resolution** (128x128). ELA artifacts are fine-grained and require higher resolution for reliable detection.
3. **Shallow feature extraction.** Even with 3 conv layers, the model has only 46,560 feature extraction parameters. Modern architectures (ResNet, EfficientNet) have orders of magnitude more.
4. **Paper gap unreduced.** The best honest accuracy (90.23%) remains 5.98pp below the paper's claim (96.21%).
5. **No robustness.** The model has not been tested against JPEG recompression, resizing, cropping, or noise -- all of which would degrade ELA quality.

### The Two-Track Conclusion

The ETASR track demonstrates that the paper's architecture is fundamentally limited for the assignment's requirements. The pretrained localization track (vR.P.x) addresses these limitations with:
- Pixel-level mask prediction (assignment requirement)
- ImageNet-pretrained features (orders of magnitude more representational power)
- 384x384 resolution (3x the detail)
- UNet architecture with skip connections (spatial information preserved at all resolutions)
