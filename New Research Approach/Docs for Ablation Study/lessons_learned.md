# Lessons Learned -- ETASR Ablation Study

| Field | Value |
|-------|-------|
| **Date** | 2026-03-16 |
| **Scope** | Retrospective insights from the complete ETASR and Pretrained ablation studies |
| **Paper** | ETASR_9593 -- "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Versions Covered** | ETASR: vR.1.0--vR.1.7 / Pretrained: vR.P.0--vR.P.30.4 / Standalone: 3 runs |

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

### Lesson 25: Progressive encoder unfreeze shows diminishing returns past BN

P.8's 3-stage progressive unfreeze (frozen → layer4 → layer3+layer4) produced its best results during Stage 0 (frozen encoder, BN only). Unfreezing layer4 in Stage 1 actually degraded performance — the model never recovered. **For datasets of ~12K images, frozen encoder + BN unfreeze is the optimal strategy. Partial unfreezing requires significantly more data or more aggressive regularization.**

### Lesson 26: Focal Loss does not improve forensic segmentation

P.9 replaced BCE with Focal Loss (alpha=0.25, gamma=2.0). Pixel F1 changed by +0.03pp (noise), but Pixel AUC regressed -0.0205 and Image ROC-AUC dropped -0.0426. **Focal Loss was designed for object detection (rare objects vs background). In forensic segmentation, Dice Loss already handles class imbalance. Adding Focal Loss is treating a disease the patient doesn't have.**

### Lesson 27: Dataset source matters critically — always validate before training

The Sagnik dataset produced 100% test accuracy — a data leak. X range [0.0, 0.76] (vs standard [0.0, 1.0]) indicates mask images were loaded instead of photographs. **Never trust a dataset at face value. Inspect input distributions, visualize samples, and sanity-check results that seem too good to be true.**

### Lesson 28: Early stopping is essential for CNN classification

The paper architecture ran 40 epochs without early stopping: train acc 98.57%, test loss 0.6185 (severely overfit). The deeper CNN with early stopping (patience=5) stopped at epoch 7: test loss 0.2178 (3x better calibrated). **Training without early stopping wastes compute and produces worse models.**

### Lesson 29: Published paper accuracy claims may not be reproducible

Nagm et al. (2024) claims 94.14% test accuracy. Reproduction achieves 90.33% — a 3.81pp gap. Likely causes: JPEG-only filtering (paper uses 9,501 vs reproduction's 12,614 images), undocumented preprocessing steps, or evaluation on validation/train data. **Always reproduce paper results on a standard dataset before building on their claims.**

### Lesson 30: Classification accuracy ≠ localization capability

The deeper standalone CNN achieves the best classification (90.76%, +3.17pp over UNet P.8's 87.59%). But it cannot produce pixel-level masks. For the assignment — which requires localization — this advantage is irrelevant. **A model that answers "is this image tampered?" better than another is useless if the assignment asks "where is the tampering?"**

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
   - Progressive unfreeze (P.8): +0.65pp (diminishing returns)
   → Variable impact, depends on architecture quality

4. LOSS FUNCTION (how errors are measured)
   - Pretrained: Focal+Dice (P.9) → +0.03pp Pixel F1, AUC regressed
   → Negligible or harmful; BCE+Dice remains optimal

5. INFRASTRUCTURE (speed/precision optimizations)
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
6. **Skip loss function experiments unless there's a clear signal.** P.9 showed Focal Loss is neutral/harmful for forensic segmentation — the default BCE+Dice is already well-suited.
7. **Don't unfreeze encoder layers on small datasets.** P.8 confirmed that 12K images is insufficient for meaningful encoder fine-tuning beyond BN adaptation.
8. **Always validate dataset integrity before training.** The Sagnik data leak would have been caught by a simple input visualization step.

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

---

## 11. Lessons from P.12 and P.14 Audit (2026-03-15)

### Lesson 31: TTA can HURT binary segmentation at fixed thresholds

P.14 applied 4-view TTA (orig, hflip, vflip, hvflip) and averaged probability maps. Result: Pixel F1 dropped 5.32pp (0.6919 -> 0.6388). **Root cause:** Averaging pushes borderline pixels (probabilities near 0.5) below the threshold. Recall dropped 7.34pp while precision gained only 3.22pp. However, Pixel AUC *improved* by +0.9pp, meaning TTA does produce better-calibrated probabilities -- the problem is the fixed 0.5 threshold. **Lesson: TTA requires threshold recalibration to be beneficial for segmentation.**

### Lesson 32: Augmentation is not free -- it introduces training instability

P.12 added Albumentations (HFlip, VFlip, Rotate90, ShiftScaleRotate, GaussianBlur, BrightnessContrast) applied jointly to ELA image and mask. Despite well-chosen transforms, val loss spiked at epochs 19 and 21 before stabilizing. The final Pixel F1 gain was marginal (+0.48pp). **Lesson: Augmentation on ELA images may corrupt the error-level signal (brightness/contrast changes alter the numerical relationship that ELA measures). Use only geometric transforms for ELA augmentation.**

### Lesson 33: Confounding variables invalidate ablation conclusions

P.12 changed TWO variables simultaneously: (1) added augmentation AND (2) switched from BCE+Dice to Focal+Dice loss. Since P.9 showed Focal+Dice was neutral/harmful, the +0.48pp gain is likely from augmentation alone -- but we *cannot* be certain. **Lesson: Single-variable discipline is not optional. Every confounded experiment produces ambiguous results.**

### Lesson 34: Code bugs have cascading consequences

P.14's cell 18 referenced `test_probs` instead of `preds_tta` (copy-paste from P.3). This single NameError crashed cells 18-27, losing: image-level metrics, confusion matrix, all visualizations, sample predictions, and model save. 40% of the notebook's output was destroyed. **Lesson: Always smoke-test variable names after refactoring. A 30-second manual review of cell dependencies catches these instantly.**

### Lesson 35: Perfect reproducibility is achievable with SEED + deterministic ops

P.10 Run-02 produced metrics **identical** to Run-01 across all reported values: same epoch-by-epoch loss curves, same LR schedule, same final Pixel F1 (0.7277). Both ran on P100 GPUs with SEED=42. **Lesson: CUDA determinism (`torch.backends.cudnn.deterministic=True`) + fixed seeds produces bit-identical results on same hardware. Always include reproducibility runs for your best model.**

### Lesson 36: Never trust a dataset that gives >99% accuracy

The ELA-CNN-Forgery-sagnik run achieved 99.95% accuracy on a supposedly standard CASIA-like dataset. Investigation revealed X range [0.0, 0.76] instead of expected [0.0, 1.0], suggesting mask images were loaded as input. **Lesson: Before ANY training, always: (1) visualize random samples, (2) check input statistics, (3) verify image dimensions match expectations. Data leaks are the #1 source of invalid ML results.**

### Lesson 37: Data augmentation benefits image-level more than pixel-level metrics

P.12's augmentation improved Image Accuracy by +1.69pp (86.79% -> 88.48%) and Image F1 by +1.06pp, but Pixel F1 gained only +0.48pp. The augmentation helped the model recognize more tampered/authentic *images* but barely improved pixel-level localization precision. **Lesson: Augmentation regularizes the encoder's image-level features more than the decoder's pixel-level reconstruction.**

---

## 12. Lessons from P.15 Multi-Quality ELA Audit (2026-03-15)

### Lesson 38: Channel independence matters more than channel semantics

P.15 replaced 3 correlated RGB ELA channels (inter-channel correlation ~0.9) with 3 independent quality-level channels (Q=75/85/95, mean range 0.040--0.068). Result: +4.09pp Pixel F1. **Lesson: When constructing multi-channel inputs for CNNs, channels with independent information are more valuable than channels that share the same underlying signal (like R/G/B from the same ELA computation). Design inputs to maximize inter-channel diversity.**

### Lesson 39: Multi-quality ELA captures forensic information at different scales

The three quality levels provide complementary forensic perspectives: Q=75 (mean=0.068) detects strong manipulations with large residuals, Q=95 (mean=0.040) reveals subtle artifacts invisible at lower quality settings. P.15's +6.16pp recall gain over P.3 suggests that multi-quality coverage helps the model detect manipulations that single-quality ELA misses entirely. **Lesson: No single ELA quality is optimal for all forgery types. Stacking multiple qualities gives the model a "multi-resolution" view of compression artifacts.**

### Lesson 40: Input representation remains the highest-impact lever

P.15 (+4.09pp from input change alone) outperformed CBAM attention (P.10, +3.57pp architectural change) and extended training (P.7, +2.34pp). Both P.3 (+23.74pp, RGB→ELA) and P.15 (+4.09pp, single-Q→multi-Q) confirm this pattern. **Lesson: Before investing in architecture changes, loss functions, or training tricks, exhaust input representation experiments first. The signal you feed the model matters 10× more than how the model processes it.**

### Lesson 41: Models that hit the epoch cap need extended training runs

P.15 reached epoch 25/25 with best epoch at 24 and LR never decaying. P.7 previously showed +2.34pp from simply extending P.3 to 50 epochs. P.15 is likely leaving performance on the table. **Lesson: If a model's best epoch is within 2 of the epoch cap AND the LR scheduler never triggered, the model is undertrained. Always follow up with an extended training experiment.**

### Lesson 42: Recall-driven improvements are more valuable than precision-driven ones for forgery detection

P.15 gained +6.16pp recall but only +0.30pp precision over P.3, resulting in +4.09pp F1. In forensic applications, missing a real forgery (FN) is worse than flagging a clean image (FP). P.15's FN rate (24.7%) is competitive, while its FP rate (4.1%) remains excellent. **Lesson: For tampered image detection, optimizing recall is the higher-value objective. A model that finds more forgeries with slightly less precision is preferable to one that is very precise but misses many manipulations.**

---

## 13. Lessons from P.14b and P.18 Audit (2026-03-15)

### Lesson 43: TTA improves calibration but hurts binary decisions at fixed thresholds

P.14b's complete evaluation reveals the TTA paradox: Pixel AUC improved +0.90pp (better probability ranking) but Pixel F1 dropped -5.32pp (worse binary masks). The FP rate fell to 1.2% (best in series) while the FN rate rose to 29.3%. **Lesson: TTA is a calibration tool, not a free accuracy boost. For segmentation tasks with fixed thresholds, TTA requires threshold recalibration (e.g., lowering from 0.5 to ~0.35-0.40) to offset the probability compression from averaging. Never apply TTA without re-tuning the decision boundary.**

### Lesson 44: Complete re-runs are worth the compute cost

P.14 Run-01 lost 40% of its output (image-level metrics, CM, visualizations, model save) due to a single `NameError`. P.14b re-run cost ~30 minutes of GPU time but recovered: Image Acc=87.43%, Macro F1=0.8619, ROC-AUC=0.9610, and the best FP rate in the series (1.2%). **Lesson: When a run has a code crash that loses critical metrics, always re-run rather than trying to extrapolate the missing data. The cost of a re-run is tiny compared to operating with incomplete information.**

### Lesson 45: Always verify checkpoint availability before running evaluation notebooks

P.18's entire run was wasted because the P.3 checkpoint wasn't uploaded to Kaggle. The notebook searched 3 locations, found nothing, and silently fell back to ImageNet weights. All 5 compression conditions produced identical garbage results (Pixel F1=0.036, 100% FPR). **Lesson: Evaluation-only notebooks that depend on external checkpoints must include a hard assertion at the top: `assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"`. Never allow silent fallback to unrelated weights.**

### Lesson 46: Untrained model outputs have a distinctive signature

P.18's invalid results were immediately identifiable: Image Acc = 40.62% = exact tampered class proportion (769/1893), Pixel AUC ≈ 0.50, identical confusion matrices across all conditions (TN=0, FP=1124). These are the hallmarks of a model that predicts the same thing for every input. **Lesson: Build a "sanity check" function that flags: (1) accuracy ≈ class proportion, (2) AUC ≈ 0.50, (3) identical predictions across conditions. Any of these signals should trigger an automatic INVALID verdict.**

### Lesson 47: Well-designed frameworks survive implementation failures

P.18's framework (5 JPEG compression conditions, per-condition metrics, degradation curves, same-seed determinism) is genuinely well-designed for robustness characterization. The only failure was operational (missing checkpoint), not methodological. **Lesson: Separate framework design quality from execution quality. A sound experimental framework that fails due to a missing file can be trivially fixed and re-run. A flawed experimental design cannot be salvaged regardless of execution quality.**


---

## 14. Lessons from P.16/P.17 DCT Experiments and P.30.x Design (2026-03-16)

### Lesson 48: DCT-only input fails catastrophically for forensic segmentation

P.16 fed DCT spatial feature maps (3 channels from JPEG block coefficients) to the same UNet+ResNet-34 pipeline that achieves F1=0.69+ with ELA. Result: F1=0.3209 -- worse than even the RGB baseline (P.1: 0.4546). **Lesson: Frequency-domain features alone cannot substitute for pixel-level error analysis. The frozen encoder's BN statistics are calibrated for ELA; feeding it a fundamentally different distribution produces garbage features. Always verify that your input representation is compatible with your encoder's learned statistics.**

### Lesson 49: DCT as a complement to ELA succeeds -- orthogonal information adds value

P.17 combined ELA (3ch) + DCT (3ch) into a 6-channel input. Result: F1=0.7302 (+3.82pp from P.3), making it the #2 experiment in the series. **Lesson: A feature that fails catastrophically alone can still be valuable as a complement. DCT provides block-level frequency information that ELA's pixel-level error analysis cannot capture. When two signals are orthogonal, fusion often outperforms either alone. The key is ensuring the primary signal (ELA) is present -- DCT cannot stand on its own.**

### Lesson 50: Models still improving at the epoch cap are undertrained

P.17 achieved best epoch at 24/25, with no LR decay triggered -- exactly the same pattern as P.15 (best at 24/25) and P.3 (best at 25/25). All three models were likely still learning when training stopped. P.7 previously showed +2.34pp from simply extending P.3 to 50 epochs. **Lesson: When best_epoch >= max_epochs - 2 AND the LR scheduler never triggered, the model is undertrained. Always follow up with an extended training run. The P.30.1 experiment is specifically designed to test this for the Multi-Q ELA + CBAM combination.**

### Lesson 51: Untested component combinations represent the largest remaining opportunity

After 23 completed experiments, the two best independent gains are Multi-Q ELA (P.15, +4.09pp) and CBAM attention (P.10, +3.54pp isolated). These have never been combined. Since they operate on different pipeline stages (input vs decoder), they should be additive. **Lesson: In ablation studies, the combinatorial space of tested components grows factorially. After establishing individual component effects, the highest-value experiments are testing combinations of top performers. The P.30.x series tests this systematically.**
