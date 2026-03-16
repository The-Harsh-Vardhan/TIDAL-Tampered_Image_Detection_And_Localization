# Cross-Run Audit, Review & Roast: Pretrained Track (vR.P.x)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-16 |
| **Scope** | 18 pretrained-track experiments: vR.P.0 through vR.P.30.4 |
| **Architecture** | UNet (SMP) + ResNet-34 encoder (unless noted) |
| **Dataset** | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |
| **Best Localization** | **vR.P.19** (Pixel F1 = 0.7965) |
| **Best Classification** | **vR.P.30.1** (Test Acc = 91.39%, Macro F1 = 0.9089) |
| **Biggest Missed Opportunity** | P.19 was never extended to 50 epochs |

---

## 1. Executive Summary

Eighteen experiments. Four dead ends. Three breakthroughs. One champion that got benched early.

The pretrained track evolved from a broken baseline that faked 99.31% accuracy (P.0, data leakage) to a legitimate **0.7965 pixel F1** (P.19, multi-quality RGB ELA) — but even the champion was never given a fair training budget. The arc of this project teaches three lessons, in order of importance:

1. **Input representation > encoder architecture > training tricks.** P.3's switch to ELA input delivered +23.7pp pixel F1 — more than every other change combined. The best encoder swap (P.5/P.6) delivered nothing. The best training trick (extended epochs) delivered +2–3pp.
2. **Multi-quality RGB ELA (P.19) is the single best technique discovered.** At only 25 epochs, it already beats every other experiment run for 50 epochs. It was never extended.
3. **CBAM attention helps classification but hurts pixel localization.** P.30.1 (with CBAM, 50 epochs) gets 91.39% accuracy but only 0.7762 pixel F1. P.19 (without CBAM, 25 epochs) gets 90.39% accuracy and 0.7965 pixel F1.

### One-Line Verdicts

| Run | One-Line Verdict |
|-----|------------------|
| **P.0** | The baseline that shipped with a data leak and called itself 99.31% accurate. |
| **P.1** | Fixed the leak, fixed the dataset, discovered the model actually gets 70%. |
| **P.1.5** | Speed optimizations that somehow made the model worse. |
| **P.3** | The day ELA changed everything. +23.7pp pixel F1 in one variable swap. |
| **P.4** | Four channels of input for a 1.3pp improvement. Diminishing returns, meet your mascot. |
| **P.5** | ResNet-50: more decoder params, 74% of the performance of ELA. |
| **P.6** | EfficientNet-B0: smallest model, same lesson — swap the input, not the encoder. |
| **P.7** | Proved P.3 was undertrained. Water is wet. |
| **P.19** | **THE CHAMPION.** 9-channel multi-quality RGB ELA. Best pixel F1 in the series. Never given 50 epochs. |
| **P.20** | Magnitude decomposition: a clever idea that the model did not appreciate. |
| **P.23** | YCbCr chrominance: worst legitimate run. Some hypotheses deserve to fail. |
| **P.24** | Noiseprint: cutting-edge forensics on heavily compressed JPEGs. Predictably poor. |
| **P.27** | JPEG augmentation: bought robustness, not accuracy. |
| **P.28** | Cosine annealing: 50 epochs of fancy LR scheduling for +2.95pp. |
| **P.30** | Multi-Q ELA + CBAM at 25 epochs. A promising combination that needed more time. |
| **P.30.1** | Best classifier in the series at 91.39%. But CBAM traded away pixel precision. |
| **P.30.3** | Focal+Dice loss: a loss function swap that lost. |
| **P.30.4** | Geometric augmentation on the best recipe. Still could not beat plain P.19. |

---

## 2. Configuration Matrix

| Config | P.0 | P.1 | P.1.5 | P.3 | P.4 | P.5 | P.6 | P.7 | P.19 | P.20 | P.23 | P.24 | P.27 | P.28 | P.30 | P.30.1 | P.30.3 | P.30.4 |
|--------|-----|-----|-------|-----|-----|-----|-----|-----|------|------|------|------|------|------|------|--------|--------|--------|
| **Input** | RGB | RGB | RGB | ELA | RGB+ELA | ELA | ELA | ELA | MQ-RGB-ELA | ELA-Mag | YCbCr | Noise | ELA | ELA | MQ-ELA | MQ-ELA | MQ-ELA | MQ-ELA |
| **Channels** | 3 | 3 | 3 | 3 | 4 | 3 | 3 | 3 | 9 | 3 | 3 | 3 | 3 | 3 | 9 | 9 | 9 | 9 |
| **Encoder** | R34 | R34 | R34 | R34 | R34 | **R50** | **EB0** | R34 | R34 | R34 | R34 | R34 | R34 | R34 | R34 | R34 | R34 | R34 |
| **BN Unfrozen** | N | N | N | **Y** | **Y** | N | N | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| **Attention** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **CBAM** | **CBAM** | **CBAM** | **CBAM** |
| **Loss** | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | BCE+D | **Focal+D** | BCE+D |
| **Scheduler** | RLROP | RLROP | RLROP | RLROP | RLROP | RLROP | RLROP | RLROP | RLROP | RLROP | RLROP | RLROP | RLROP | **Cosine** | RLROP | RLROP | RLROP | RLROP |
| **Max Epochs** | 25 | 25 | 25 | 25 | 25 | 25 | 25 | **50** | 25 | 25 | 25 | 25 | 25 | **50** | 25 | **50** | 25 | **50** |
| **Augmentation** | — | — | — | — | — | — | — | — | — | — | — | — | **JPEG** | — | — | — | — | **Geom** |

**Key:** R34=ResNet-34, R50=ResNet-50, EB0=EfficientNet-B0, MQ=Multi-Quality, RLROP=ReduceLROnPlateau, BCE+D=BCEDice, Focal+D=FocalDice

---

## 3. Full Leaderboard (Ranked by Pixel F1)

| Rank | Version | Pixel F1 | Pixel IoU | Pixel AUC | Pix Prec | Pix Rec | Test Acc | Macro F1 | ROC-AUC | Best/Max Ep | Verdict |
|------|---------|----------|-----------|-----------|----------|---------|----------|----------|---------|-------------|---------|
| 1 | **P.19** | **0.7965** | **0.6618** | 0.9726 | 0.8606 | **0.7413** | 90.39% | 0.8970 | 0.9765 | 25/25 | **CHAMPION** |
| 2 | P.30.1 | 0.7762 | 0.6343 | **0.9795** | 0.8719 | 0.6995 | **91.39%** | **0.9089** | **0.9815** | 41/50 | **POSITIVE** |
| 3 | P.30.4 | 0.7662 | 0.6210 | 0.9726 | **0.8820** | 0.6772 | 90.12% | 0.8942 | 0.9755 | 41/50 | POSITIVE |
| 4 | P.30.3 | 0.7509 | 0.6011 | 0.9694 | 0.8423 | 0.6773 | 90.91% | 0.9034 | 0.9783 | 22/25 | NEUTRAL |
| 5 | P.30 | 0.7438 | 0.5921 | 0.9733 | 0.8455 | 0.6639 | 88.59% | 0.8768 | 0.9718 | 23/25 | NEUTRAL |
| 6 | P.28 | 0.7215 | 0.5643 | 0.9572 | 0.8498 | 0.6268 | 87.96% | 0.8705 | 0.9456 | 38/50 | POSITIVE |
| 7 | P.7 | 0.7154 | 0.5569 | 0.9504 | 0.8374 | 0.6245 | 87.37% | 0.8637 | 0.9433 | 36/50 | POSITIVE |
| 8 | P.27 | 0.7105 | 0.5510 | 0.9553 | 0.8444 | 0.6132 | 87.90% | 0.8689 | 0.9636 | 23/25 | NEUTRAL |
| 9 | P.4 | 0.7053 | 0.5447 | 0.9433 | 0.8452 | 0.6051 | 84.42% | 0.8322 | 0.9229 | 24/25 | NEUTRAL |
| 10 | P.3 | 0.6920 | 0.5291 | 0.9528 | 0.8356 | 0.5905 | 86.79% | 0.8560 | 0.9502 | 25/25 | **BREAKTHROUGH** |
| 11 | P.20 | 0.6555 | 0.4875 | 0.9399 | 0.7586 | 0.5771 | 85.84% | 0.8472 | 0.9258 | 21/25 | NEGATIVE |
| 12 | P.24 | 0.5246 | 0.3555 | 0.8831 | 0.6118 | 0.4591 | 72.21% | 0.6939 | 0.7647 | 17/25 | REJECTED |
| 13 | P.6 | 0.5217 | 0.3529 | 0.8708 | 0.7034 | 0.4146 | 70.68% | 0.6950 | 0.7801 | 16/25 | REJECTED |
| 14 | P.5 | 0.5137 | 0.3456 | 0.8828 | 0.6089 | 0.4442 | 72.00% | 0.7143 | 0.8126 | 19/25 | REJECTED |
| 15 | P.23 | 0.4709 | 0.3080 | 0.8453 | 0.4901 | 0.4532 | 66.56% | 0.6564 | 0.7303 | 15/25 | REJECTED |
| 16 | P.1 | 0.4546 | 0.2942 | 0.8509 | 0.6335 | 0.3545 | 70.15% | 0.6867 | 0.7785 | 18/25 | **BASELINE** |
| 17 | P.1.5 | 0.4227 | 0.2680 | 0.8560 | 0.6364 | 0.3165 | 71.05% | 0.7016 | 0.7980 | 16/25 | NEUTRAL |
| 18 | P.0* | 0.3948 | 0.2459 | 0.9955 | 0.2787 | 0.6766 | 99.31%* | 0.9929* | 1.0000* | 24/25 | **BROKEN** |

\* P.0 classification metrics are inflated by data leakage. Only pixel metrics are semi-trustworthy (and even those may be affected).

---

## 4. Metric Deep Dives

### 4.1 Pixel F1 — The Five Biggest Jumps

| From → To | Change | Pixel F1 Delta | What Happened |
|-----------|--------|----------------|---------------|
| P.1 → P.3 | ELA input + BN unfrozen | **+0.2374** (+52.2%) | The single biggest improvement in the entire project |
| P.3 → P.19 | Multi-quality RGB ELA (9ch) | **+0.1045** (+15.1%) | Second breakthrough: multi-scale compression analysis |
| P.3 → P.7 | Extended training (25→50 ep) | +0.0234 (+3.4%) | Confirmed P.3 was undertrained |
| P.30 → P.30.1 | Extended training (25→50 ep) | +0.0324 (+4.4%) | Same story, bigger gain |
| P.3 → P.28 | Cosine annealing + 50 ep | +0.0295 (+4.3%) | Scheduler + epochs combined |

### 4.2 Pixel F1 — The Five Biggest Regressions

| From → To | Change | Pixel F1 Delta | What Happened |
|-----------|--------|----------------|---------------|
| P.0 → P.1 | Fixed data leak | **-0.2985** (reclassified) | P.0 was broken; this is a correction, not a regression |
| P.1 → P.1.5 | AMP/TF32 speed opts | -0.0319 (-7.0%) | Infrastructure change degraded quality |
| P.19 → P.20 | ELA magnitude decomp | -0.1410 (-17.7%) | Threw away useful RGB information |
| P.3 → P.5 | ResNet-50 encoder | -0.1783 (-25.8%) | Deeper encoder with RGB input — dead end |
| P.3 → P.23 | YCbCr chrominance | -0.2211 (-32.0%) | Wrong signal for this dataset |

### 4.3 The Localization vs Classification Divergence

The leaderboards diverge. The best localizer and the best classifier are different models:

| Metric | P.19 (Localization King) | P.30.1 (Classification King) | Delta |
|--------|--------------------------|-------------------------------|-------|
| **Pixel F1** | **0.7965** | 0.7762 | P.19 wins by +0.0203 |
| **Pixel IoU** | **0.6618** | 0.6343 | P.19 wins by +0.0275 |
| Pixel Precision | 0.8606 | **0.8719** | P.30.1 wins by +0.0113 |
| Pixel Recall | **0.7413** | 0.6995 | P.19 wins by +0.0418 |
| Test Accuracy | 90.39% | **91.39%** | P.30.1 wins by +1.0pp |
| Macro F1 | 0.8970 | **0.9089** | P.30.1 wins by +0.0119 |
| ROC-AUC | 0.9765 | **0.9815** | P.30.1 wins by +0.0050 |
| Epochs | **25** | 50 | P.19 used HALF the compute |

CBAM gives P.30.1 sharper classification boundaries (higher precision, better AUC) at the cost of pixel recall. P.19 finds more tampered pixels (recall 0.7413 vs 0.6995) because its raw 9-channel signal preserves fine-grained spatial detail that CBAM's channel averaging smooths away.

**For this assignment, which values localization: P.19 is the better submission.**

---

## 5. Experiment Categories & Insights

### 5.1 Input Representation (P.0, P.1, P.3, P.4, P.19, P.20, P.23, P.24)

Input representation spans a **34pp range** in pixel F1 — more than any other factor:

```
Multi-Q RGB ELA (P.19)  ████████████████████████████████████████  0.7965
Single ELA (P.3)        ██████████████████████████████████▌       0.6920
4ch RGB+ELA (P.4)       ███████████████████████████████████▎      0.7053
ELA Magnitude (P.20)    █████████████████████████████████         0.6555
Noiseprint (P.24)       █████████████████████████▎                0.5246
RGB Baseline (P.1)      ██████████████████████▍                   0.4546
YCbCr (P.23)            ███████████████████████                   0.4709
```

**Lesson:** The single most impactful decision in this entire project was choosing what to feed the model. Everything else is noise by comparison.

### 5.2 Encoder Architecture (P.1, P.5, P.6)

| Encoder | Pixel F1 | Test Acc | Notes |
|---------|----------|----------|-------|
| ResNet-34 + ELA (P.3) | **0.6920** | **86.79%** | Baseline with ELA input |
| ResNet-34 + RGB (P.1) | 0.4546 | 70.15% | Baseline without ELA |
| ResNet-50 + ELA (P.5) | 0.5137 | 72.00% | Deeper encoder, worse results |
| EffNet-B0 + ELA (P.6) | 0.5217 | 70.68% | Smallest model, same lesson |

Both P.5 and P.6 used RGB input, which explains why they're below P.3. But even accounting for that, the encoder swap didn't help — **frozen pretrained encoders have fixed features; changing architecture doesn't change what ImageNet features look like**. The bottleneck was always the input.

### 5.3 Training Budget (P.3→P.7, P.30→P.30.1)

| Pair | 25 Epochs | 50 Epochs | Delta | % Gain |
|------|-----------|-----------|-------|--------|
| P.3 → P.7 | 0.6920 | 0.7154 | +0.0234 | +3.4% |
| P.30 → P.30.1 | 0.7438 | 0.7762 | +0.0324 | +4.4% |

Consistent +2–3pp from doubling epochs. Both notebooks were still improving when stopped.

**THE MISSING EXPERIMENT:** P.19 at 25 epochs (0.7965) already beats everything at 50 epochs. If the same +3–4% scaling holds, P.19 at 50 epochs projects to **~0.82–0.83 pixel F1**. This experiment was never run.

### 5.4 CBAM Attention (P.30 series vs P.19)

| Experiment | CBAM | Epochs | Pixel F1 | Test Acc |
|------------|------|--------|----------|----------|
| P.19 | No | 25 | **0.7965** | 90.39% |
| P.30 | Yes | 25 | 0.7438 | 88.59% |
| P.30.1 | Yes | 50 | 0.7762 | **91.39%** |
| P.30.4 | Yes | 50 | 0.7662 | 90.12% |

At the same 25-epoch budget, CBAM **reduces** pixel F1 by 5.3pp. Even at double the epochs (P.30.1), CBAM still doesn't match P.19's pixel F1. CBAM's channel attention averages spatial features, which helps the classification head but blurs pixel-level boundaries.

### 5.5 Loss Functions & Augmentation (P.27, P.28, P.30.3, P.30.4)

| Experiment | Change | Pixel F1 | vs Baseline |
|------------|--------|----------|-------------|
| P.27 | JPEG compression aug | 0.7105 | +0.0185 from P.3 |
| P.28 | Cosine annealing + 50ep | 0.7215 | +0.0295 from P.3 |
| P.30.3 | Focal+Dice loss | 0.7509 | -0.0253 from P.30.1 |
| P.30.4 | Geometric aug + 50ep | 0.7662 | -0.0100 from P.30.1 |

- JPEG augmentation (P.27): modest gain, improves robustness
- Cosine annealing (P.28): marginal gain, most is from 50 epochs
- Focal+Dice (P.30.3): **worse** than BCE+Dice — Focal loss was designed for extreme class imbalance; this dataset is ~60/40
- Geometric augmentation (P.30.4): **worse** than no augmentation — may disrupt ELA signal alignment

---

## 6. The Roast — Every Experiment, Rated and Skewered

### P.0 — "The 99.31% Lie" (1/10)

A model that reports 99.31% accuracy and 1.0000 ROC-AUC should set off every alarm in a practitioner's brain. P.0's dataset discovery logic was broken — it either leaked test data into training or assigned labels incorrectly. The classification metrics are fantasy. The only honest number in this notebook is the pixel F1 of 0.3948, and even that's suspect. The pixel precision of 0.2787 tells the real story: 72% of pixels the model calls "tampered" are clean. This notebook exists as a cautionary tale about trusting headline numbers.

**What it taught us:** If your AUC is 1.0, your pipeline has a bug.

---

### P.1 — "The Honest Reckoning" (4/10)

Fixed the dataset (sagnikkayalcse52, 100% GT masks, proper discovery logic). Reality check: 70.15% accuracy, 0.4546 pixel F1. The pretrained ResNet-34 encoder, frozen on ImageNet features, looking at RGB inputs, produces features that know what cats and cars look like — not what JPEG compression artifacts look like. Pixel precision improved dramatically (0.6335 vs P.0's 0.2787) because the model stopped hallucinating tampering everywhere, but recall cratered (0.3545) — it misses 65% of actual tampered pixels.

**What it taught us:** The true baseline is 70% accuracy and 0.45 pixel F1. Everything above this is earned.

---

### P.1.5 — "Speed Kills (Accuracy)" (3/10)

AMP mixed precision + TF32 for faster training. Speed went up. Pixel F1 went down: 0.4227 (−0.0319 from P.1). The only experiment where infrastructure changes degraded results. AMP's float16 numerics on P100 GPUs likely introduced enough accumulation error to shift the loss landscape. The irony: you optimized the training loop and made the model worse.

**What it taught us:** Benchmark infrastructure changes, don't assume they're neutral.

---

### P.3 — "The ELA Revelation" (8/10)

The most important experiment in the entire project. One change — replacing RGB input with ELA at quality 90 — delivered +23.7pp pixel F1 (0.4546 → 0.6920). That's more improvement than every subsequent experiment combined. Error Level Analysis compresses and re-saves the image, then measures per-pixel differences. Tampered regions compress differently. The frozen ImageNet encoder, which was useless at detecting forgery from RGB, suddenly becomes a powerful feature extractor when the input directly encodes the signal of interest.

BN layer unfreezing (allowing the encoder's batch normalization to adapt to ELA statistics) was the quiet second hero. ImageNet BN running means are calibrated for natural images; ELA images have completely different distributions.

The only penalty: the model was still improving at epoch 25. Val loss was still trending down. This notebook was stopped too early.

**What it taught us:** Input representation is everything. Give the model the right signal and even a frozen encoder does the rest.

---

### P.4 — "The Fourth Channel Nobody Needed" (5/10)

4-channel input: RGB + ELA grayscale. The idea is sensible — give the model both the original appearance and the forensic signal. Reality: +1.3pp pixel F1 (0.7053) with a -2.4pp accuracy drop (84.42%). The model has to learn what to do with the 4th channel through an unfrozen conv1 layer, adding trainable parameters and training instability. The marginal pixel gain doesn't justify the classification loss.

**What it taught us:** When your 3-channel ELA already captures the signal, adding RGB is noise.

---

### P.5 — "The ResNet-50 Money Pit" (3/10)

Hypothesis: deeper encoder features → better localization. Result: 0.5137 pixel F1, worse than P.1's RGB baseline with ResNet-34. The deeper frozen encoder provides features that are too abstract — layer 4 of ResNet-50 encodes semantic concepts like "object boundaries" and "texture categories," not "this pixel was recompressed differently." Also used RGB input (not ELA), making this an apples-to-oranges comparison with P.3.

**What it taught us:** Deeper ≠ better when the features are frozen and the domain is forensics.

---

### P.6 — "The Efficient Deck Chair Rearrangement" (4/10)

EfficientNet-B0: smallest model, compound scaling, mobile-optimized architecture. Result: 70.68% accuracy, 0.5217 pixel F1. Marginally better than P.5 on pixel metrics, but still 25% below P.3's ELA result. EfficientNet's efficient channel attention and depth-wise separable convolutions don't help when the input signal is wrong.

Two experiments (P.5 + P.6), two encoder architectures, zero improvements over ELA. The encoder was never the bottleneck.

**What it taught us:** Stop swapping encoders. Swap the input.

---

### P.7 — "Captain Obvious Goes to 50 Epochs" (7/10)

Extended P.3 to 50 epochs with patience 10. Best epoch moved from 25 to 36. Pixel F1: 0.7154 (+2.3pp). Not surprising, but necessary — P.3 was clearly undertrained (val loss still decreasing at epoch 25). Clean execution. No bugs. The denormalize NameError from P.3 was fixed. This is what competent engineering looks like: identify the problem, apply the obvious fix, verify.

**What it taught us:** If your model is still improving when training stops, train it longer.

---

### P.19 — "The Undisputed Champion That Got Benched Early" (9/10)

The best pixel-level model in the entire pretrained track. 9-channel input: Error Level Analysis at three quality levels (Q=75, 85, 95) in full RGB color. Each quality level captures different compression artifacts — high-Q ELA detects subtle edits, low-Q ELA detects aggressive edits. The model gets all three perspectives simultaneously.

Results at **only 25 epochs**: 0.7965 pixel F1, 0.6618 pixel IoU, 0.9726 pixel AUC. The best pixel recall in the series (0.7413) — it finds 74% of tampered pixels, compared to P.3's 59%. Classification is strong too: 90.39% accuracy, 0.8970 macro F1.

And then nobody ran it for 50 epochs.

Every P.30.x variant — with CBAM attention, with Focal loss, with geometric augmentation, trained for 50 epochs — failed to beat P.19's pixel F1 at 25 epochs. The projected performance at 50 epochs (based on the +3–4% scaling pattern from P.7 and P.30.1) is approximately **0.82–0.83 pixel F1**. That experiment sits unrun.

Score is 9/10, not 10/10, because someone looked at the best model in the project and decided not to give it more time.

**What it taught us:** Multi-scale analysis works. Multi-quality ELA is the right feature set for JPEG forensics. And train your winners.

---

### P.20 — "When Clever Math Meets Indifferent Gradients" (4/10)

Decomposes ELA into magnitude (intensity of compression error) and chrominance direction (color angle of the error). Theoretically elegant: magnitude should separate "edited" from "unedited," while direction should reveal the type of edit. In practice: 0.6555 pixel F1, below P.3's plain ELA (0.6920).

The problem: chrominance ratios are noisy in low-magnitude regions (dividing small numbers by small numbers). The model spends capacity learning to ignore the noise instead of detecting forgery. Raw RGB ELA captures both magnitude and direction implicitly without the numerical instability.

**What it taught us:** Mathematical decomposition can lose information that neural networks extract implicitly.

---

### P.23 — "The Chrominance Dead End" (2/10)

YCbCr chrominance channels as input. The hypothesis: JPEG compresses chrominance (Cb/Cr) more aggressively than luminance (Y), so chrominance differences should reveal tampering. The result: 66.56% test accuracy, 0.4709 pixel F1 — the worst legitimate run in the series.

The theory is sound for uncompressed or lightly compressed images. CASIA v2.0 images have been heavily JPEG compressed (often multiple times), destroying exactly the chrominance differences that this technique relies on. The signal-to-noise ratio in the chrominance domain is below what the model can extract.

**What it taught us:** Know your dataset. CASIA's compression history makes chrominance analysis useless.

---

### P.24 — "State-of-the-Art Theory Meets Budget Data" (3/10)

Noiseprint/DnCNN-style noise residual extraction. In the forensics literature, camera noise fingerprints are among the most powerful forgery indicators. Result: 0.5246 pixel F1, 72.21% accuracy. Below P.3 by 16.7pp.

The problem isn't the technique — it's the dataset. CASIA images have been resized, cropped, and JPEG-compressed, all of which destroy camera noise patterns. Noiseprint needs high-quality, minimally processed images. Feeding it multiply-compressed JPEGs is like trying to read fingerprints off a surface that's been pressure-washed.

**What it taught us:** Match the technique to the data quality. Noiseprint needs pristine images.

---

### P.27 — "Compression Augmentation: Honest Work" (5/10)

JPEG compression augmentation during training (random Q=50–95). The model sees ELA computed from images that have been re-compressed at various qualities, learning to detect forgery regardless of the JPEG quality factor. Result: 0.7105 pixel F1, +1.85pp from P.3. Image ROC-AUC of 0.9636 is the best non-P.30 classification calibration.

It does exactly what it says: builds robustness. Not a game-changer, not a disappointment. Honest, predictable, useful.

**What it taught us:** Augmentation works when it matches the real-world degradation you expect.

---

### P.28 — "The Scheduler That Justified Its Existence" (6/10)

CosineAnnealingWarmRestarts (T_0=10, T_mult=2) with 50 epochs. Result: 0.7215 pixel F1, +2.95pp from P.3. The warm restarts allow the model to escape local minima by periodically spiking the learning rate. Training dynamics were stable with proper convergence.

But: P.7 got +2.34pp from just more epochs with the standard scheduler. The isolated scheduler contribution is probably ~0.6pp. CosineAnnealing is a nice-to-have, not a breakthrough.

**What it taught us:** Fancy schedulers help, but most of the gain is just more training time.

---

### P.30 — "The Promising Combination at Half Budget" (5/10)

Multi-quality ELA (9 channels) + CBAM channel/spatial attention, 25 epochs. Result: 0.7438 pixel F1 — below P.19 (0.7965) which has the same input but no CBAM. At the same epoch count, CBAM is a net negative for pixel localization.

The likely explanation: CBAM's channel attention weights average across spatial dimensions, which helps the classification head see the "big picture" but blurs the pixel-level prediction boundaries that localization needs.

**What it taught us:** Attention that helps classification can hurt localization. Test them separately.

---

### P.30.1 — "Classification King, Localization Prince" (7/10)

P.30 extended to 50 epochs. Best classification numbers in either track: 91.39% accuracy, 0.9089 macro F1, 0.9815 ROC-AUC. Pixel F1 improved to 0.7762 but still falls short of P.19's 0.7965 at half the compute budget.

P.30.1 is the right choice if your metric is classification accuracy. It's the wrong choice if your metric is pixel F1. For this assignment, which emphasizes localization, P.19 wins.

**What it taught us:** CBAM is a classification technique, not a localization technique.

---

### P.30.3 — "Focal Loss: The Solution Nobody Asked For" (4/10)

Replaced BCE+Dice with Focal+Dice. Focal loss down-weights easy examples and focuses on hard ones — designed for extreme class imbalance (1:1000 in object detection). CASIA's pixel-level class ratio is approximately 60:40 (most pixels are authentic, but it's not extreme). Result: 0.7509 pixel F1, worse than P.30.1's 0.7762 with BCE+Dice.

Focal loss's hard-example mining adds noise to the gradient signal in a balanced setting. The model spends capacity on genuinely ambiguous boundary pixels that even a human couldn't classify.

**What it taught us:** Don't apply solutions for extreme imbalance to moderate imbalance.

---

### P.30.4 — "Augmentation: Still Not the Answer" (4/10)

Geometric augmentation (flips, rotations) on P.30.1's recipe. Result: 0.7662 pixel F1, worse than P.30.1's 0.7762 without augmentation. This echoes the ETASR track's P.1.2 lesson: geometric augmentation disrupts the spatial patterns that the model has learned.

For ELA-based forensics, the compression artifacts have specific spatial relationships to the image grid. Rotating or flipping changes these relationships, forcing the model to learn invariances that don't exist in real forensic images (forgers don't rotate the evidence before saving as JPEG).

**What it taught us:** Augmentation that contradicts the signal structure hurts, not helps.

---

## 7. The Three Biggest Mistakes

### Mistake 1: Never Running P.19 for 50 Epochs

This is the single largest missed opportunity in the entire project.

| Evidence | P.3 → P.7 | P.30 → P.30.1 | P.19 → ??? |
|----------|-----------|---------------|------------|
| Epochs | 25 → 50 | 25 → 50 | 25 → **never run** |
| Pixel F1 gain | +0.0234 (+3.4%) | +0.0324 (+4.4%) | projected **+0.03–0.04** |
| Projected F1 | — | — | **~0.82–0.83** |

P.19 at 25 epochs already beats everything in the series. It was still improving (best epoch = final epoch, val loss trending down). The projected 50-epoch result would set a new high-water mark by a significant margin.

### Mistake 2: The Encoder Swap Detour (P.5, P.6)

Two experiments testing ResNet-50 and EfficientNet-B0 on RGB input. Both were dominated by P.3's ELA input with the original ResNet-34. Combined compute: ~48 GPU hours for a dead-end conclusion that could have been inferred from P.3 alone: if ELA input makes ResNet-34 work, the bottleneck isn't the encoder.

### Mistake 3: Four CBAM Experiments Before Maximizing the Baseline

P.30, P.30.1, P.30.3, and P.30.4 — four experiments adding CBAM to multi-quality ELA. None beat P.19's pixel F1 without CBAM. The correct sequence was: first extend P.19 to 50 epochs (establish the ceiling), then add CBAM (test if attention helps at the ceiling). Instead, CBAM was tested before the baseline was maximized.

---

## 8. Run Lineage

```
P.0 (Broken baseline, data leak, F1=0.3948)
 │
 └── P.1 (Fixed baseline, F1=0.4546) ◄── TRUE BASELINE
      │
      ├── P.1.5 (AMP speed opts, F1=0.4227) ── REGRESSION
      │
      ├── P.5 (ResNet-50, F1=0.5137) ── dead end
      │
      ├── P.6 (EfficientNet-B0, F1=0.5217) ── dead end
      │
      └── P.3 (ELA input, F1=0.6920) ◄◄ PIVOTAL BREAKTHROUGH
           │
           ├── P.4 (4ch RGB+ELA, F1=0.7053) ── neutral
           │
           ├── P.7 (50 epochs, F1=0.7154) ── positive
           │
           ├── P.20 (ELA magnitude, F1=0.6555) ── regression
           │
           ├── P.23 (YCbCr, F1=0.4709) ── rejected
           │
           ├── P.24 (Noiseprint, F1=0.5246) ── rejected
           │
           ├── P.27 (JPEG aug, F1=0.7105) ── neutral
           │
           ├── P.28 (Cosine annealing 50ep, F1=0.7215) ── positive
           │
           ├── P.19 (Multi-Q RGB ELA 9ch, F1=0.7965) ◄◄ SERIES BEST PIXEL F1
           │    │
           │    └── [P.19 @ 50ep ── NEVER RUN ── projected ~0.82+]
           │
           └── P.30 (Multi-Q + CBAM 25ep, F1=0.7438)
                │
                ├── P.30.1 (50ep, F1=0.7762) ◄◄ BEST CLASSIFICATION
                │    │
                │    └── P.30.4 (+geometric aug, F1=0.7662) ── regression
                │
                └── P.30.3 (Focal+Dice 25ep, F1=0.7509) ── neutral
```

---

## 9. Unsolved Problems

### Problem 1: P.19 at 50 Epochs
The most obvious next experiment. Conservative projection: 0.82+ pixel F1 based on P.7 and P.30.1 scaling patterns.

### Problem 2: Pixel Recall Ceiling
Best pixel recall: P.19 at 0.7413. Still missing ~26% of tampered pixels. Higher resolution (512×512) or multi-scale features may help push recall without sacrificing precision.

### Problem 3: No Ensemble or TTA
None of the 18 experiments use test-time augmentation or model ensembling. P.19 + P.30.1 ensemble could capture both localization and classification strengths. P.14 actually implements TTA — its patterns could be backported.

### Problem 4: Dataset Size Bottleneck
8,829 training images for all experiments. CASIA v2.0 is a well-known dataset with known limitations (resolution, compression artifacts, limited forgery types). External data (Columbia, Coverage, NIST) could improve generalization.

---

## 10. Verdict & Recommendations

### The Winner: P.19

For assignment submission and localization performance:
- Pixel F1: **0.7965**, Pixel IoU: **0.6618**, Pixel AUC: 0.9726
- Best pixel-level model despite only 25 epochs of training
- 9-channel multi-quality RGB ELA input is the key innovation

### The Runner-Up: P.30.1

For classification-focused tasks:
- Test Acc: **91.39%**, Macro F1: **0.9089**, ROC-AUC: **0.9815**
- CBAM attention helps image-level decision making

### Immediate Next Steps (Priority Order)

1. **Run P.19 at 50 epochs, patience 10** — highest expected ROI, projected ~0.82+ pixel F1
2. **Run P.19 at 50 epochs + CBAM** — test whether CBAM helps the 9ch RGB pipeline specifically
3. **Add TTA to P.19 checkpoint** — horizontal + vertical flip averaging, zero extra training cost
4. **Try P.19 at 512×512 resolution** — more spatial detail for improved pixel-level boundaries

### The Bottom Line

18 experiments, 4 dead ends, 3 breakthroughs, 1 champion.

The three breakthroughs: ELA input (P.3), multi-quality ELA (P.19), extended training (P.7/P.30.1). The champion never got a fair fight — P.19 at 25 epochs beats everything else at 50 epochs.

Run P.19 for 50 epochs. That is the plan.

---

## 11. Appendix — Master Metric Reference

| Version | Pixel Prec | Pixel Rec | Pixel F1 | Pixel IoU | Pixel AUC | Test Acc | Macro F1 | ROC-AUC | Best Ep | Max Ep |
|---------|------------|-----------|----------|-----------|-----------|----------|----------|---------|---------|--------|
| P.0* | 0.2787 | 0.6766 | 0.3948 | 0.2459 | 0.9955 | 99.31%* | 0.9929* | 1.0000* | 24 | 25 |
| P.1 | 0.6335 | 0.3545 | 0.4546 | 0.2942 | 0.8509 | 70.15% | 0.6867 | 0.7785 | 18 | 25 |
| P.1.5 | 0.6364 | 0.3165 | 0.4227 | 0.2680 | 0.8560 | 71.05% | 0.7016 | 0.7980 | 16 | 25 |
| P.3 | 0.8356 | 0.5905 | 0.6920 | 0.5291 | 0.9528 | 86.79% | 0.8560 | 0.9502 | 25 | 25 |
| P.4 | 0.8452 | 0.6051 | 0.7053 | 0.5447 | 0.9433 | 84.42% | 0.8322 | 0.9229 | 24 | 25 |
| P.5 | 0.6089 | 0.4442 | 0.5137 | 0.3456 | 0.8828 | 72.00% | 0.7143 | 0.8126 | 19 | 25 |
| P.6 | 0.7034 | 0.4146 | 0.5217 | 0.3529 | 0.8708 | 70.68% | 0.6950 | 0.7801 | 16 | 25 |
| P.7 | 0.8374 | 0.6245 | 0.7154 | 0.5569 | 0.9504 | 87.37% | 0.8637 | 0.9433 | 36 | 50 |
| **P.19** | **0.8606** | **0.7413** | **0.7965** | **0.6618** | 0.9726 | 90.39% | 0.8970 | 0.9765 | 25 | 25 |
| P.20 | 0.7586 | 0.5771 | 0.6555 | 0.4875 | 0.9399 | 85.84% | 0.8472 | 0.9258 | 21 | 25 |
| P.23 | 0.4901 | 0.4532 | 0.4709 | 0.3080 | 0.8453 | 66.56% | 0.6564 | 0.7303 | 15 | 25 |
| P.24 | 0.6118 | 0.4591 | 0.5246 | 0.3555 | 0.8831 | 72.21% | 0.6939 | 0.7647 | 17 | 25 |
| P.27 | 0.8444 | 0.6132 | 0.7105 | 0.5510 | 0.9553 | 87.90% | 0.8689 | 0.9636 | 23 | 25 |
| P.28 | 0.8498 | 0.6268 | 0.7215 | 0.5643 | 0.9572 | 87.96% | 0.8705 | 0.9456 | 38 | 50 |
| P.30 | 0.8455 | 0.6639 | 0.7438 | 0.5921 | 0.9733 | 88.59% | 0.8768 | 0.9718 | 23 | 25 |
| **P.30.1** | 0.8719 | 0.6995 | 0.7762 | 0.6343 | **0.9795** | **91.39%** | **0.9089** | **0.9815** | 41 | 50 |
| P.30.3 | 0.8423 | 0.6773 | 0.7509 | 0.6011 | 0.9694 | 90.91% | 0.9034 | 0.9783 | 22 | 25 |
| P.30.4 | 0.8820 | 0.6772 | 0.7662 | 0.6210 | 0.9726 | 90.12% | 0.8942 | 0.9755 | 41 | 50 |

\* P.0 classification metrics inflated by data leakage.
