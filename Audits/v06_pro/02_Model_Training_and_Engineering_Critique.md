# Model, Training, and Engineering Critique

This note challenges the architecture choice, the loss design, the training stability story, and the engineering maturity claims.

## Finding 1: U-Net plus ResNet34 is a convenience baseline, not a strongly justified forensic architecture

**Claim in docs**

- `Docs6/03_Model_Architecture.md` argues that U-Net is proven for dense prediction and ResNet34 transfer learning compensates for small CASIA size.
- `Docs6/01_System_Architecture.md` calls the design a baseline aligned with assignment constraints.

**Technical objection**

The argument is mostly: it is standard, pretrained, and fits on a T4. That is reasonable for implementation speed, but it does not establish that the model has the right inductive bias for forensic localization.

**Why it matters**

Interviewers do not usually challenge whether a baseline can run. They challenge whether the chosen feature extractor is aligned with the evidence the task requires.

**Stronger answer or remediation**

Position the model explicitly as a starter benchmark and explain what it misses: weak boundary reasoning, limited global context, and no dedicated forensic feature stream.

## Finding 2: The project does not seriously answer "why not DeepLabV3+?"

**Claim in docs**

- `Docs6/03_Model_Architecture.md` lists only small encoder swaps such as EfficientNet-B0 and B1 as future work.

**Technical objection**

DeepLabV3+ is an obvious baseline for irregular masks because atrous spatial context can help with large receptive fields while preserving detail. The docs never explain why U-Net should beat or match it for splicing boundaries.

**Why it matters**

This is a standard senior-interview question because DeepLabV3+ is not a research-frontier straw man. It is a very normal alternative baseline.

**Stronger answer or remediation**

Prepare a concrete tradeoff statement: "We chose U-Net for implementation speed and decoder simplicity, not because we proved it superior. DeepLabV3+ is a valid comparison we would add next."

## Finding 3: The project does not seriously answer "why not transformers?"

**Claim in docs**

- `Docs6/11_Research_Alignment.md` mentions transformer hybrids as future work.

**Technical objection**

Mentioning a stronger family is not the same as justifying its exclusion. Tamper localization often requires long-range context: does this pasted region match the lighting, noise, and semantics of the surrounding scene? That is exactly where transformer-style context can help.

**Why it matters**

A reviewer can reasonably ask whether the chosen model is architecturally outdated for the stated task.

**Stronger answer or remediation**

State that the project traded off representational strength for simplicity and resource limits, then explain how you would benchmark a lightweight SegFormer or hybrid encoder next.

## Finding 4: RGB-only input is a major forensic limitation

**Claim in docs**

- `Docs6/03_Model_Architecture.md` uses `in_channels=3`.
- `Docs6/11_Research_Alignment.md` admits that stronger papers use edge, noise, frequency, or multi-domain fusion.

**Technical objection**

Natural-image RGB features are not the whole story in image forensics. Compression inconsistencies, noise residuals, demosaicing artifacts, and boundary artifacts are often weak or invisible in standard RGB.

**Why it matters**

The chosen model may learn semantic shortcuts rather than forensic evidence. On CASIA, that can still look good if the dataset has strong low-level regularities.

**Stronger answer or remediation**

Add a minimal second stream or channel family such as SRM residuals, ELA, or high-pass residual maps. At minimum, explain that the current model is intentionally blind to important forensic domains.

## Finding 5: BCE plus Dice is reasonable, but the implementation leaves important holes

**Claim in docs**

- `Docs6/04_Training_Strategy.md` and both v6 notebooks use BCE plus Dice as the main loss.

**Technical objection**

Three issues are left unaddressed:

1. `BCEWithLogitsLoss` is used without `pos_weight`, despite repeated claims that tampered pixels are typically under 5 percent.
2. The Dice term is computed across the entire batch, not per image, so large masks dominate smaller ones.
3. The loss has no explicit boundary component, even though tamper localization quality is often decided at edges.

**Why it matters**

The docs claim the loss is chosen for class imbalance and small-region sensitivity, but the actual formulation only partially supports that claim.

**Stronger answer or remediation**

Use per-sample Dice, compare against Focal or Tversky variants, and consider a boundary-aware auxiliary loss if precise localization is important.

## Finding 6: Training stability is presented as controlled, but not deeply analyzed

**Claim in docs**

- `Docs6/04_Training_Strategy.md` includes gradient clipping, AMP, early stopping, and accumulation.

**Technical objection**

Those are stabilizers, not proof of stability. The docs do not discuss:

- BatchNorm behavior with batch size 4
- Whether encoder layers should be frozen initially
- LR warmup or scheduling
- Gradient norm monitoring
- Whether validation variance is high because the dataset is small

**Why it matters**

A small dataset with a pretrained BatchNorm encoder can look stable while still being noisy, overfit, and fragile to seed changes.

**Stronger answer or remediation**

Show multi-seed variance or at least discuss why single-seed early stopping is insufficient evidence of robustness.

## Finding 7: Overfitting controls are thin for such a small dataset

**Claim in docs**

- `Docs6/04_Training_Strategy.md` uses flips, rotations, early stopping, and weight decay.

**Technical objection**

That is a light regularization package for a small and biased dataset. The project excludes many augmentations because they may destroy forensic cues, which is sensible, but the result is still a weak defense against memorization of dataset-specific artifacts.

**Why it matters**

This creates a trap: the model may either overfit the dataset or underfit because the safe augmentation space is narrow.

**Stronger answer or remediation**

Admit that this is a real tradeoff. Then test at least one conservative augmentation ablation and report whether the model is seed-sensitive or split-sensitive.

## Finding 8: The image-level detector is architecturally weak

**Claim in docs**

- `Docs6/03_Model_Architecture.md` uses a derived image score rather than a classification head.

**Technical objection**

This design assumes that strong pixel probabilities automatically form a good image-level detector. That is not guaranteed. A noisy hot pixel or localized false positive can trigger an image-level decision.

**Why it matters**

If the use case includes screening images at scale, a heuristic detector is much harder to calibrate, compare, and defend than a learned binary head.

**Stronger answer or remediation**

Add a dual-task head or a separate classifier trained against image labels, then compare it against the heuristic aggregator.

## Finding 9: Engineering maturity is overstated if the project is presented beyond assignment scope

**Claim in docs**

- `Docs6/08_Engineering_Practices.md` highlights reproducibility, artifact structure, and deterministic setup.

**Technical objection**

The work is still notebook-centric. There is no modular package layout, no CLI/config separation, no tests around data transforms or metrics, no containerized environment, and no deployment path.

**Why it matters**

For an interview, there is a big difference between "well-organized notebook project" and "engineered ML system." The current state is much closer to the former.

**Stronger answer or remediation**

Describe it honestly as a reproducible experimental notebook pipeline and explain what would be required to turn it into a service or maintainable training repository.

## Finding 10: Reproducibility claims should be narrower

**Claim in docs**

- `Docs6/08_Engineering_Practices.md` emphasizes seeding, deterministic loaders, and manifest persistence.

**Technical objection**

Those controls help, but they do not make the entire training run fully reproducible across hardware, library versions, or mixed-precision kernels. The documentation occasionally sounds more deterministic than the setup really is.

**Why it matters**

Overstated reproducibility is easy to challenge in a technical interview, especially on GPU workloads.

**Stronger answer or remediation**

Say: "We improved repeatability within the same environment, but exact metric identity across runs and runtimes is not guaranteed."

## Bottom line

The model and training pipeline are acceptable as a baseline build, but the technical defense is shallow if the author presents it as a carefully justified architecture. The strongest honest framing is:

"We chose a simple pretrained segmentation baseline that is easy to train and analyze on limited hardware, then documented the main reasons it is probably not enough for a modern forensic system."
