# 03 - Model Architecture and ML Reasoning Audit

## Bottom line

The architecture is a defensible baseline and a weak argument. The notebook uses a standard SMP U-Net with a ResNet34 encoder because it is easy to stand up, pretrained, and likely to fit on commodity GPUs. That is fine as a starting point. It is not fine as if it were strong task-specific reasoning.

## 1. U-Net plus ResNet34 is a convenience baseline

Cell 20 builds:

- `smp.Unet`
- `encoder_name='resnet34'`
- ImageNet-pretrained encoder
- one-channel segmentation output

That stack is common because it is easy, documented, and stable. It is not automatically the right choice for image forensics. Tamper localization is not generic semantic segmentation. It often depends on compression artifacts, noise residuals, inconsistent acquisition traces, and subtle copy-paste inconsistencies that raw RGB backbones do not model particularly well.

Using a standard segmentation baseline is acceptable. Pretending the baseline is well-justified for the forensic problem is not.

## 2. The notebook outsources its reasoning to Docs8

Cell 19 says the architecture is retained from v6.5 because `Docs8 Section 03` determined the architecture is not the primary bottleneck. That is weak. The notebook is supposed to carry the reasoning itself. Instead it tells the reviewer to trust an external planning memo.

Worse, this is circular logic. The project chose the architecture, then cites its own documentation to claim the architecture is not the issue, but never shows an ablation against obvious alternatives. That is not evidence. That is self-endorsement.

## 3. There is no proper detection head

This is the most important architectural miss. The assignment requires detection and localization. The model only predicts a mask. Later, Cell 33 turns the maximum mask probability into an image-level score.

So what is the actual architecture?

- A segmentation model.
- No learned image-level branch.
- No shared classification head.
- No multi-task loss.
- No calibrated image-level predictor.

That means the project solves one task and improvises the other.

## 4. No comparison against obvious alternatives

If you want to keep U-Net plus ResNet34, fine. Then say clearly it is a baseline and benchmark at least one alternative. The notebook does not do that. It does not compare against:

- DeepLabV3+,
- a lightweight FPN variant,
- a dual-head segmentation plus classification model,
- a multi-stream forensic input design,
- even a different encoder family.

The notebook does mention some of these ideas indirectly through Docs8, which makes the lack of experiments look even worse. The author knows the alternatives and still chooses not to test them.

## 5. RGB-only modeling is an unaddressed limitation

The entire pipeline runs on normalized RGB images in Cells 15 and 16. That is reasonable for a baseline, but it should be described as a known limitation, not as if it were enough for the full forensic task. A lot of tamper evidence is not semantic and not cleanly visible in RGB appearance.

This limitation becomes more serious for copy-move manipulation, where the pasted region comes from the same image distribution. Plain RGB segmentation features are much more likely to lean on boundaries and low-level artifacts than actual manipulation reasoning.

## 6. One thing the notebook does get right

Cell 20 at least performs a shape sanity check on the model output. That is basic competence, but it is real competence. The problem is not that the author cannot build a segmentation model. The problem is that they are overselling what this architecture proves.

## Verdict

The architecture is acceptable only if it is presented honestly:

- segmentation-first baseline,
- not a strong forensic design,
- not a complete detection-and-localization architecture,
- not justified beyond convenience and prior familiarity.

That honesty is missing. As submitted, the architecture story is superficial. It should either be narrowed to "baseline localization model" or upgraded with real multi-task modeling and at least one alternative comparison.
