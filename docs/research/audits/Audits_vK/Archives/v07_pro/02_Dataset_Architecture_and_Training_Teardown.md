# Dataset, Architecture, and Training Teardown

This note covers the technical core: the data, the model, and the training stack. This is where a baseline either looks honest or starts pretending convenience is insight.

## 1. Dataset choice: CASIA is the easy answer, not the strong answer

CASIA is convenient because it has masks and decades of inertia behind it. That is exactly why it needs to be handled carefully. The project mostly handles it like a comfortable default.

### What is wrong with the current CASIA story

1. **It is old.**
   The dataset is a legacy classical-forgery benchmark. It does not reflect diffusion edits, localized generative inpainting, AI retouching, or modern multi-stage editing pipelines.

2. **It is narrow.**
   The project is explicitly scoped to classical tampering only (`Docs7/00_Master_Report.md:68-76`), which is honest. The problem is that the rest of the presentation still talks like it built a broad tamper detector.

3. **Leakage control is weak.**
   The notebook checks file-path overlap only (`notebooks/tamper_detection_v6.5_kaggle.ipynb:512-519`). That does not detect source-scene overlap, near duplicates, derivative manipulations, or reused content.

4. **The docs do not even describe the forgery-type parsing correctly.**
   `Docs7/02_Dataset_and_Preprocessing.md:15` says splicing is `_S_`, copy-move is `_C_`, and removal exists. The code uses `_D_` for splicing, `_S_` for copy-move, and anything else becomes `unknown` (`notebooks/tamper_detection_v6.5_kaggle.ipynb:374-381`).

### What a senior reviewer expects

They do not expect you to magically solve the dataset problem. They expect you to stop overstating what the dataset proves.

Minimum credible statement:

"CASIA is a legacy benchmark useful for a classical-manipulation baseline, but it is not evidence of modern forgery generalization."

### What to improve

1. Add duplicate-aware or similarity-aware split analysis.
2. Report results by tamper area and forgery type, not just overall averages.
3. State clearly that modern AI-generated manipulations are not covered.
4. Fix the docs so the dataset semantics actually match the code.

## 2. Architecture: U-Net + ResNet34 is defensible, not deeply justified

The base model is:

```python
smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=1)
```

Evidence: `Docs7/03_Model_Architecture.md:7-18`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:753-757`.

That is a valid baseline. It is also the most generic safe option in the room.

### Why the current justification is weak

The docs argue:

- U-Net is good for dense prediction.
- ResNet34 is a good quality-to-cost tradeoff.
- Transformers are heavy and CASIA is small.

All of that is true. None of it proves the model is well matched to forensic evidence.

The missing question is:

Why should an RGB ImageNet encoder be a strong default for forensic traces that often live in noise residuals, compression inconsistencies, and seam artifacts rather than object semantics?

The project never answers that at a technical level.

### The alternatives it mostly dodges

The docs mention transformer-based systems as future work (`Docs7/03_Model_Architecture.md:132-139`, `Docs7/11_Research_Alignment.md:113-118`), but they do not seriously address:

- DeepLabV3+ as a standard irregular-mask baseline
- SegFormer as a lighter modern encoder-decoder alternative
- dual-stream RGB + forensic residual designs
- edge-aware variants

That omission matters because the project is for tamper localization, not generic road-scene segmentation.

## 3. Research awareness exposes the model weakness

The most damaging evidence against the model choice is in the project's own research notes.

`Research_Paper_Analysis_Report.md:66-76` highlights:

- spatial + frequency fusion
- multi-stream architectures
- edge enhancement
- self-attention
- cross-attention
- dual-task designs

Then `Research_Paper_Analysis_Report.md:143-148` admits the current system lacks:

- forensic feature extraction
- edge supervision
- multi-domain fusion
- attention mechanisms
- stronger augmentation

That means the project already knows the baseline is underpowered for modern forensic reasoning. It just keeps submitting it anyway and hoping "assignment scope" covers the gap.

## 4. RGB-only input is a major limitation

`Docs7/03_Model_Architecture.md:15` locks `in_channels=3`.

That means the model sees RGB only. No SRM residuals. No ELA. No frequency-domain features. No chrominance emphasis. No edge stream.

For natural-image segmentation, RGB is normal.

For image forensics, RGB-only is the lazy path.

The project knows this. `Docs7/11_Research_Alignment.md:81-88` explicitly says multi-domain fusion improves performance. `Research_Paper_Analysis_Report.md:57-60` and `66-68` say the same thing.

## 5. The image-level detector is architecturally weak

`Docs7` claims top-k mean; v6.5 code uses `max(prob_map)` (`Docs7/03_Model_Architecture.md:88-101`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:1236`).

Either way, it is still a heuristic. There is no learned binary head. That makes the detection side of the assignment much weaker than the localization side.

This is especially bad because the assignment explicitly asks for both detection and localization (`Assignment.md:5-8`, `34-35`).

The author had research evidence for a better approach too. `Research_Paper_Analysis_Report.md:77` and `203-206` discuss dual-task classification + segmentation. The project just did not implement it.

## 6. Training strategy: good baseline hygiene, weak forensic rigor

### What is good

The training loop is operationally reasonable:

- AMP
- gradient accumulation
- clipping
- checkpointing
- AdamW with differential learning rates
- early stopping

Evidence: `Docs7/04_Training_Strategy.md:49-64`, `87-105`, `109-155`, `190-210`.

That is solid notebook engineering.

### What is weak

#### Batch-level Dice

The Dice term is computed across the whole batch:

```python
intersection = (probs * targets).sum()
dice_loss = 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
```

Evidence: `Docs7/04_Training_Strategy.md:18-25`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:804-811`.

That means large masks can dominate small ones. So the docs' claim that Dice helps with tiny tampered regions is only partly true.

#### No explicit positive reweighting

The docs repeatedly stress class imbalance (`Docs7/04_Training_Strategy.md:28-31`), but `BCEWithLogitsLoss` is used with default settings. No `pos_weight`. No focal variant. No explicit hard-example handling.

#### Micro-batch BatchNorm issue

The effective batch is said to be 16 through accumulation (`Docs7/04_Training_Strategy.md:151`). Fine. BatchNorm still sees 4 images per forward pass. The docs do not seriously engage with that.

#### No multi-scale training

`Docs7/00_Master_Report.md:63` explicitly says there is no multi-scale training or test-time augmentation. On a forgery-localization task where tiny regions matter, that is a real omission.

## 7. Colab/Kaggle feasibility: plausible, not fully proven

The runtime design is believable for T4-class GPUs. That is a positive.

But feasibility is still presented more confidently than the repo earns:

1. There is no end-to-end runtime proof in the repo.
2. The assignment wanted a single Colab deliverable; the project splits effort across Kaggle and Colab.
3. DataParallel support inside a notebook is not a meaningful sign of maturity. It is just another flag.

## Bottom line

This is a baseline that can probably train. It is not a strong forensic design, and the project already contains enough research material to know that.

If the author presents this as "the right architecture for tamper detection," that is weak.

If the author presents it as "a resource-aware classical-forgery baseline with known forensic blind spots," that is honest.

## High-value fixes

1. Add a real detection head instead of relying on `max(prob_map)`.
2. Rework Dice to per-sample computation.
3. Add at least one forensic side channel: ELA, SRM, or edge supervision.
4. Test one serious alternative baseline instead of pretending U-Net won by default.
5. Downgrade all generalization claims until leakage and dataset-bias risks are better controlled.
