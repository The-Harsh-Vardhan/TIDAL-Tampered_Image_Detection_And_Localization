# Assignment and Problem Framing Teardown

This note isolates the first question an interviewer should ask: did the author actually solve the assignment that was given, or did they solve a nearby problem and then write enough documentation to blur the difference?

## Requirement-by-requirement judgment

### 1. Tampered image detection and localization

**Judgment: Partial**

The localization half exists. The v6.5 notebooks produce pixel-level logits from a segmentation model (`notebooks/tamper_detection_v6.5_kaggle.ipynb:753-767`). So this is not a fake localization project.

The detection half is flimsy. Instead of a real detection head, image-level prediction is hacked together from the segmentation output using `max(prob_map)` (`notebooks/tamper_detection_v6.5_kaggle.ipynb:1236-1239`). `Docs7` still describes top-k mean (`Docs7/03_Model_Architecture.md:88-101`, `Docs7/05_Evaluation_Methodology.md:25`), so the project cannot even keep its own detection rule straight.

Verdict: the project can localize. It only sort of detects.

### 2. Model architecture design for predicting tampered regions

**Judgment: Partial**

Yes, there is an architecture. No, the design reasoning is not strong enough to claim this was carefully matched to the forensic problem. The argument is mostly:

- U-Net is common for segmentation.
- ResNet34 is pretrained.
- It fits on a T4.

That is baseline reasoning, not forensic reasoning (`Docs7/03_Model_Architecture.md:21-24`).

### 3. Freedom to choose architecture and loss functions

**Judgment: Technically satisfied, strategically underused**

The assignment gave freedom. The project used that freedom to pick the safest possible standard baseline. That is legal. It is not impressive.

The problem is not "you chose U-Net." The problem is "you chose U-Net and then defended it like that was a deep architectural insight."

### 4. Optimize for performance while remaining runnable on Colab or similar GPU

**Judgment: Partial**

The configuration is plausible for Colab/Kaggle T4-class hardware:

- 384x384 inputs
- batch size 4
- accumulation 4
- AMP enabled
- ResNet34 encoder

Evidence: `Docs7/00_Master_Report.md:90-99`, `Docs7/04_Training_Strategy.md:72-84`.

Why this is still only partial:

1. The assignment explicitly asks for a single Google Colab notebook (`Assignment.md:42-47`), but the repo sprawls across multiple notebook generations and runtime variants.
2. Static repo inspection does not prove the Colab notebook actually completes cleanly end to end.
3. The repo repeatedly confuses "plausible to run" with "verified to run."

### 5. Clear reasoning behind architectural decisions

**Judgment: Partial leaning weak**

The project does give reasons. They are just not strong reasons.

Examples:

- `Docs7/03_Model_Architecture.md:21-24` argues U-Net because dense prediction needs semantic + spatial fusion. Fine, but generic.
- `Docs7/03_Model_Architecture.md:23` claims ImageNet pretraining helps extract "compression artifacts" before fine-tuning. That is a stretch presented with too much confidence.
- `Docs7/03_Model_Architecture.md:132-139` dismisses transformer models mainly on VRAM and data-size grounds. That is a resource defense, not a task defense.

## The core framing problem

The project never cleanly answers: what is this system for?

Possible product stories:

1. Analyst assistance for already-suspicious images.
2. Mass-scale binary triage of incoming images.
3. Fine-grained localization to support human review.

Those are not the same task. The metric priorities and architecture tradeoffs change across them. The project avoids choosing, which lets it borrow the advantages of each story without paying the design cost of any of them.

That is why the image-level detection logic looks bolted on. Because it is.

## False or inflated assignment claims

### CASIA was not explicitly required

`Docs7/02_Dataset_and_Preprocessing.md:10` says "The assignment explicitly targets this dataset."

`Assignment.md:14-17` says the opposite. CASIA is just one example among several possible public datasets.

This is not harmless wording drift. It is the project rewriting the assignment to justify its own dataset choice.

### The deliverable is not cleanly single-notebook

The assignment says the entire implementation must be in a single Google Colab notebook (`Assignment.md:42-47`).

The repo contains:

- multiple historical notebooks
- both Kaggle and Colab variants
- scripts used to generate notebook content
- multiple documentation generations

That may be fine for development. It is not fine if the submission is presented like a final, singular artifact.

## Real-world relevance problem

The project repeatedly uses general language like "tamper detection" while training on classical CASIA-style manipulations only.

`Docs7/00_Master_Report.md:68-76` eventually declares GANs, deepfakes, video tampering, and other domains out of scope. Good. That honesty should have been the headline, not the footnote.

The safest honest framing is:

"This is a classical image-forgery localization baseline for CASIA-style splicing/copy-move data."

Anything broader is marketing.

## What a senior reviewer would conclude

The author understands how to assemble a reasonable assignment baseline. The author does not yet show strong control over problem framing, claim discipline, or submission hygiene. The project solves a narrower problem than it says, and the documentation keeps smoothing over that gap instead of confronting it directly.

## Immediate fixes

1. Replace all "assignment explicitly targets CASIA" language with the truth.
2. Name one official submission notebook and demote everything else to development history.
3. State one concrete use case and make the metric story serve that use case.
4. Reframe the project as classical tamper localization, not generic tamper detection.
5. Admit explicitly that image-level detection is heuristic and weaker than the localization component.
