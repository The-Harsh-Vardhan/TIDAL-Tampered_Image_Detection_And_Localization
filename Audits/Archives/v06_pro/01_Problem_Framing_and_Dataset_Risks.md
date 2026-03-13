# Problem Framing and Dataset Risks

This note focuses on the front end of the project definition: what problem is actually being solved, whether the dataset supports that claim, and what risks are introduced by the current split and annotation story.

Each finding follows the same pattern:

- Claim in docs
- Technical objection
- Why it matters
- Stronger answer or remediation

## Finding 1: The project states the task, but does not define the operating problem

**Claim in docs**

- `Docs6/00_Master_Report.md` and `Docs6/01_System_Architecture.md` describe an image tampering detection and localization pipeline.
- `Docs6/11_Research_Alignment.md` says localization should be treated as dense prediction rather than classification only.

**Technical objection**

The documentation never specifies who consumes the output and what decision the model supports. Is this a triage tool for human analysts? A hard automated gate? A newsroom assist tool? A moderation ranking model? Those choices change what should be optimized.

**Why it matters**

Without a clear use case, the architecture and metrics are only loosely justified. Pixel F1 may be appropriate for one scenario and almost irrelevant for another. If the real operational need is "flag suspicious images for review," then a weak image-level detector undermines the whole system.

**Stronger answer or remediation**

State one target workflow explicitly and tie metrics to it. Example: "This system is an analyst-assist model that proposes suspicious regions on already-flagged images; therefore recall on tampered images and spatial localization quality matter more than standalone binary precision."

## Finding 2: Localization is asserted as necessary, not demonstrated as necessary

**Claim in docs**

- `Docs6/11_Research_Alignment.md` argues that localization should be treated as segmentation.
- `Docs6/03_Model_Architecture.md` centers the project on U-Net and pixel masks.

**Technical objection**

The docs do not explain why localization is required for the chosen application instead of image-level classification, retrieval, or analyst ranking. The project later adds image-level decisions through a heuristic score anyway, which suggests the binary decision is still important.

**Why it matters**

If image-level tamper detection is the real deployment need, the system is optimizing the wrong objective first and backfilling classification with a weak heuristic.

**Stronger answer or remediation**

Explain the minimum product requirement. If localization is needed for trust and analyst efficiency, say so and quantify the benefit. If classification is also required, treat it as a first-class task with a dedicated head and metric suite.

## Finding 3: CASIA is convenient, but outdated and structurally narrow

**Claim in docs**

- `Docs6/02_Dataset_and_Preprocessing.md` selects CASIA because it has image-mask pairs and is convenient on Kaggle.
- `Docs6/13_References.md` links the project to CASIA 2.0 from 2013.

**Technical objection**

CASIA is a legacy benchmark built around classical manipulations. It does not represent the editing ecosystem the project would be compared against in a modern interview: diffusion retouching, semantic inpainting, local relighting, generative face swaps, or cross-tool editing pipelines.

**Why it matters**

A reviewer will immediately discount broad claims such as "tamper detection system" if the evidence base is classical splicing and copy-move only.

**Stronger answer or remediation**

Reframe the scope honestly: "This is a classical manipulation localization baseline." If broader claims are needed, add at least one newer benchmark or a targeted stress set of modern edits.

## Finding 4: Dataset bias is acknowledged, but not operationalized

**Claim in docs**

- `Docs6/02_Dataset_and_Preprocessing.md` notes small size, noisy masks, and classical-only tampering.
- `Docs6/00_Master_Report.md` calls out annotation quality and narrow scope.

**Technical objection**

The project lists limitations, but does not show what they do to the experiment design. There is no subgroup analysis by mask size, no test stratification by difficulty, no discussion of camera or scene bias, and no attempt to estimate how much of the score may come from dataset shortcuts.

**Why it matters**

A limitation section is not a substitute for a risk-aware evaluation plan. Simply admitting dataset weaknesses does not prevent misleading conclusions.

**Stronger answer or remediation**

Tie each limitation to a mitigation. Example: "Because mask quality is noisy, we report both region overlap and qualitative boundary failures. Because CASIA is narrow, we avoid any claim about modern AI-generated edits."

## Finding 5: Leakage prevention is weaker than the documentation tone suggests

**Claim in docs**

- `Docs6/02_Dataset_and_Preprocessing.md` verifies zero file overlap across train, validation, and test splits.
- `Docs6/00_Master_Report.md` presents leakage checks as an important issue that was resolved.

**Technical objection**

Path disjointness is not the same as content disjointness. CASIA does not provide source-image grouping metadata, and the docs admit that. That means related derivatives, alternate tampered versions, near duplicates, or different manipulations from the same source image could still cross splits.

**Why it matters**

If related images leak across splits, the measured generalization can be much higher than real-world generalization. This is one of the first things a skeptical ML reviewer will challenge.

**Stronger answer or remediation**

Use perceptual hashing, CLIP embedding similarity, or manual source grouping heuristics to reduce near-duplicate leakage. If that is not feasible, explicitly downgrade the strength of all generalization claims.

## Finding 6: Stratifying only by forgery type is too weak

**Claim in docs**

- `Docs6/02_Dataset_and_Preprocessing.md` stratifies by `forgery_type` with classes `authentic`, `splicing`, and `copy-move`.

**Technical objection**

This preserves coarse class balance, but it does not balance image difficulty, mask size, object scale, post-processing severity, or scene similarity. For localization tasks, those factors are often more important than high-level manipulation type.

**Why it matters**

A stable class ratio can hide unstable segmentation difficulty. The test set may still be much easier or much harder than the training set in ways the current split does not control.

**Stronger answer or remediation**

Track and stratify by mask-area buckets at minimum. Better still, add analyses by object size, tampered-area fraction, and scene duplication risk.

## Finding 7: Annotation quality is mentioned, but not measured

**Claim in docs**

- `Docs6/00_Master_Report.md` and `Docs6/02_Dataset_and_Preprocessing.md` mention coarse or noisy boundaries in CASIA-derived masks.

**Technical objection**

The project accepts the masks as ground truth without any estimate of label uncertainty, edge ambiguity, or missing regions.

**Why it matters**

If labels are noisy, region overlap metrics become harder to interpret. A model that is "wrong" against the mask may be more correct than the annotation at the boundary.

**Stronger answer or remediation**

Add a small manual audit of random masks, report obvious failure modes in the labels, and avoid over-interpreting small metric differences between experiments.

## Finding 8: Image resizing may erase forensic signal before the model sees it

**Claim in docs**

- `Docs6/00_Master_Report.md`, `Docs6/01_System_Architecture.md`, and `Docs6/04_Training_Strategy.md` standardize images to 384 x 384 for T4 memory headroom.

**Technical objection**

That resolution choice is driven by hardware, not forensic signal preservation. Small pasted regions, seam artifacts, and local compression inconsistencies may be weakened or blurred by uniform resizing.

**Why it matters**

The project may underperform on exactly the subtle edits it claims to localize, and the evaluation would not reveal whether resizing is the cause.

**Stronger answer or remediation**

Run at least one ablation on higher resolution, tiled inference, or patch-based training. If that is not possible, treat the resize choice as a known limitation rather than a neutral preprocessing step.

## Bottom line

The dataset story is the most fragile part of the project. The documentation is honest enough to mention several limitations, but the experiment design does not compensate for them. In an interview, the safest defensible claim is:

"This is a baseline classical-tamper localization project on a legacy benchmark, useful for demonstrating pipeline design, not for proving robust modern forgery detection."
