# Audit7 Pro - Principal-Level Teardown

Scope: static audit of `Docs6/`, `Docs7/`, the v6/v6.5 notebooks, dataset references, research-paper inventory, and prior audits. This is not proof that any notebook trained end to end. It is proof of what the repo claims, what the code actually does, and where the submission starts lying to itself.

## ASSIGNMENT COMPLIANCE VERDICT

**Verdict: Partial**

This repo does build a segmentation-style tamper localization baseline. It does not deserve a full pass because the assignment was not just "make a U-Net and some plots." The assignment asked for tampered-image detection and localization, a clear architecture rationale, evaluation rigor, and a runnable single Colab-scale submission. The project only half-clears that bar.

Why it is not a fail:

1. There is a real pixel-mask model, not just an image classifier. The v6.5 notebooks instantiate `smp.Unet` with a ResNet34 encoder and produce per-pixel logits (`notebooks/tamper_detection_v6.5_kaggle.ipynb:753-767`).
2. The pipeline includes dataset validation, split persistence, checkpointing, threshold sweep, visualization, robustness tests, and optional Colab/Kaggle variants. That is a legitimate baseline attempt, not empty posturing.
3. The assignment required Colab-scale feasibility, not state of the art. A ResNet34 U-Net on 384x384 inputs is at least plausible for that constraint (`Assignment.md:26-30`, `Docs7/03_Model_Architecture.md:21-24`).

Why it is not a pass:

1. The deliverable is not clean. The assignment requires a single Google Colab notebook (`Assignment.md:42-47`), while the repo is a graveyard of revisions plus dual Kaggle/Colab variants and layered docs pretending that is the same thing.
2. The architecture rationale is weak. Most of the "why" boils down to "fits on T4" and "standard baseline," which is an execution convenience, not a forensic argument (`Docs7/03_Model_Architecture.md:21-24`, `Docs7/11_Research_Alignment.md:27-36`).
3. The evaluation story is compromised. `Docs7` still claims top-k image scoring, but v6.5 code uses `max(prob_map)` (`Docs7/03_Model_Architecture.md:88-101`, `Docs7/05_Evaluation_Methodology.md:25`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:1236-1239`). Mixed-set metrics are also inflated by empty-mask conventions (`Docs7/05_Evaluation_Methodology.md:52-61`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:856-891`).
4. Several "validation experiments" are documentation theater, not part of the actual executed pipeline (`Docs7/13_Validation_Experiments.md:139-147`).
5. The project keeps talking like it solved the general tamper-detection problem while actually training a classical CASIA-era RGB segmentation baseline on a leakage-prone legacy benchmark (`Docs7/02_Dataset_and_Preprocessing.md:17-21`, `Docs7/11_Research_Alignment.md:81-88`).

Bottom line: this is a credible internship baseline build wrapped in documentation that repeatedly overstates its rigor and coherence. That is the definition of `Partial`.

## PROJECT ROAST

### 1. Assignment compliance

**Roast**

This submission cannot decide what its final artifact is. The assignment asked for one Colab notebook. The repo answers with a zoo: `tamper_detection_v6_colab.ipynb`, `tamper_detection_v6_kaggle.ipynb`, `tamper_detection_v6.5_colab.ipynb`, `tamper_detection_v6.5_kaggle.ipynb`, plus stale documentation from older generations. That is not a polished submission. That is version sprawl pretending to be maturity.

Worse, `Docs7/02_Dataset_and_Preprocessing.md:10` claims "The assignment explicitly targets this dataset," which is simply false. `Assignment.md:14-17` lists CASIA as one example among several. The project is already bending the assignment text to excuse its own comfort zone.

**What a senior ML engineer would expect instead**

A single authoritative notebook, one runtime story, one scoring rule, one dataset narrative, and a requirement-by-requirement mapping that does not need forensic recovery work from the reviewer.

**Concrete fix**

Pick `tamper_detection_v6.5_colab.ipynb` as the only submission artifact, freeze it, delete ambiguity in the docs, and stop claiming the assignment mandated CASIA.

Evidence: `Assignment.md:14-17`, `Assignment.md:42-47`, `Docs7/02_Dataset_and_Preprocessing.md:10`, `Docs7/00_Master_Report.md:3-4`.

### 2. Problem understanding

**Roast**

The project says "tamper detection and localization" but never really defines the operational problem. Is this analyst assist? automated rejection? newsroom triage? moderation ranking? You never lock that down, so half the design choices float in mid-air. Then you glue a heuristic image-level detector on top of a segmentation model and act like that settles the product question. It does not. It just exposes that the project optimized the wrong thing first and patched the missing part later.

**What a senior ML engineer would expect instead**

A clear target workflow and explicit failure-cost priorities. If the real product is triage, then image-level recall and calibrated suspicion scores matter. If the product is analyst assistance, then boundary quality and false positive region behavior matter more.

**Concrete fix**

Write one target use case and map every metric to it. Then either defend localization-first as essential or admit this is mostly a localization demo with a weak detector bolted on.

Evidence: `Assignment.md:5-9`, `Docs7/03_Model_Architecture.md:86-101`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:1236-1239`.

### 3. Dataset choice

**Roast**

CASIA is a legacy benchmark, not a 2026 credibility shield. The project knows this and still leans on it like it is a get-out-of-jail card. You are training on classical splicing/copy-move data and then talking about tamper detection in general. That gap is huge. The dataset is old, small, biased, and structurally vulnerable to content leakage. The code checks path overlap and calls it "No data leakage detected." That is not leakage control. That is checking whether two strings are identical.

Then the docs cannot even describe the forgery types consistently. `Docs7/02_Dataset_and_Preprocessing.md:15` says splicing is `_S_`, copy-move is `_C_`, and removal exists. The v6.5 notebook uses `_D_` for splicing, `_S_` for copy-move, and throws everything else into `unknown` (`notebooks/tamper_detection_v6.5_kaggle.ipynb:374-381`). That is not a minor typo. That is the dataset semantics being documented wrong.

**What a senior ML engineer would expect instead**

An honest scope statement: classical forgery localization baseline on CASIA-derived masks, not modern universal tamper detection. They would also expect source-aware or duplicate-aware split controls, or at minimum a loud warning that generalization claims are weak.

**Concrete fix**

Keep CASIA if you want a baseline, but say exactly what it is good for and what it is bad for. Add duplicate-aware grouping or similarity-based split audits. Fix the forgery-type documentation so it matches the code.

Evidence: `Docs7/02_Dataset_and_Preprocessing.md:15-21`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:374-381`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:490-519`.

### 4. Model architecture

**Roast**

The architecture defense is shallow. "U-Net is standard" and "ResNet34 fits on T4" is not a forensic argument. It is a scheduling argument. For tamper localization, the hard question is whether an ImageNet-pretrained RGB encoder is actually the right feature extractor for compression noise, seam artifacts, resampling traces, and forensic inconsistencies. The project never answers that. It just repeats that ResNet34 is efficient and pretrained.

The docs also duck obvious alternatives. DeepLabV3+, SegFormer, HRNet, Mask2Former, dual-stream forensic models, edge-aware models, transformer hybrids - all of them are either waved away or punted to future work. That is fine for an assignment baseline, but stop dressing it up like a deeply reasoned design.

**What a senior ML engineer would expect instead**

A direct statement that this is a convenience baseline selected for implementation speed and memory budget, not because it is the best inductive bias for forensic evidence.

**Concrete fix**

Benchmark at least one serious alternative baseline or explicitly label the current model as the simplest defendable baseline. If you keep U-Net, explain what forensic signals it probably misses.

Evidence: `Docs7/03_Model_Architecture.md:21-24`, `Docs7/03_Model_Architecture.md:132-139`, `Docs7/11_Research_Alignment.md:31-36`, `Research_Paper_Analysis_Report.md:66-76`, `Research_Paper_Analysis_Report.md:143-148`.

### 5. Training strategy

**Roast**

The training stack is competent enough to run, but the reasoning is incomplete. `BCEDiceLoss` is implemented as batch-level Dice across the whole batch (`Docs7/04_Training_Strategy.md:18-25`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:804-811`). That means large masks can dominate small ones while the docs brag about small-region sensitivity. Nice story. Weak implementation.

You also use `BCEWithLogitsLoss` without `pos_weight`, despite repeatedly emphasizing severe foreground imbalance. Gradient accumulation is then used to simulate batch size 16, but BatchNorm inside the ResNet encoder still only sees micro-batches of 4. The project never seriously discusses whether that matters. It just assumes "effective batch size" means everything is solved. It is not.

**What a senior ML engineer would expect instead**

Per-sample Dice or a more explicit imbalance strategy, some discussion of BatchNorm under micro-batching, and at least one controlled ablation on loss variants or weighting.

**Concrete fix**

Switch Dice to per-image computation, test `pos_weight` or Tversky/Focal variants, and stop claiming the loss is tuned for small tampered regions unless you actually implement it that way.

Evidence: `Docs7/04_Training_Strategy.md:18-25`, `Docs7/04_Training_Strategy.md:28-45`, `Docs7/04_Training_Strategy.md:151-154`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:796-813`.

### 6. Evaluation methodology

**Roast**

This is where the project starts gaming itself. `Docs7/05_Evaluation_Methodology.md:25` says image-level detection uses top-k mean. The actual v6.5 code uses `tamper_score = probs[i].view(-1).max().item()` (`notebooks/tamper_detection_v6.5_kaggle.ipynb:1236`). That is not cosmetic drift. That changes detector behavior, threshold sensitivity, and any claim about robustness to isolated hot pixels.

Then the metric definitions hand out perfect localization scores to authentic true negatives and even assign recall `1.0` when ground truth is empty but the model predicts tampering (`notebooks/tamper_detection_v6.5_kaggle.ipynb:888-891`). Mixed-set averages can therefore look cleaner than the actual tampered-region performance. That is not rigorous evaluation. That is a background-heavy metric setup begging to flatter the model.

**What a senior ML engineer would expect instead**

Tampered-only localization as the headline, mixed-set metrics as context, macro and micro reporting, and separate calibration for image-level detection if it is not a learned head.

**Concrete fix**

Make tampered-only Pixel-F1/IoU the main claim, report authentic false-positive burden separately, mark empty-GT recall as not applicable, and stop reusing one segmentation threshold as if it magically calibrates the image-level detector too.

Evidence: `Docs7/05_Evaluation_Methodology.md:16-25`, `Docs7/05_Evaluation_Methodology.md:52-79`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:856-891`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:1210-1239`.

### 7. Validation experiments

**Roast**

`Docs7/13_Validation_Experiments.md` is the most polished part of the repo that is not actually part of the pipeline. Mask randomization, shortcut checks, prediction solidity, boundary-band analysis - all of it sounds smart. None of it is integrated into the standard notebook workflow. The doc itself admits these are post-hoc diagnostics requiring separate runs (`Docs7/13_Validation_Experiments.md:139-147`). So if the author talks about them like they validated the model, that is smoke.

**What a senior ML engineer would expect instead**

Either run the experiments and report results, or keep them clearly labeled as proposed follow-up work rather than evidence already earned.

**Concrete fix**

Move these tests from "smart document" to "actual executed appendix." If they were not run, say so bluntly and stop borrowing credibility from hypothetical experiments.

Evidence: `Docs7/13_Validation_Experiments.md:7-17`, `Docs7/13_Validation_Experiments.md:139-147`.

### 8. Explainability

**Roast**

The Grad-CAM story is more careful than most student projects, but it still has the classic problem: the implementation targets `output.mean()` for a segmentation model (`notebooks/tamper_detection_v6.5_kaggle.ipynb:1562-1565`) and then visualizes the best tampered samples only (`notebooks/tamper_detection_v6.5_kaggle.ipynb:1625-1629`). That is self-congratulatory sampling plus a fuzzy target. Of course the heatmaps can be made to look plausible.

The docs say Grad-CAM is only diagnostic. Good. Then keep it there. Do not treat it like proof that the model learned genuine forensic reasoning.

**What a senior ML engineer would expect instead**

Failure-case attribution, authentic false-positive examples, and quantitative sanity checks before making any serious interpretability claim.

**Concrete fix**

Restrict the claim to "rough sanity check," add hard failures and false positives to the visualization set, and stop pretending `output.mean()` heatmaps tell a clean causal story.

Evidence: `Docs7/07_Visualization_and_Explainability.md:19`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:1556-1568`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:1625-1629`.

### 9. Engineering quality

**Roast**

This project wants credit for engineering maturity because it has `CONFIG`, checkpoints, flags, and optional DataParallel. That is still a notebook pipeline. A notebook with a few helper functions is not a mature ML system. It is an organized experiment. The repo is also full of doc drift and version churn, which is exactly what real engineering is supposed to prevent.

Even inside `Docs7`, the story contradicts itself. `Docs7/12_Complete_Notebook_Structure.md:68-69` claims threshold-aware early stopping from the sweep and top-k image scoring. `Docs7/04_Training_Strategy.md:162-187` correctly says training validation uses fixed `0.5`. The notebook agrees with the latter (`notebooks/tamper_detection_v6.5_kaggle.ipynb:966-1008`, `1068-1075`). So the authoritative structure doc is not authoritative. Great.

**What a senior ML engineer would expect instead**

One source of truth, reproducible metrics definitions, and zero ambiguity about what the code actually does.

**Concrete fix**

Call it what it is: a reproducible notebook baseline. Then either collapse docs into one accurate technical report or accept that reviewers will stop trusting the written story.

Evidence: `Docs7/00_Master_Report.md:14`, `Docs7/12_Complete_Notebook_Structure.md:68-69`, `Docs7/04_Training_Strategy.md:162-187`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:966-1008`, `notebooks/tamper_detection_v6.5_kaggle.ipynb:1068-1075`.

### 10. Research awareness

**Roast**

The project clearly read papers. It did not clearly absorb them. `Research_Paper_Analysis_Report.md:51-76` and `Docs7/11_Research_Alignment.md:81-88` both acknowledge that stronger tamper localization systems use multi-domain fusion, noise streams, edge supervision, and transformer-style context. Then v6.5 stays RGB-only, single-stream, no edge loss, no dual-task head, no frequency domain, no real ablation against a stronger baseline. That is not research alignment. That is research-themed decoration around a baseline.

The repo knows what the next real steps are and still submits the same architecture story as if the gap is just optional polish. It is not. The papers are telling you the current baseline is missing the forensic signals that matter.

**What a senior ML engineer would expect instead**

Literature-informed humility: "we built a simple baseline and here is exactly how it falls short of the modern field."

**Concrete fix**

Stop using the paper list as decorative legitimacy. Either run one stronger comparison or state openly that the project is literature-aware but not literature-competitive.

Evidence: `Research_Paper_Analysis_Report.md:66-76`, `Research_Paper_Analysis_Report.md:143-148`, `Docs7/11_Research_Alignment.md:31-36`, `Docs7/11_Research_Alignment.md:81-88`.

## FINAL SUMMARY

### Top 10 weaknesses

1. The repo does not present one clean authoritative submission artifact even though the assignment requires a single Colab notebook.
2. `Docs7` falsely claims the assignment explicitly targets CASIA when the assignment only lists CASIA as one option.
3. The dataset story is weak: CASIA is old, narrow, and leakage-prone, while the docs overstate what path-overlap checks prove.
4. `Docs7` misdocuments forgery-type semantics relative to the notebook code.
5. Image-level detection is still a heuristic and the docs misstate which heuristic the code actually uses.
6. Mixed-set localization reporting is inflated by empty-mask handling and misleading recall semantics on empty ground truth.
7. The architecture justification is mostly "cheap and standard," not task-specific forensic reasoning.
8. The loss design does not really match the project's claims about small-region sensitivity and imbalance handling.
9. The validation-experiment section contains mostly unexecuted diagnostics, not earned evidence.
10. The repo cites stronger research directions while implementing none of the key forensic signals they emphasize.

### Top 5 strengths

1. The author correctly frames localization as a segmentation problem instead of stopping at image-level classification.
2. The v6.5 baseline includes practical training hygiene: dataset validation, split persistence, checkpointing, AMP, and threshold sweep.
3. The pipeline is plausibly runnable on Colab/Kaggle-class hardware.
4. The code includes tampered-only reporting and robustness testing instead of only showing cherry-picked masks.
5. The docs at least acknowledge several limitations instead of pretending the model solves modern tamper forensics.

### What the author clearly understands

1. Basic segmentation pipeline design for tamper localization.
2. How to assemble a practical PyTorch notebook baseline with sensible training utilities.
3. That dataset quality, leakage risk, and robustness matter in forensic tasks.
4. That explainability tools like Grad-CAM are diagnostic rather than formal proof.
5. That the current system is simpler than multi-trace and transformer-heavy research models.

### What the author likely does not understand yet

1. The difference between "the docs discuss a validation idea" and "the project actually validated the claim."
2. How easily empty-mask metric conventions can distort mixed-set localization conclusions.
3. Why forensic feature design is not the same as generic semantic segmentation transfer learning.
4. Why one threshold for segmentation and image-level detection is a convenience hack, not a principled evaluation choice.
5. How much credibility is lost when documentation contradicts implementation on core behavior.
6. How weak "it fits on T4" sounds as an architecture defense in a senior technical review.

### Concrete steps to improve before submission

1. Freeze one final Colab notebook and make every doc point to it.
2. Correct all scoring-rule and split-strategy documentation so it matches the actual code.
3. Rewrite the assignment-compliance story without pretending CASIA was mandatory.
4. Make tampered-only localization the headline metric and report authentic false positives separately.
5. Replace the image-level `max(prob_map)` shortcut with a learned head or a better-calibrated aggregator.
6. Run at least one real shortcut-learning test, not just a proposed one.
7. Add duplicate-aware or source-aware split checks, or downgrade generalization claims even more aggressively.
8. Tighten the architecture defense by benchmarking or at least discussing DeepLabV3+/SegFormer honestly.
9. Fix the loss design if you want to keep claiming small-region sensitivity.
10. Reframe the project as a classical-forgery baseline, not a general tamper-detection system.
