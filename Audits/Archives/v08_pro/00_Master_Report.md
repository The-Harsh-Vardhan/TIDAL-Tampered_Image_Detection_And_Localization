# Audit8 Pro - Principal-Level Documentation Audit

Scope: static audit of `Docs8/` as the claimed refined design after `Docs7/`, `Audit7 Pro/`, and `Run01`. This is not proof that Notebook v8 was executed. It is proof of what `Docs8` now claims, what it still leaves undone, and whether the documentation finally deserves trust.

## ASSIGNMENT COMPLIANCE VERDICT

**Verdict: Partial**

`Docs8` is materially better than `Docs7` in one important way: it finally admits several ugly truths that the older documentation tried to blur. It leads with tampered-only performance concerns, admits copy-move near-failure, calls out missing `pos_weight`, acknowledges heuristic image-level detection, and explicitly says U-Net/ResNet34 is a stable baseline rather than some grand forensic revelation (`Docs8/00_Project_Evolution_Summary.md:82-96`, `Docs8/03_Model_Architecture_Evolution.md:76-89`, `Docs8/05_Evaluation_Methodology_Evolution.md:66-93`).

That improvement is real. It is also not enough for a pass.

Why it is not a pass:

1. The assignment requires a single Google Colab notebook deliverable. `Docs8` still marks that only as partial and admits Colab was not verified end to end (`Assignment.md:42-47`, `Docs8/01_Assignment_Requirement_Alignment.md:57-67`).
2. The key technical repairs are still planned, not implemented: `pos_weight`, per-sample Dice, scheduler, expanded augmentation, Boundary F1, separate image-level calibration, and a learned detection head all sit in future-tense v8 checklists (`Docs8/04_Training_Strategy_Evolution.md:89-186`, `Docs8/05_Evaluation_Methodology_Evolution.md:103-178`, `Docs8/08_Notebook_V8_Implementation_Plan.md:15-263`).
3. The detector is still a heuristic `max(prob_map)` pipeline and the hardest assignment-relevant case, copy-move, is still weak at F1=0.3105 (`Docs8/06_Run01_Results_Analysis.md:93`, `Docs8/06_Run01_Results_Analysis.md:81-99`, `Docs8/00_Project_Evolution_Summary.md:91-96`).
4. `Docs8` is more honest about leakage risk, but it still slips and declares "No bugs or data leakage detected" while also admitting only path overlap was checked and perceptual-duplicate checks were not run (`Docs8/00_Project_Evolution_Summary.md:75-78`, `Docs8/01_Assignment_Requirement_Alignment.md:17-18`, `Docs8/02_Dataset_Evolution.md:94-96`).
5. The documentation lineage is cleaner in prose than in references. `Docs8` says it bridges `Docs7`, `Audit7 Pro`, and `Run01`, but the actual citations and source attributions mostly point back to `Audit6 Pro` and `Audit 6.5 Notebook`, not `Audit7 Pro` (`Docs8/00_Project_Evolution_Summary.md:5-11`, `Docs8/10_References.md:13-18`, `Docs8/08_Notebook_V8_Implementation_Plan.md:23-24`, `Docs8/08_Notebook_V8_Implementation_Plan.md:91-92`, `Docs8/08_Notebook_V8_Implementation_Plan.md:123-124`).

Bottom line: `Docs8` is a better confession. It is not yet a better finished submission.

## PROJECT ROAST

### 1. Docs8 fixed the tone before it fixed the project

**Roast**

The author finally stopped lying to themselves quite as hard. Great. The problem is that `Docs8` mostly upgrades the honesty layer while the engineering layer remains a to-do list.

**Why this is technically a problem**

Assignment compliance is about what the project does, not how elegantly the documentation admits its own flaws. `Docs8` repeatedly says "v8 must add" the real fixes instead of proving those fixes exist (`Docs8/01_Assignment_Requirement_Alignment.md:20-23`, `Docs8/04_Training_Strategy_Evolution.md:89-186`, `Docs8/05_Evaluation_Methodology_Evolution.md:103-178`).

**What a senior ML engineer would expect instead**

A clean distinction between executed behavior and planned improvements, followed by a final deliverable where the critical fixes are already integrated rather than narrated.

### 2. The assignment still asked for one Colab notebook, not a design novella

**Roast**

`Docs8` talks like the delivery problem is solved while its own requirement matrix still says the single Colab notebook is only partial. That is not a minor paperwork issue. That is the assignment deliverable.

**Why this is technically a problem**

`Assignment.md` requires the entire implementation in a single Google Colab notebook (`Assignment.md:42-47`). `Docs8` marks that row as partial, cites a Kaggle notebook as evidence, and says Colab still needs end-to-end verification (`Docs8/01_Assignment_Requirement_Alignment.md:57-67`).

**What a senior ML engineer would expect instead**

One authoritative notebook, one validated runtime path, and zero ambiguity about whether the submission artifact actually runs on the required platform.

### 3. CASIA is still being smuggled in as destiny instead of choice

**Roast**

The old documentation lied that the assignment explicitly targeted CASIA. `Docs8` improves the surrounding discussion, then quietly keeps a softer version of the same bad habit by calling CASIA "the assignment's expected dataset."

**Why this is technically a problem**

The assignment explicitly says use one or more public datasets and lists CASIA as an example, not a mandate (`Assignment.md:14-17`). `Docs8/02_Dataset_Evolution.md:112` still rewrites that into a stronger claim than the assignment makes.

**What a senior ML engineer would expect instead**

"We chose CASIA because it is convenient for a classical baseline, not because the assignment forced it." That is the honest version.

### 4. The project still detects with a hack

**Roast**

The localization model exists. The detection story is still a bolt-on trick. `max(prob_map)` is not a principled image-level detector. It is a desperation shortcut that survived because nobody wanted to budget a real head.

**Why this is technically a problem**

The assignment requires both detection and localization (`Assignment.md:5-8`, `Assignment.md:34-35`). `Docs8` openly documents that image-level scoring remains heuristic and that a learned classification head is merely a future experiment (`Docs8/01_Assignment_Requirement_Alignment.md:43-51`, `Docs8/03_Model_Architecture_Evolution.md:112-115`, `Docs8/09_Future_Experiments.md:128-139`).

**What a senior ML engineer would expect instead**

Either a learned dual-task design or a very explicit statement that the current system is localization-first with a weaker heuristic detector attached.

### 5. Copy-move is still the part where the project falls apart

**Roast**

The assignment offers bonus credit for subtle tampering. The project's answer for copy-move is basically "yes, it is bad, but we have ideas." That is not robustness. That is an apology memo.

**Why this is technically a problem**

`Docs8` records copy-move F1 at 0.3105, worse than splicing and dragging down the tampered-only score (`Docs8/00_Project_Evolution_Summary.md:51-54`, `Docs8/01_Assignment_Requirement_Alignment.md:73-80`, `Docs8/11_Training_Failure_Cases.md:52-95`). Since copy-move is exactly the kind of subtle manipulation the assignment calls out, weak handling here matters more than a polished average metric.

**What a senior ML engineer would expect instead**

Either materially improved copy-move performance or a humbler claim that the current baseline is not strong on subtle same-image manipulations.

### 6. The architecture reasoning got more honest, not more strong

**Roast**

`Docs8` finally admits U-Net/ResNet34 is a baseline chosen for stability, not because it is the right forensic inductive bias. That is better than the old story. It is still a confession that the project is deferring the hard architecture question.

**Why this is technically a problem**

The documentation now says the serious alternatives - `DeepLabV3+`, forensic input streams, transformers, learned classification head - are all postponed to later phases (`Docs8/03_Model_Architecture_Evolution.md:88-115`). That means the project still has not proven the chosen architecture is a good answer to the forensic problem. It has only proven it is a manageable baseline.

**What a senior ML engineer would expect instead**

At least one meaningful comparison or a clearly delimited claim: "this is the simplest defensible baseline, not the best design."

### 7. The training fixes are still written in the future tense because the current training logic is still weak

**Roast**

`Docs8` correctly identifies the holes: no `pos_weight`, batch-level Dice, no scheduler, weak augmentation. Then it writes a nice implementation plan instead of delivering a corrected training pipeline.

**Why this is technically a problem**

Those are not decorative details. They are central to why the current model needs a threshold of 0.1327 and overfits by epoch 15 (`Docs8/04_Training_Strategy_Evolution.md:68-85`, `Docs8/08_Notebook_V8_Implementation_Plan.md:19-68`).

**What a senior ML engineer would expect instead**

The corrected loss and scheduler logic already integrated in the final submission, with the old failure mode retired instead of documented.

### 8. The evaluation section is more credible than Docs7, but still not fully earned

**Roast**

`Docs8` does the right thing by centering tampered-only metrics and calling mixed-set inflation out. Fine. But Boundary F1, mask-size stratification, separate image-level thresholding, and improved calibration are still proposed upgrades, not evidence.

**Why this is technically a problem**

The current trustworthy part is the critique, not the repaired evaluation pipeline (`Docs8/05_Evaluation_Methodology_Evolution.md:103-178`). The project still has not demonstrated the improved methodology in an executed run.

**What a senior ML engineer would expect instead**

One evaluated run using the new reporting scheme, not just a reporting template.

### 9. The shortcut-learning section sounds quantitative, but it is still inference stacked on inference

**Roast**

The shortcut-learning doc tries to estimate the exact share of performance coming from artifact reliance. That looks analytical until you notice the whole estimate is built from nuisance-degradation gaps and extrapolation.

**Why this is technically a problem**

`Docs8/07_Shortcut_Learning_Risk_Assessment.md:81-95` converts robustness drops into an estimated feature budget. That is a plausible hypothesis, not a rigorous falsification test. No mask randomization, no content-matched controls, no feature ablation with executed evidence. Just a stronger-sounding story.

**What a senior ML engineer would expect instead**

Call it a hypothesis and validate it with executed falsification experiments before assigning percentages to shortcut reliance.

### 10. The documentation trust problem is reduced, not resolved

**Roast**

`Docs8` says it is the response to `Audit7 Pro`, then builds much of its evidence trail around `Audit6 Pro` and `Audit 6.5 Notebook`. That makes the lineage look curated rather than clean.

**Why this is technically a problem**

If the new documentation really incorporates `Audit7 Pro`, the references should show that. Instead, `Docs8/10_References.md:15-18` cites `Audit6 Pro` and `Audit 6.5 Notebook` as the key upstream audit sources, and the implementation plan repeatedly cites those same older audits (`Docs8/08_Notebook_V8_Implementation_Plan.md:23-24`, `Docs8/08_Notebook_V8_Implementation_Plan.md:50`, `Docs8/08_Notebook_V8_Implementation_Plan.md:91-92`, `Docs8/08_Notebook_V8_Implementation_Plan.md:123-124`, `Docs8/08_Notebook_V8_Implementation_Plan.md:170`, `Docs8/08_Notebook_V8_Implementation_Plan.md:215`). That is not fatal, but it is sloppy enough to trigger trust questions.

**What a senior ML engineer would expect instead**

One consistent audit lineage, with the latest critical review serving as the obvious upstream reference instead of a half-updated mixture.

## TOP 10 WEAKNESSES

1. `Docs8` is a better self-critique than `Docs7`, but it still documents planned fixes rather than a corrected final submission.
2. The project still does not satisfy the "single Google Colab notebook" requirement cleanly.
3. CASIA is still framed too aggressively as the expected dataset even though the assignment only lists it as one option.
4. Image-level detection is still a heuristic `max(prob_map)` rule instead of a learned component.
5. Copy-move performance remains weak on the assignment's hardest bonus-relevant manipulation type.
6. U-Net/ResNet34 is now defended honestly as a baseline, but still not justified as a strong forensic architecture.
7. Critical training repairs (`pos_weight`, per-sample Dice, scheduler, stronger augmentation) remain future tense.
8. Evaluation improvements are mostly specified, not executed.
9. Shortcut-learning claims are stronger than the underlying validation evidence.
10. Documentation credibility is improved but still compromised by inconsistent audit lineage and unresolved checklist-style cleanup.

## TOP 5 STRENGTHS

1. `Docs8` is substantially more honest than `Docs7` about mixed-set inflation, copy-move weakness, heuristic detection, and RGB-only limitations.
2. The document set uses `Run01` evidence instead of pure design theater, which is a real upgrade in technical maturity.
3. The author now explicitly distinguishes baseline stability from forensic optimality in the architecture discussion.
4. The training and evaluation docs identify the right first-order fixes: `pos_weight`, scheduler, augmentation, tampered-only reporting, and better calibration.
5. Research awareness is better framed than before: stronger alternatives are acknowledged instead of silently ignored.

## WHAT THE AUTHOR CLEARLY UNDERSTANDS

The author understands how to audit a failing ML baseline once real results exist. `Docs8` shows clear awareness of class imbalance, calibration problems, metric inflation, copy-move difficulty, shortcut-learning risk, and the difference between a stable segmentation baseline and a stronger forensic architecture. The author also understands practical notebook engineering better than the average internship submission: config-driven workflow, checkpointing, W&B hooks, and explicit failure-case tracking are all real strengths.

## WHAT THE AUTHOR LIKELY DOES NOT UNDERSTAND YET

The author still appears to underestimate the gap between "we identified the fix" and "the project is now fixed." There is also still a weak instinct to treat improved documentation as partial credit for unimplemented technical work. The project remains too willing to infer strong conclusions from partial evidence: leakage is declared absent after path checks, shortcut reliance gets pseudo-quantified without a real falsification test, and assignment compliance is scored generously even while the Colab deliverable remains partial. The author also still has not fully internalized that a heuristic detector attached to a segmentation model is not the same as a properly designed joint detection-and-localization system.

## HOW TO FIX THIS PROJECT

1. Stop treating `Docs8` as the finish line. Implement the P0 fixes first: `pos_weight`, scheduler, per-sample Dice, stronger augmentation, and the `cudnn.benchmark` cleanup.
2. Produce one verified Colab notebook and make every document point to that exact artifact.
3. Rewrite the dataset framing so CASIA is presented as a chosen baseline, not as an assignment expectation.
4. Replace `max(prob_map)` with a learned image-level head or clearly label the detector as provisional.
5. Run one executed v8 result set using the revised evaluation protocol, not just a template for one.
6. Run at least one real shortcut-learning falsification test before assigning numbers to artifact reliance.
7. Add a near-duplicate/content-leak check or stop claiming leakage is absent.
8. Benchmark at least one architecture alternative - `DeepLabV3+` is the obvious low-friction choice since `Docs8` already names it.
9. Clean the documentation lineage so `Audit7 Pro` is either the real upstream critique or the docs stop pretending it is.
10. Keep the final claim narrow and honest: classical tamper localization baseline, not a generally reliable tamper-forensics system.
