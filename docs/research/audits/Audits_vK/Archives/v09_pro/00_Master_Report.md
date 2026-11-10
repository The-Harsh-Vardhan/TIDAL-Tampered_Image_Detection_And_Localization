# ASSIGNMENT COMPLIANCE VERDICT

Partial

`Docs9` is a real improvement over `Docs8`. It finally corrects CASIA framing, explicitly approves a learned image-level detection head, treats Colab verification as a mandatory gate, and uses a decision log instead of just throwing every shiny idea into the backlog. Those are meaningful design upgrades.

It still does not earn a pass.

Why it is still only `Partial`:

1. `Docs9` keeps quietly promoting unexecuted v8 notebooks into "current status." Both `notebooks/v8-tampered-image-detection-localization-kaggle.ipynb` and `notebooks/v8-tampered-image-detection-localization-colab.ipynb` exist, but both currently have `0` executed cells and `0` output cells. That means Docs9 cannot honestly use them as empirically validated evidence.
2. The assignment requires a single Google Colab notebook that actually runs. Docs9 recognizes that, but recognition is not compliance. Colab readiness is still planned, not proven.
3. Docs9 fixes the detection story on paper by approving a learned classification head, but the assignment is judged on implemented behavior, not on good intentions.
4. The approved v9 scope is bloated: dual-task head, ELA, edge loss, pHash, Boundary F1, PR curves, three-seed validation, DeepLab comparison, augmentation ablation, and Colab verification. That is no longer a tight internship submission plan. It is a small research program with internship branding.
5. Several "approved" technical ideas are shakier than the docs admit. The pHash plan is not actually a near-duplicate check as written, the ELA integration sketch is technically messy, and the mask-randomization test is being oversold as a shortcut-learning verdict.

Bottom line: `Docs9` is a better design set than `Docs8`, but it is still not approval-ready. It has stronger instincts, cleaner framing, and better prioritization. It also still confuses planned fixes with grounded evidence and keeps trying to win the assignment by adding more machinery than the assignment actually needs.

Supporting audits:

- [01_Assignment_Alignment_and_Problem_Understanding_Audit.md](01_Assignment_Alignment_and_Problem_Understanding_Audit.md)
- [02_Dataset_and_Architecture_Decision_Audit.md](02_Dataset_and_Architecture_Decision_Audit.md)
- [03_Training_Evaluation_and_Feature_Creep_Audit.md](03_Training_Evaluation_and_Feature_Creep_Audit.md)
- [04_Engineering_Quality_and_Document_Credibility_Audit.md](04_Engineering_Quality_and_Document_Credibility_Audit.md)
- [05_Docs9_vs_Audit8_Pro_Closure_Map.md](05_Docs9_vs_Audit8_Pro_Closure_Map.md)

# PROJECT ROAST

### 1. Docs9 is acting like v8 already earned credibility when the repo says otherwise

**Roast**

This project keeps trying to launder unexecuted notebooks into "current status." It is the ML equivalent of printing a medal before running the race.

**Why this is technically a problem**

`Docs9/00_Project_Evolution_Summary.md` and `Docs9/01_Assignment_Alignment_Review.md` repeatedly talk about v8 as if it established a validated baseline. The repo evidence does not support that. Both v8 notebooks currently have zero executed cells and zero output cells. That means Docs9 is standing on a cardboard floor and calling it a foundation.

**What a senior ML engineer would expect instead**

One clean sentence: "The v8 notebooks exist as implementations, but they are not preserved as executed evidence in the repo, so all v8 claims remain provisional until rerun." That is the honest version.

### 2. Docs9 fixes assignment compliance in prose and then acts like that is half a pass

**Roast**

The document finally learned to say "single Colab notebook" without squirming. Great. The notebook still is not verified on Colab, which means the requirement is still unmet. Congratulations on discovering the problem statement after eight rounds.

**Why this is technically a problem**

`Assignment.md` requires a single Google Colab notebook. `Docs9/01_Assignment_Alignment_Review.md` correctly calls that non-negotiable, but the compliance logic still leans on planned verification instead of actual verification. The requirement is binary. Either the notebook runs end to end on Colab or it does not.

**What a senior ML engineer would expect instead**

No compliance optimism until the Colab artifact is executed and archived.

### 3. The v9 plan is suffering from feature creep dressed up as rigor

**Roast**

At some point this stopped being an internship assignment plan and became a hobbyist paper outline with a GPU budget problem.

**Why this is technically a problem**

Docs9 approves all of the following for v9: learned classification head, ELA channel, edge loss, pHash check, Boundary F1, PR curves, three-seed validation, DeepLabV3+ comparison, augmentation ablation, and Colab verification. That is too much change at once for a project that still has not locked down one empirically trusted baseline. The more moving parts you add, the easier it becomes to confuse debugging, ablation, and actual improvement.

**What a senior ML engineer would expect instead**

A narrower v9: learned detection head, one credibility fix for evaluation, one runtime validation pass, and maybe one optional comparison. Not a mini research agenda.

### 4. The pHash "near-duplicate" plan is mislabeled at best and wrong at worst

**Roast**

Docs9 says "near-duplicate check" and then shows code that groups identical hash strings. That is not near-duplicate detection. That is exact-match bucketization with nicer marketing.

**Why this is technically a problem**

`Docs9/06_Notebook_V9_Implementation_Plan.md` computes a pHash string and groups images by exact hash equality. Real near-duplicate detection needs a Hamming-distance threshold or a group-aware search, not just identical hashes. `Docs9/03_Feasible_Improvements.md` even claims O(n²) comparison while the sample code does not do any actual pairwise distance search. The design is internally inconsistent.

**What a senior ML engineer would expect instead**

Define the duplicate criterion explicitly, compare hashes with a distance threshold, and perform grouping before split assignment.

### 5. ELA is being sold harder than the evidence justifies

**Roast**

Docs9 talks about ELA like it is the magic forensic vitamin the project was missing. It is not. It is a cheap, sometimes useful cue that may help, may do nothing, or may overfit to dataset compression habits.

**Why this is technically a problem**

`Docs9/03_Feasible_Improvements.md` says ELA highlights "exactly the signal that copy-move boundaries produce." That is too strong. Copy-move from the same source image often preserves compression history better than splicing, so ELA is not some guaranteed copy-move savior. Worse, the implementation sketch in `Docs9/06_Notebook_V9_Implementation_Plan.md` treats the grayscale ELA map as an Albumentations `image` target, runs full image normalization machinery, then manually divides by `255` before concatenation. That is underspecified and likely wrong.

**What a senior ML engineer would expect instead**

Describe ELA as a lightweight hypothesis, isolate it in a clean ablation, and specify the channel normalization path precisely.

### 6. The architecture reasoning is still weak, just with more paperwork

**Roast**

Docs9 added a comparison experiment and decided that counts as architectural rigor. One one-seed DeepLabV3+ run is not architecture justification. It is a checkbox.

**Why this is technically a problem**

U-Net/ResNet34 is still fundamentally a convenience baseline. `Docs9` improves the story by approving a DeepLabV3+ comparison, but the same docs already admit that DeepLab is unlikely to solve the RGB limitation. So the project still has not established why this is the right forensic design; it has only established that it might benchmark one adjacent decoder.

**What a senior ML engineer would expect instead**

Either keep the claim narrow: "stable baseline, not optimized forensic architecture," or run a tighter comparison program with very limited scope and actual decision criteria.

### 7. Docs9 keeps forecasting metrics like it has a crystal ball

**Roast**

Nothing says "premature confidence" quite like printing target F1 ranges and AUC jumps before the design has survived its first honest run.

**Why this is technically a problem**

`Docs9/00_Project_Evolution_Summary.md`, `Docs9/04_Improvement_Decision_Log.md`, and `Docs9/08_Future_Research_Directions.md` all publish expected gains and confidence bands as if they were grounded estimates. They are not. They are wishcasting attached to an unexecuted baseline and several unvalidated changes.

**What a senior ML engineer would expect instead**

Success criteria are fine. Forecasted gains should be labeled as hypotheses, not treated like a quasi-quantitative performance model.

### 8. The mask-randomization test is still being oversold

**Roast**

Docs8 faked certainty with pseudo-percentages. Docs9 upgraded to a slightly better ritual and immediately started treating it like a decisive falsification test.

**Why this is technically a problem**

Shuffling masks and reevaluating is better than the old shortcut-learning theater, but it still does not prove the model learned genuine forensic cues. It can only show that predictions are not equally compatible with arbitrary masks. `Docs9/07_Risk_Assessment.md` calls it the single most important diagnostic. That is inflated confidence again.

**What a senior ML engineer would expect instead**

Treat mask randomization as one weak-to-moderate sanity check, not the centerpiece of shortcut-learning validation.

### 9. The docs keep calling hard things easy

**Roast**

If you have to write custom 4-channel pretrained-weight adaptation logic, dual-output losses, ELA-specific augmentation paths, and new evaluation branches, it is not "three lines." Stop pretending engineering labor disappears when written in markdown.

**Why this is technically a problem**

`Docs9/03_Feasible_Improvements.md` labels several items as easy that are only easy in abstract. The implementation plan itself immediately reveals the real complexity. That kind of underestimation is exactly how notebook projects slip from "one more feature" into brittle mess.

**What a senior ML engineer would expect instead**

Difficulty estimates that reflect the full change surface: model definition, data pipeline, normalization, checkpoint compatibility, training loop, evaluation, and debugging.

### 10. Colab feasibility is still a promise under stress, not a proven design constraint

**Roast**

Docs9 says it cares about Colab constraints and then approves a pipeline whose own risk section estimates roughly eleven hours of work on a T4 before the bugs even show up. That is not constraint-aware design. That is spreadsheet optimism.

**Why this is technically a problem**

`Docs9/07_Risk_Assessment.md` puts the v9 experiment set near Colab session limits. That is already too tight for a project with multi-seed runs, new dependencies, and architectural changes. If Colab feasibility matters, the design should optimize for getting one trustworthy submission artifact out, not for squeezing a research matrix into a session budget.

**What a senior ML engineer would expect instead**

Separate the submission path from the research path. One clean Colab submission run first. Extra seeds and comparison experiments only after that artifact is secured.

# TOP 10 DESIGN PROBLEMS

1. Docs9 quietly treats unexecuted v8 notebooks as empirically validated current state.
2. Assignment compliance is still discussed too generously relative to the unverified Colab deliverable.
3. The approved v9 scope is bloated far beyond what a focused internship submission needs.
4. The pHash "near-duplicate" design is technically underspecified and mislabeled.
5. The ELA integration plan is messy and likely incorrect in its current normalization/augmentation sketch.
6. ELA is oversold as a likely copy-move fix despite weak causal grounding.
7. DeepLabV3+ comparison is too small and too shallow to really justify architecture choice.
8. Docs9 publishes metric targets, confidence estimates, and time budgets as if they were evidence-backed.
9. The mask-randomization test is an improvement over Docs8, but still overclaimed as shortcut-learning proof.
10. Difficulty estimates are consistently too optimistic for the amount of code and debugging implied.

# TOP 5 STRENGTHS

1. Docs9 corrects the CASIA framing and finally treats it as a chosen baseline rather than an assignment mandate.
2. The decision log is explicit, structured, and much cleaner than the earlier "everything is future work" style.
3. Docs9 directly responds to Audit8 Pro instead of pretending the critique does not exist.
4. Approving a learned image-level classification head is the right architectural correction for assignment alignment.
5. Docs9 explicitly recognizes Colab verification as a mandatory submission gate instead of a nice-to-have.

# WHAT THE AUTHOR CLEARLY UNDERSTANDS

The author clearly understands the first-order weaknesses of the earlier pipeline: heuristic detection is weak, copy-move is hard, mixed-set segmentation metrics can mislead, and Colab verification matters because the assignment says it matters. Docs9 also shows better project triage than earlier docs. The approved/deferred/rejected structure is a real maturity upgrade.

The author also understands that architecture, loss design, data integrity, and evaluation methodology are connected. The docs are no longer talking about the model as if one trick solves everything. They recognize class imbalance, boundary quality, leakage risk, and forensic signal limitations as separate design concerns.

# WHAT THE AUTHOR LIKELY DOES NOT UNDERSTAND YET

The author still does not fully understand the difference between a cleaner plan and a trusted system. Docs9 is much better at framing the right problems, but it still gives itself too much credit for planned corrections. There is also still a tendency to mistake "more experiments" for "better design." That is a common trap: when confidence is low, people add ablations, extra metrics, and side studies instead of locking down one reliable deliverable.

There is also a gap in technical humility around implementation details. The docs repeatedly classify nontrivial changes as easy, forecast gains too confidently, and overstate what lightweight tests like mask randomization can prove. That is not ignorance. It is overconfidence in weak evidence.

# HOW TO IMPROVE DOCS9 BEFORE IMPLEMENTING NOTEBOOK V9

1. Rewrite every "current status" claim that depends on v8 so it distinguishes code existence from executed evidence.
2. Split v9 into `submission path` and `research path`.
   Submission path: learned detection head, one clean evaluation fix, verified Colab run.
   Research path: ELA, DeepLab comparison, multi-seed runs, augmentation ablation.
3. Fix the pHash design so it performs real near-duplicate detection with a stated distance threshold and grouping-before-splitting logic.
4. Rewrite the ELA plan with a precise normalization and augmentation contract. Do not treat grayscale ELA as a generic extra `image` target without specifying transform behavior.
5. Downgrade all expected metric gains and time estimates to hypotheses, not projected outcomes.
6. Narrow the architecture claim: U-Net/ResNet34 remains a baseline until proven otherwise.
7. Treat the mask-randomization test as a sanity check, not a shortcut-learning verdict.
8. Make Colab verification the first hard gate, not the last line item after a big experiment matrix.
9. Keep the final submission story narrow and honest: assignment-aligned tamper localization baseline with explicit limitations.
10. Stop trying to make Docs9 sound approval-ready before the notebook earns it.
