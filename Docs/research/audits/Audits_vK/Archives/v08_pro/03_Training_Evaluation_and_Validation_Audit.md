# Training, Evaluation, and Validation Audit

This is the part of `Docs8` that improves the most on self-awareness and still fails the most on completion. The docs finally identify the right failure modes. They just do not get to claim the repaired system yet.

## 1. Training strategy: the diagnosis is good, the fix is pending

`Docs8/04_Training_Strategy_Evolution.md` is one of the strongest documents in the set because it stops pretending the old training stack was fine.

It directly names the core issues:

- no `pos_weight`
- no scheduler
- batch-level Dice
- micro-batch BatchNorm instability
- minimal augmentation

Evidence: `Docs8/04_Training_Strategy_Evolution.md:28-39`.

That is far more honest than older versions.

The problem is that every meaningful correction is still in the "v8 Training Strategy" section, not in an executed result:

- scheduler addition (`Docs8/04_Training_Strategy_Evolution.md:93-103`)
- `pos_weight` (`Docs8/04_Training_Strategy_Evolution.md:105-115`)
- expanded augmentation (`Docs8/04_Training_Strategy_Evolution.md:117-134`)
- per-sample Dice (`Docs8/04_Training_Strategy_Evolution.md:138-151`)

That means the current project is still the broken training logic plus a better explanation of why it is broken.

## 2. The threshold story is still a red flag

`Docs8` does the right thing by centering the threshold anomaly:

- best threshold is 0.1327
- this indicates poor calibration
- the likely driver is unweighted BCE under severe background dominance

Evidence: `Docs8/00_Project_Evolution_Summary.md:27-29`, `Docs8/04_Training_Strategy_Evolution.md:68-74`, `Docs8/05_Evaluation_Methodology_Evolution.md:73-77`.

That diagnosis is credible.

What is not credible yet is any claim that calibration has been fixed. The document only forecasts a shift toward 0.30-0.50 after future v8 work (`Docs8/04_Training_Strategy_Evolution.md:115`, `Docs8/08_Notebook_V8_Implementation_Plan.md:42-44`, `Docs8/08_Notebook_V8_Implementation_Plan.md:257-258`).

## 3. The augmentation story is finally less naive

`Docs8` correctly recognizes that the old "protect forensic cues by avoiding strong augmentation" philosophy backfired (`Docs8/02_Dataset_Evolution.md:35-37`).

That is a real conceptual improvement over `Docs7`.

But again, the change is still prospective:

- `ColorJitter`
- `ImageCompression`
- `GaussNoise`
- `GaussianBlur`

are proposed, not part of the current executed evidence (`Docs8/02_Dataset_Evolution.md:70-78`, `Docs8/04_Training_Strategy_Evolution.md:117-134`, `Docs8/08_Notebook_V8_Implementation_Plan.md:93-113`).

So the correct summary is:

"The author learned why the old augmentation logic was weak."

Not:

"The project now has a robust augmentation pipeline."

## 4. Evaluation methodology: major honesty upgrade, still not a repaired stack

This is where `Docs8` clearly surpasses `Docs7`.

It explicitly states:

- mixed-set metrics are inflated
- tampered-only metrics should lead
- per-forgery reporting should be standard
- image-level detection remains heuristic

Evidence: `Docs8/05_Evaluation_Methodology_Evolution.md:54-93`.

That is good.

But the repaired evaluation logic is still mostly a specification:

- finer threshold sweep
- mask-size stratification
- Boundary F1
- separate image-level threshold calibration
- PR curves
- confidence-stratified analysis

Evidence: `Docs8/05_Evaluation_Methodology_Evolution.md:103-178`.

So the audit must separate:

- **acknowledged flaw**: yes
- **executed correction**: no

## 5. Metric inflation is now admitted, but not yet retired

`Docs8` is refreshingly blunt here:

- mixed-set Pixel-F1 = 0.7208
- tampered-only Pixel-F1 = 0.4101
- the gap is 0.3107

Evidence: `Docs8/05_Evaluation_Methodology_Evolution.md:66-71`.

That is the right framing.

The project still does not get to use this honesty as a replacement for corrected reporting in an actual v8 run. Until the new evaluation layout is executed, the project remains historically tied to an inflated primary metric story.

## 6. Validation experiments are still more plan than proof

`Docs8` improves on `Docs7` by moving some validation thinking into concrete monitoring and experiment plans. That is better than vague theory.

But the core validation gap remains:

- Boundary F1 is planned.
- Mask-size stratification is planned.
- Multi-seed evaluation is planned.
- Cross-dataset testing is planned.
- Shortcut-risk falsification is still mostly inferred from robustness gaps.

Evidence: `Docs8/05_Evaluation_Methodology_Evolution.md:138-178`, `Docs8/07_Shortcut_Learning_Risk_Assessment.md:124-148`, `Docs8/09_Future_Experiments.md:145-212`.

The result is the same old problem in improved packaging:

the validation story is smarter than the validation evidence.

## 7. Shortcut-learning analysis: plausible, not proven

This is the most analytically ambitious document in `Docs8`, and also the easiest one to overrate.

What it does well:

- identifies plausible shortcut channels
- connects robustness plateaus to artifact reliance
- ties copy-move weakness to the absence of cross-source statistical differences

Evidence: `Docs8/07_Shortcut_Learning_Risk_Assessment.md:22-73`.

What it overreaches on:

- estimating exact percentages of performance attributable to artifacts (`Docs8/07_Shortcut_Learning_Risk_Assessment.md:81-95`)
- treating nuisance-degradation gaps as if they isolate causal feature contributions

That is a useful hypothesis-generation exercise. It is not a rigorous validation result.

## 8. The implementation plan proves the work is not done yet

`Docs8/08_Notebook_V8_Implementation_Plan.md` is useful because it is concrete. It is also damning because it shows how much remains unfinished.

The pre-flight checklist and post-run checklist are explicit:

- `pos_weight` not yet added
- scheduler not yet added
- doc-code mismatch not yet reconciled
- per-sample Dice not yet implemented
- improved reporting not yet in place

Evidence: `Docs8/08_Notebook_V8_Implementation_Plan.md:211-263`.

That checklist is a confession that the current project state is still pre-fix.

## 9. What a senior reviewer would conclude

The author now understands the training and evaluation weaknesses much better than before. That is obvious.

The author still seems tempted to borrow maturity from the quality of the diagnosis. That is the trap. Good postmortem writing does not equal corrected methodology.

## 10. Bottom line

Training:

- diagnosis strong
- implementation pending

Evaluation:

- honesty improved substantially
- corrected pipeline still unexecuted

Validation:

- experiment roadmap decent
- evidence still too thin for stronger claims
