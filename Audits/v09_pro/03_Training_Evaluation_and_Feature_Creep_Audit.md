# Training, Evaluation, and Feature Creep Audit

## Bottom line

Docs9 is trying to do too many things at once. The training and evaluation ideas are individually plausible, but the combined v9 scope is overbuilt for a project that still has not secured one trusted baseline run.

## 1. The loss stack is becoming crowded without enough evidence

v8 already carried BCE plus Dice, `pos_weight`, per-sample Dice, scheduler, stronger augmentation, and richer evaluation. Docs9 wants to add:

- classification loss,
- edge loss,
- per-forgery-type tracking,
- Boundary F1,
- PR curves,
- mask-randomization testing,
- multi-seed validation,
- augmentation ablation.

That is a lot of change piled on top of a baseline that still lacks preserved execution evidence in the repo. This is how projects stop knowing which change actually helped.

## 2. Class imbalance reasoning is still not fully disciplined

Docs9 assumes v8 already repaired imbalance handling with `pos_weight` and per-sample Dice. That is probably directionally true in code, but the repo still does not preserve executed v8 evidence. So Docs9 is again building planning confidence on unverified prior state.

Worse, Docs9 never really revisits whether the `pos_weight` computation itself was methodologically clean enough. It treats the problem as solved because the feature exists, not because the evidence is settled.

## 3. Scheduler and training controls are treated as if settled

Docs9 keeps `ReduceLROnPlateau`, retains warmup options, and mostly freezes the training-control story. That is fine if the baseline is stable. The problem is that the baseline is not actually repo-validated. The docs are again assuming a level of stability that the stored artifact does not prove.

This is a recurring Docs9 habit:

- existence of code,
- upgraded to current status,
- current status upgraded to design premise.

That chain is too generous.

## 4. Evaluation discipline is better than before, but still overconfident

Docs9 does improve the evaluation plan:

- it retains tampered-only emphasis,
- it adds a learned image-level head,
- it approves Boundary F1 and PR curves,
- it keeps copy-move visible.

Those are real improvements.

The problem is the confidence level. The docs keep speaking as if this evaluation stack is now rigorous by construction. It is not. It is still a plan, and parts of it are shaky:

- mask randomization is not shortcut-learning proof,
- Boundary F1 is useful but not a credibility shield,
- one-seed architecture comparison does not validate the whole design,
- multi-seed runs help confidence but do not rescue weak reasoning.

## 5. The mask-randomization test is better, not decisive

Docs9 replaces Docs8's pseudo-quantitative shortcut claims with a shuffled-mask test. That is an improvement. It is not the slam-dunk the docs want it to be.

If F1 drops after mask shuffling, that tells you the predictions are not equally compatible with arbitrary masks. Fine. It does not tell you the model learned genuine forensic cues rather than image-content heuristics, boundary priors, or dataset artifacts that still correlate with the correct masks.

This should be one sanity check. Docs9 keeps trying to make it the shortcut-learning kingmaker.

## 6. Multi-seed and ablation plans are feature creep in audit clothing

Three-seed validation, DeepLab comparison, and augmentation ablation are all defensible experiments. Together, inside one v9 assignment plan, they are too much.

This is where Docs9 drifts away from the assignment. The assignment asked for:

- a model,
- a clear rationale,
- sound evaluation,
- runnable cloud delivery.

It did not ask for a compact research matrix. Docs9 is over-rotating from "we were too loose before" to "we must now validate everything at once."

That is not rigor. That is lack of prioritization.

## 7. Forecasted metrics are being used like evidence

Docs9 repeatedly publishes target ranges:

- F1,
- AUC,
- threshold bands,
- robustness gaps,
- time budgets,
- confidence levels.

Those are not findings. They are forecasts. The docs are cleaner than before, but they still keep slipping from "expected" to "implied confidence" in ways that make the plan sound more grounded than it is.

## Verdict

Training and evaluation in Docs9 are smarter than they were in Docs8. The main failure is scope discipline. The project is trying to improve the model, improve the evidence, improve the architecture story, improve the leakage story, and improve the runtime story all at once. That is too much for a clean v9 implementation pass.

Docs9 needs a narrower submission path and a clearly separate research path.
