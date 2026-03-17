# 01 - Project Roast

You asked for a technical interview-grade roast, so here it is.

## 1) You submitted an unexecuted notebook and called it a result

### Roast
This is the ML equivalent of turning in a gym plan and claiming a six-pack.

### Why it is a problem
No training logs, no final metric outputs, no rendered qualitative panels, no executed artifacts in notebook output cells.

### Senior expectation
Executed notebook with visible losses, validation trajectory, test metrics, and prediction visuals embedded in output cells.

## 2) You over-engineered before proving baseline reliability

### Roast
You stacked AMP, gradient accumulation, focal classification, BCE+Dice, strong augmentation, robustness suite, Grad-CAM, and shortcut tests before proving one stable baseline run.

### Why it is a problem
When performance degrades, root cause analysis becomes guesswork because too many knobs changed simultaneously.

### Senior expectation
Start with a minimal baseline, prove it runs end-to-end, then add one improvement at a time with measurable deltas.

## 3) You promised evidence but delivered only code pathways

### Roast
The notebook is full of "will save" and "will log" logic, but no visible outputs. That is theater, not evaluation.

### Why it is a problem
Reviewers cannot verify any claim without rerunning everything, which defeats the submission objective.

### Senior expectation
Submission artifact must contain preserved outputs so reviewers can validate quality instantly.

## 4) Assignment asks for reasoned architecture decisions; you gave narrative, not decisions

### Roast
"We used this architecture" is not reasoning. It is a declaration.

### Why it is a problem
No comparison against alternatives, no tradeoff analysis, no ablation proving the classification head helps.

### Senior expectation
At least one concise comparison table and one controlled ablation.

## 5) Data integrity checks are incomplete

### Roast
You wrote helper utilities, then skipped strict enforcement where it matters.

### Why it is a problem
Silent mask issues and potential split leakage can poison training and fake confidence.

### Senior expectation
Fail-fast assertions for mask presence (tampered class), split disjointness checks, and integrity summary report.

## 6) Visualization requirement is only half-met

### Roast
You wrote plotting code, but never showed real outputs in the submitted artifact.

### Why it is a problem
Segmentation credibility depends on visual sanity checks. Without them, metrics can hide catastrophic failure modes.

### Senior expectation
Show real original/GT/pred/overlay panels for best, median, and worst cases.

## Final Roast Verdict

This notebook looks like someone who knows what a good pipeline should contain but didn’t deliver proof that this specific pipeline actually worked.
