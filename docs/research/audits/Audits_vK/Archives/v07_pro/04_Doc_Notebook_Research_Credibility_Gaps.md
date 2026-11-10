# Documentation, Notebook, and Research Credibility Gaps

This note is about trust. A technically modest project can still be respectable if the docs are honest. A baseline becomes hard to trust when the written story and the code keep drifting apart.

## 1. Docs7 still contains materially wrong claims

### CASIA claim

- `Assignment.md:14-17`: CASIA is one example dataset.
- `Docs7/02_Dataset_and_Preprocessing.md:10`: "The assignment explicitly targets this dataset."

That is false. Full stop.

### Split strategy claim

- `Docs7/02_Dataset_and_Preprocessing.md:92-99`: stratified split on binary authentic vs tampered
- `notebooks/tamper_detection_v6.5_kaggle.ipynb:490-500`: stratified on `forgery_type`

Again, not cosmetic.

### Image-level score claim

- `Docs7/03_Model_Architecture.md:88-101`: top-k mean
- `Docs7/05_Evaluation_Methodology.md:25`: top-k mean
- `notebooks/tamper_detection_v6.5_kaggle.ipynb:1236`: `max(prob_map)`

That is a behavioral contradiction.

### Threshold-aware training claim

- `Docs7/12_Complete_Notebook_Structure.md:68`: early stopping uses threshold-aware validation from sweep
- `Docs7/04_Training_Strategy.md:186`: training validation uses fixed 0.5
- `notebooks/tamper_detection_v6.5_kaggle.ipynb:966-1008`, `1068-1075`: notebook agrees with fixed 0.5 during training

So the structure doc is wrong.

## 2. Prior audits already warned about this kind of drift

The repo contains its own cautionary tale.

### Audit 5

`Audit 5/00_Master_Report.md:7-9` scored the v5 state at `8.8/10` and explicitly praised alignment around top-k image scoring.

### Audit6

`Audit6/00_Master_Report.md:27-30` then documented that Docs6 drifted away from the v6 notebooks and that the image-level scoring rule no longer matched reality.

### Silent-failure audit

`Audit6/07_v6_5_Silent_Failure_Audit.md:30-35` went further and called out:

- mixed-set metric inflation
- empty-mask recall weirdness
- `max(prob_map)` fragility
- doc-code mismatch

### What this means

The project did not just have one typo. It repeatedly changed core behavior while keeping old narrative scaffolding in place. That is a credibility pattern.

## 3. Research alignment is half real, half cosplay

The project absolutely did read relevant papers. That part is obvious.

The problem is how those papers are used.

### What the repo genuinely understands

`Docs7/11_Research_Alignment.md` and `Research_Paper_Analysis_Report.md` correctly identify strong modern directions:

- edge-aware localization
- multi-stream fusion
- transformer or hybrid global context
- SRM/noise residual features
- dual-task classification + localization

### What the repo actually implements

v6.5 is still:

- RGB-only
- single-stream
- U-Net + ResNet34
- BCE + Dice
- heuristic image-level detection
- no edge supervision
- no dual-task head
- no frequency/noise branch

So the research alignment story often reads like:

"Here is why stronger systems matter. Anyway, we did not build any of that."

That is not fatal, but it needs a much humbler tone than the docs currently use.

## 4. The engineering story is oversold

Flags and helper functions do not equal engineering maturity.

Yes, the repo has:

- `CONFIG`
- optional AMP
- optional DataParallel
- checkpoint portability logic
- W&B guards

That is decent notebook craftsmanship. It is not the same thing as:

- reliable packaging
- tested metrics code
- clean repo source of truth
- clear release artifact
- robust submission hygiene

The presence of multiple generations of notebooks and docs actively undercuts any claim of mature engineering discipline.

## 5. Interview risk questions the author should expect

These are the questions that will expose whether the author actually understands the project or just assembled a polished baseline:

1. Why does `Docs7` say top-k mean while v6.5 code uses `max(prob_map)`?
2. Why did you claim the assignment explicitly targeted CASIA when it did not?
3. Why does the dataset doc say binary-label stratification while the code stratifies on `forgery_type`?
4. Why does the structure doc claim threshold-aware early stopping when the notebook validates at fixed 0.5 during training?
5. How much do empty authentic images inflate your mixed-set Pixel-F1?
6. Why is recall `1.0` when ground truth is empty but you predicted tampering?
7. What evidence do you have that file-path disjointness prevented leakage in CASIA?
8. Why is RGB-only ResNet34 a good forensic feature extractor?
9. Why did you not compare against DeepLabV3+ or SegFormer if architecture choice is part of the assignment?
10. Which of the validation experiments were actually run versus only documented?
11. Why should anyone trust Grad-CAM on `output.mean()` as more than a rough visual sanity check?
12. If this is a single-notebook assignment submission, why does the repo contain so many competing notebook variants?

## 6. What would restore credibility fastest

1. Freeze one final notebook and one final doc set.
2. Remove or correct every sentence that contradicts actual code behavior.
3. Separate "implemented" from "proposed" with zero ambiguity.
4. Reframe research alignment as "literature-informed baseline," not "frontier-aware system design."
5. Add an explicit known-risks section that names the strongest remaining evaluation and leakage problems.

## Bottom line

The project is not nonsense. The credibility problem is that the code and the narrative are no longer tightly coupled. Once that happens, every metric, every justification, and every "trust me, we thought about this" paragraph loses weight.

For an internship assignment, that is fixable.

For a principal-level technical review, it is a red flag until cleaned up.
