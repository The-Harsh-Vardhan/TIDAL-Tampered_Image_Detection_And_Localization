# Data Pipeline Review

## What the pipeline gets right

Cells 8 to 13 at least attempt real dataset discovery and validation instead of hardcoding a CSV and praying. That is a good start.

Useful pieces:

- cell 10 validates image readability
- cell 10 checks image-mask dimension matches for tampered pairs
- cell 12 stratifies splits by `forgery_type`
- cell 13 saves a split manifest
- cell 17 seeds workers and uses persistent workers

That is the good news. Now for the part that actually matters.

## The authentic-mask handling is sloppy

Cell 9 output says the dataset has:

- `IMAGE/Au: 7491 files`
- `MASK/Au: 7491 files`

Then cell 10 completely ignores `MASK/Au` and cell 16 fabricates all-zero masks whenever `mask_path is None`.

That is a huge assumption. Maybe those authentic masks are blank. Maybe they are not. The notebook never checks. If the dataset already provides authentic masks, a serious pipeline validates them. It does not throw them away and substitute invented zeros because that was convenient.

Senior expectation: verify that `MASK/Au` is actually blank and dimension-aligned, then either use it or explicitly document why it is safe to replace.

## Image-mask pairing is acceptable, but brittle

Cell 10 pairs tampered images to masks mostly by filename match, then falls back to alternate extensions. That is decent as a practical heuristic, but it still assumes the dataset naming convention is clean and consistent.

Problems:

- no checksum or metadata-based verification
- no audit of duplicated stems across extensions
- no audit of suspicious multiple-mask candidates

For an internship notebook this is survivable. It is not robust engineering.

## Mask binarization is blind faith

Cell 16 does:

`mask = (mask > 0).astype(np.uint8)`

That assumes every nonzero pixel should become foreground. If the masks are perfectly binary already, fine. If they contain soft edges, labels, compression noise, or annotation artifacts, you just converted them into junk without inspection.

Senior expectation: inspect raw mask value histograms once, confirm mask conventions, and document the assumption instead of silently binarizing everything.

## The resize policy distorts geometry

Cell 11 sample outputs show raw sample shapes like `img=(256, 384, 3)`. Cell 15 then resizes everything to `384x384`.

That is aspect-ratio distortion. For segmentation, especially forgery localization with small copy-move regions, this matters. You are stretching both image content and mask shape into a square whether they like it or not.

Senior expectation: preserve aspect ratio with padding or scale-shorter-side logic, or at least justify why square resizing does not destroy the target signal.

## The leakage check is fake rigor

Cell 12 prints `No data leakage detected.` That is technically true only in the tiny sense that file paths do not overlap across splits.

What it does not rule out:

- near-duplicate images
- sibling authentic/tampered variants sharing scene content
- derivative manipulations of the same source image
- content leakage via naming conventions or source families

Calling that "no data leakage" is too strong. It should say "no path overlap detected."

## The sample validation is shallow

Cell 11 calls the section `DATASET VALIDATION SUMMARY`, then the actual sample inspection is three samples. All three shown in the output are splicing. That is not validation. That is peeking at three files and declaring victory.

Senior expectation: inspect authentic, copy-move, and splicing examples separately, and explicitly validate the edge cases you are most likely to mishandle.

## Split strategy is reasonable, but incomplete

The 70/15/15 stratified split by forgery type in cell 12 is fine as a baseline. It keeps class proportions stable across authentic, copy-move, and splicing.

It still misses important structure:

- no stratification by mask size, despite mask size turning out to dominate failure in cell 33
- no grouping by source image family
- no duplicate or near-duplicate blocking

Given the later failure on tiny masks and copy-move, the split logic should have been more aware of what actually makes the task hard.

## Reliability verdict

Usable, but not fully trustworthy.

The pipeline is good enough to produce a real training run, but it still has too many unvalidated assumptions:

- authentic masks are ignored even though they exist
- mask binarization is undocumented
- aspect ratio is destroyed
- leakage control is shallow

This is not a broken pipeline. It is a pipeline that thinks basic hygiene checks count as a real audit.
