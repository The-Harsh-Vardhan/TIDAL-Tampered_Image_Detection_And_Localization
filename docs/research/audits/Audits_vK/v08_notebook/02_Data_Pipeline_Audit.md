# 02 - Data Pipeline Audit

## Bottom line

The data pipeline is not obviously broken, but it is much weaker than the notebook wants you to believe. The code checks just enough to look careful while skipping the validations that actually matter for a forensic segmentation task.

## 1. Dataset discovery is brittle

Cell 9 walks all of `/kaggle/input` and grabs the first directory tree containing folders named `Image` and `Mask`. That is lazy. If multiple datasets are attached, the notebook can lock onto the wrong one with zero warning. A serious pipeline pins a dataset root explicitly or validates the dataset identity beyond two generic folder names.

This is the first smell of the whole notebook: it wants convenience now and correctness later.

## 2. Image-mask pairing is filename-based guesswork

Cell 10 pairs tampered images to masks by:

- same filename first,
- then same stem with a short list of alternate extensions.

That is not robust pairing logic. It assumes the dataset naming scheme is trivial and never checks whether the mask content actually corresponds to the image content. `validate_dimensions()` only checks width and height. Matching shape is not the same thing as matching annotation.

If a candidate submitted this in an interview and called it "validated pairing," I would push back immediately. This is light hygiene, not rigorous pairing.

## 3. Authentic masks are ignored instead of verified

Cell 8 explicitly says the dataset structure includes `Mask/Au/`. Cell 10 completely ignores those masks for authentic images and manufactures zero masks instead. That might be fine if the authentic masks are all-empty and perfectly aligned, but the notebook never proves that.

This is exactly the kind of silent assumption that burns people in forensics pipelines. If the dataset already ships masks, validate them. Do not pretend they do not exist because zero-filling is easier.

## 4. Mask binarization is blunt and unverified

Cell 16 loads grayscale masks and converts them with `(mask > 0).astype(np.uint8)`. That is acceptable only if the dataset is truly binary and any nonzero pixel means tampered region membership. The notebook never checks that assumption. It does not inspect unique mask values, it does not confirm annotation conventions, and it does not document whether soft edges or multivalue labels exist.

This is not necessarily wrong. It is just unsupported. Unsupported assumptions are how evaluation pipelines lie to you.

## 5. Fixed square resizing can deform tampered regions

Cell 15 resizes every image and mask to `384 x 384`. That is operationally convenient and scientifically dangerous. Thin or tiny tampered regions can be warped by forced aspect-ratio changes, especially in a task where boundary accuracy matters. The notebook never quantifies how much mask area distribution shifts after resizing, and it never checks whether the smallest masks become effectively unlearnable.

For a localization assignment, blindly squashing everything into a square without analyzing the consequences is sloppy.

## 6. Split logic is shallow

Cell 12 stratifies on `forgery_type` only: `authentic`, `splicing`, or `copy-move`. That is better than no stratification, but it still ignores:

- mask size,
- source-image identity,
- near-duplicate derivatives,
- compression family,
- difficulty distribution.

This matters because forensic datasets often contain related images or multiple manipulations derived from the same base asset. If those leak across splits, path-level disjointness means nothing.

## 7. Leakage checks are fake comfort

Cell 12 checks that file paths do not overlap across train, validation, and test. Fine. That proves only that exact same filenames were not reused. It does not prove there are no duplicates, near-duplicates, or sibling derivatives across splits.

Calling that "No data leakage detected" is embarrassing. It should say "No exact path overlap detected." Anything stronger is overclaiming.

## 8. Validation is tiny and mostly cosmetic

Cell 11 performs a "sample load check" on the first three pairs. Three. In a dataset pipeline with authentic and tampered images, multiple file extensions, and inferred forgery types. That is not validation. That is a warm-up print statement.

The notebook also infers forgery type from filename patterns in Cell 10. If the filename convention changes, the labels silently degrade to `unknown`. Again: convenient, brittle, and under-validated.

## 9. One real positive

The split manifest in Cell 13 is a good instinct. Saving the train, validation, and test membership is exactly what a reproducible project should do. The problem is that the upstream split quality is still too weak for that manifest to mean much.

## Verdict

The data pipeline is adequate as a rough baseline loader. It is not rigorous enough for a high-confidence tamper-localization submission. The biggest issues are brittle dataset discovery, assumption-heavy pairing, ignored authentic masks, fixed-size warping without analysis, and leakage checks that stop at path overlap.

This is the kind of pipeline that can limp through training and still leave you unable to trust the results.
