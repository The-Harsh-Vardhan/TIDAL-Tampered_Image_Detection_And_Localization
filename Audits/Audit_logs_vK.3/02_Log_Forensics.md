# 2. Log Forensics

Review sources:

- `Notebooks/vK.3_training_logs.txt`
- `Audits/Audit v7.1/Audit-vK.7.1.md`

The log file has `529` lines and contains multiple experiment fragments rather than one clean, linear training story.

Near lines `24-108`, the data preparation section reports:

- dataset size: `12614`
- class counts: `7491` authentic, `5123` tampered
- split sizes: `8829 / 1892 / 1893`
- no missing masks in the discovered metadata preview

That looks reasonable at face value, but the first actual training run near lines `137-208` does not use those full split sizes. Instead, it reports `Train: 1893 Val: 1892 Test: 1892`, which is already a red flag because the train split has collapsed to the size of the nominal test split.

Cross-checking the project's later notebook structure explains why this matters. The earlier source-preserved experiment path is duplicated, and its old CSV wiring points `TRAIN_CSV` to `test_metadata.csv` and `TEST_CSV` to `val_metadata.csv`. The prior audit warned that the duplicated experiment block made it unclear which path was authoritative; the log now shows the practical consequence of that ambiguity. The first run is very likely not training on the intended full training split at all.

Near lines `209-211`, the log records that W&B fell back to offline mode because no API key was configured. That is not the main problem, but it is one more sign that experiment tracking was not tightly controlled.

Near lines `413-493`, the file shifts into a later run with richer metrics and the correct-looking full training size (`Train: 8829 Val: 1892 Test: 1893`). This is clearly the more serious run. It also emits `JpegCompression` deprecation warnings earlier in the file, which tells us the augmentation stack was not fully cleaned up.

The later run, however, is truncated. The file stops at epoch 32 without a `Training finished` line, without a final test summary, and without a closure artifact that would let a reviewer treat it as a complete experiment.
