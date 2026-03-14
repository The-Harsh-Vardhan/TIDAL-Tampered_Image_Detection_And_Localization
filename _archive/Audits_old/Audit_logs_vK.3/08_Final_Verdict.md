# 8. Final Verdict

Review sources:

- `Notebooks/vK.3_training_logs.txt`
- `Audits/Audit v7.1/Audit-vK.7.1.md`

Training quality is mixed. There is evidence that the classifier branch learns useful signal, especially in the later run. There is not enough trustworthy evidence to say the localization branch is performing well.

Metric trustworthiness is low. The first run is effectively invalid as a benchmark, and the second run still contains segmentation behavior that is too suspicious to accept without metric verification.

Experiment hygiene is weak. The duplicated legacy path, split confusion, offline-tracking fallback, deprecated augmentation warning, and truncated later run all point to a workflow that was not under tight control.

Readiness for submission claims is poor. These logs do not justify strong claims about pixel-level tamper localization. At best, they support a partially promising detector with an unproven, and possibly mismeasured, localization pipeline.
