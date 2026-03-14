# 6. Brutal Roast

Review sources:

- `Notebooks/vK.3_training_logs.txt`
- `Audits/Audit v7.1/Audit-vK.7.1.md`

Here is the blunt version.

This log is what happens when a project starts acting impressed with its classifier before it has earned trust on localization. The accuracy curve is climbing, everyone wants to celebrate, and the mask metrics are quietly screaming that something is off.

The flat `0.5949` segmentation story is not "stable." It is suspicious. It looks like a broken thermometer being presented as climate science. If I saw this in an internship submission, my first assumption would not be "good localization." My first assumption would be "your metric pipeline is lying to you."

The first run is worse than weak. It is structurally compromised. If the model is training on `1893` samples when the prepared training split was `8829`, and the legacy block is miswiring train/test CSVs, then that run is not a result. It is a cautionary tale about why duplicated pipelines are dangerous.

Then the second run shows up, finally uses the right training size, pushes validation accuracy much higher, and still fails to make the localization story convincing. And even that run does not finish cleanly. No final test summary. No clean closeout. Just a training trace that stops midstream and leaves the reviewer to guess what happened next.

That is not research discipline. That is experiment sprawl.
