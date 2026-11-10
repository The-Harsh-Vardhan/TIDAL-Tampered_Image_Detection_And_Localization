# 8. Notebook Engineering Quality

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

From a readability standpoint, the notebook is much improved. The markdown structure is clear, the sections are well labeled, and the flow is easier to audit than a monolithic notebook. That is a real strength.

From an engineering standpoint, it is still inconsistent.

The biggest issue is duplication. The notebook contains both a source-preserved earlier experiment block and a second "effective submission" training pipeline (cells 29-45 and 51-75). That duplication makes it harder to know which code path should be trusted as the real experiment.

The second issue is state incoherence. The main effective training/evaluation cells are unexecuted (cells 71, 73, 75), but downstream model-loading and visualization cells are executed (cells 77-87). That is not a clean top-to-bottom artifact. It is a stateful notebook with broken provenance.

There are also smaller engineering rough edges:

- hardcoded `/kaggle/working/...` paths remain in the effective training block (cell 55)
- W&B integration is useful but brittle and secrets-dependent (cell 50)
- notebook claims in markdown are stronger than what execution state supports (cells 3, 91)

So the notebook is polished in presentation, but not disciplined in execution hygiene.
