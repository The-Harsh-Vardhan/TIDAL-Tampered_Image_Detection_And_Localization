# 1. Overview of Improvement Strategy

This roadmap uses an ablation-first improvement philosophy for `vK.7.1 Image Detection and Localisation.ipynb`.

- One substantive change per experiment notebook
- Compare every experiment against the same baseline control run
- Merge only after measured benefit
- Preserve assignment compliance in every experiment

Operational rule:

- Measurement and artifact completeness are mandatory for every run. A notebook run is invalid unless the saved `.ipynb` preserves its metric outputs and final visual outputs.

The intended workflow is:

1. Execute `vK.7.1` unchanged once to establish the baseline control.
2. Create one 10.x notebook per independent improvement.
3. Compare every notebook using the frozen evaluation protocol.
4. Merge only successful changes.
5. Test combinations later in a dedicated merge notebook rather than combining them immediately.
