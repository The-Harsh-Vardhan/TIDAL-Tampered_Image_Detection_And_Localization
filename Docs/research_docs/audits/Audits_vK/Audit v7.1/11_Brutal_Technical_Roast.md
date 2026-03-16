# 11. Brutal Technical Roast

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

This notebook is the ML equivalent of dressing a baseline up in a blazer and hoping nobody notices the missing results.

You clearly understand the assignment. You implemented the dual-task model. You wrote the metrics. You even wrote the exact four-panel submission visualization the reviewer wants to see. Then you stopped one step short of proving anything in the saved artifact.

That is the frustrating part. The technical core is not embarrassing. The experimental discipline is.

You left the notebook in a state where the markdown says "fulfilled," but the execution trail says "trust me." The effective training block is unexecuted. The final test metrics are not shown. The final assignment-style panel is not rendered. Yet the notebook still talks like the case is closed.

That is not how a strong submission works. A strong submission makes the reviewer's job easy. This one makes the reviewer reconstruct your intent from unexecuted code and partially executed downstream cells.

The model choice is also safe to the point of being timid. Plain U-Net, plain classifier head, plain thresholded metrics, no serious forensic reasoning, no leakage control, no tampered-only localization analysis, no calibration view, no confusion matrix. This is a baseline, not a compelling forensic study.

As an internship submission, I would call it competent but not trustworthy. The code suggests you can build. The artifact suggests you do not yet think like a reviewer.
