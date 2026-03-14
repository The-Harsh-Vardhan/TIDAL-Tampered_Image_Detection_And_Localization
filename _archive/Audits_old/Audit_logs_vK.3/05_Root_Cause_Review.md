# 5. Root-Cause Review

Review sources:

- `Notebooks/vK.3_training_logs.txt`
- `Audits/Audit v7.1/Audit-vK.7.1.md`

Several likely causes are supported by the evidence.

The first is evaluation inflation or distortion from empty-mask samples. If authentic images with all-zero masks are included in the segmentation metrics, a model that predicts low-activation or empty masks too often can still look better than it deserves. The prior audit flagged this, and the training curves are consistent with that problem.

The second is metric computation weakness. The early second-run behavior where Dice, IoU, and F1 all sit at `0.5949` is suspicious. Dice and F1 matching is not inherently wrong in binary segmentation, but Dice and IoU tracking identically at the same value over many epochs is a serious warning sign. At minimum, the metric path needs validation on a controlled batch.

The third is task imbalance. The classifier head appears to be winning the representation battle inside the shared backbone. The model is learning to answer "is this image tampered?" more effectively than "where exactly is the tampered region?"

The fourth is checkpoint policy. Even if the segmentation branch had moments of relative strength, the training loop is not selecting for them. It is explicitly rewarding image-level accuracy instead.

The fifth is run management. The first run appears polluted by legacy split wiring, and the second run is incomplete. That means the project is not just suffering from a modeling problem. It is suffering from experiment-control problems.
