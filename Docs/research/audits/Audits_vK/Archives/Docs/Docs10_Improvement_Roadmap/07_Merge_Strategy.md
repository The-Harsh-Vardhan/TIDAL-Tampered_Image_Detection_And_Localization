# 7. Merge Strategy

Merge decisions must follow explicit rules.

## Rule 1: Performance-Oriented Changes

Performance changes merge only if:

- at least one primary localization metric improves by at least `+0.01` absolute on validation
- the same primary metric also improves by at least `+0.01` absolute on test
- image accuracy does not drop by more than `0.01` absolute
- ROC-AUC does not drop by more than `0.01` absolute

If validation improves but test does not, the change does not merge.

## Rule 2: Engineering or Evaluation Changes

Engineering or evaluation changes can merge if they measurably improve compliance, evaluation validity, reproducibility, or runtime efficiency and do not cause more than `0.005` absolute degradation in any primary metric.

## Rule 3: Split-Policy Changes

If the leakage-aware split from `10.3` is accepted:

- rerun the baseline control on the new split
- rerun all later performance experiments on that split
- compare only within that split regime

## Rule 4: Interaction Effects

If two winning changes appear complementary, do not merge them directly into the main notebook first. Create a later combined notebook, for example `vK.10.M1`, and test the interaction explicitly.

## Rule 5: Promotion to Mainline

The next main implementation should be created only after the merged candidate:

- passes the frozen evaluation protocol
- preserves all required visual outputs
- remains assignment-aligned
- retains a complete saved notebook artifact
