# Docs9 vs Audit8 Pro Closure Map

This map answers the only question that matters:

Did `Docs9` actually close the major issues raised in `Audit8 Pro`, or did it mostly become better at describing them?

## Closure Table

| Audit8 Pro criticism | Docs9 response | Status | Audit note |
|---|---|---|---|
| Colab deliverable was still not clean or verified | `Docs9/01_Assignment_Alignment_Review.md` and `Docs9/02_Audit8_Pro_Response.md` now treat Colab verification as mandatory | Partially Resolved | Better prioritization, still not executed |
| CASIA was being framed too aggressively as assignment-mandated | Docs9 explicitly reframes CASIA as a chosen baseline | Resolved | This is one of the cleanest Docs9 fixes |
| Detection was still a `max(prob_map)` hack | Docs9 approves a learned classification head and dual-task model | Partially Resolved | Correct fix on paper, still not implemented evidence |
| Copy-move remained weak and under-addressed | Docs9 keeps it visible and proposes ELA + edge loss + per-type tracking | Partially Resolved | Better honesty, but still speculative and ELA is oversold |
| U-Net/ResNet34 was a convenience baseline, not a strong forensic design | Docs9 approves a DeepLabV3+ comparison and keeps stronger architectures deferred | Partially Resolved | Better than pure narrative, still shallow evidence |
| Docs8 relied on planned v8 repairs instead of executed corrections | Docs9 now talks as if v8 exists as a usable baseline | Now Worse | The docs are cleaner, but they now over-credit an unexecuted v8 state |
| Evaluation improvements were specified more than executed | Docs9 adds Boundary F1, PR curves, and learned detection to the plan | Partially Resolved | Evaluation design is stronger, execution gap remains |
| Shortcut-learning claims were stronger than the evidence | Docs9 replaces pseudo-percentages with a mask-randomization test | Partially Resolved | Better than before, but still overclaimed as stronger proof than it is |
| Leakage checking was weak | Docs9 approves pHash checking | Partially Resolved | Right direction, flawed design sketch |
| Documentation lineage was messy | Docs9 uses a much cleaner internal lineage centered on Assignment -> Docs8 -> Audit8 Pro -> Docs9 | Resolved | This is materially better |

## What Docs9 genuinely fixed

1. **CASIA framing is corrected.**
   Docs9 finally stops pretending the assignment mandated CASIA. That is a real improvement in intellectual honesty.

2. **Audit8 Pro is treated as a real input, not as an inconvenience.**
   `Docs9/02_Audit8_Pro_Response.md` is a serious response document, not a defensive shrug.

3. **Detection alignment is corrected at the design level.**
   Approving a learned image-level head is the right architectural answer to the assignment wording.

4. **Colab is elevated to a hard gate.**
   Docs9 is much more disciplined about making Colab verification non-optional.

5. **The decision process is cleaner.**
   Approved / Deferred / Rejected is a strong upgrade over vague future-work dumping.

## What Docs9 only partially fixed

1. **Architecture justification.**
   DeepLabV3+ comparison helps. One comparison experiment does not fully solve the reasoning gap.

2. **Copy-move strategy.**
   Better diagnosis and visibility, but still no strongly grounded solution.

3. **Evaluation credibility.**
   Better metrics are planned, but the docs still lean on unexecuted baseline assumptions.

4. **Leakage control.**
   pHash is the right instinct, but the proposed implementation is not rigorous enough as written.

5. **Shortcut-learning validation.**
   The test is better than Docs8's pseudo-quantification, but still not decisive.

## What is still open

1. The single Colab notebook deliverable is still unverified.
2. The repo still does not preserve executed v8 evidence.
3. The project still does not have a tightly scoped submission path separate from the research path.
4. Architecture justification remains baseline-level, not strongly evidence-backed.
5. Colab feasibility under the full v9 experiment slate is still optimistic rather than demonstrated.

## What got worse

The main regression is subtle but important:

Docs8 mostly spoke in future tense about v8. Docs9 now talks like v8 is a stable, validated current baseline even though the repo-preserved notebooks still show zero executed cells and zero outputs. That is a more polished form of overclaiming, not a smaller one.

## Bottom line

Docs9 is a real improvement over Docs8. It fixes several honesty and prioritization problems and responds to Audit8 Pro far more directly. It still does not fully close the core trust gap because it keeps treating planned or unpreserved behavior as stronger evidence than it is.

The correct summary is:

- better design discipline,
- better framing,
- better prioritization,
- still too much confidence,
- still not ready for approval.
