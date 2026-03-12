# Assignment Alignment and Problem Understanding Audit

## Bottom line

Docs9 is substantially more aligned with the assignment than Docs8. It finally stops pretending heuristic detection is acceptable, it corrects CASIA framing, and it treats Colab verification as mandatory. That is real progress.

It still is not a pass-worthy design set because it keeps leaning on planned fixes and unexecuted v8 artifacts as if they already reduced risk.

## 1. Detection plus localization: the design finally says the right thing

This is the biggest improvement in Docs9. `Docs9/01_Assignment_Alignment_Review.md` explicitly admits that the assignment requires both image-level detection and pixel-level localization, and `Docs9/02_Audit8_Pro_Response.md` approves a learned classification head to replace `max(prob_map)`.

That is correct. It is also still only a design correction. The repo evidence shows both v8 notebooks exist but currently contain zero executed cells and zero output cells. So when Docs9 talks about the project's "current status," it is still speaking more confidently than the evidence allows.

The right audit stance is:

- Docs9 fixes the reasoning gap on paper.
- Docs9 does not yet fix assignment compliance in deliverable form.

## 2. Problem framing is much better, but still occasionally too clean

Docs9 shows a better understanding of the actual task than earlier iterations:

- splicing and copy-move are treated as different problems,
- metric inflation is acknowledged,
- dataset limitations are no longer hidden,
- the system is framed more like an analyst-assistance baseline than a magic detector.

That said, the docs still smooth over too much. They speak about the project as if it already has a stable, trusted v8 baseline and now just needs incremental upgrades. That is not true in the repo-backed sense. There is implemented code, yes. There is not preserved empirical proof.

This matters because design reviews are supposed to discipline ambition. Instead, Docs9 is still allowing itself to inherit confidence from an unexecuted baseline.

## 3. Real-world implications are acknowledged, but only partially internalized

Docs9 understands that CASIA is limited and that copy-move is a particularly hard form of subtle tampering. It also avoids the worst earlier sin of pretending the chosen dataset was assignment-mandated. Good.

What it still does not fully internalize is the consequence of those limitations:

- if the dataset is narrow and artifact-prone,
- and the baseline has not been revalidated in executed form,
- then the right next step is not a broad upgrade matrix.

The right next step is one trusted submission path.

Docs9 still behaves like the project has earned the right to branch into research-style improvement tracks. It has not.

## 4. The assignment alignment language is stronger than the actual design state

`Docs9/01_Assignment_Alignment_Review.md` gives status labels like "Met as of v8" for augmentation and visual results. That is too generous when v8 execution is not preserved in the repo. The augmentation code exists. The visual-result sections exist. That is not the same as a verified submission artifact.

This is a subtle credibility bug:

- the author no longer lies aggressively,
- but they still round provisional design state upward into "met."

For a principal review, that is still a problem.

## 5. Where Docs9 genuinely improves understanding

These improvements are real:

1. It correctly reframes CASIA as a chosen dataset.
2. It treats dual-task design as the correct response to the assignment wording.
3. It recognizes that rigorous evaluation means more than one flattering metric.
4. It explicitly keeps Colab deliverability in scope.
5. It acknowledges copy-move as the weak point instead of burying it.

Those are meaningful signs that the author is learning.

## Verdict

Docs9 demonstrates much better understanding of the assignment problem than Docs8 did. It now understands what the assignment is actually asking for. The remaining issue is not basic misunderstanding. It is failure to enforce discipline: the docs still reward themselves too early for planned fixes and keep inflating "better design" into "nearly compliant system."
