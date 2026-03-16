# Dataset and Architecture Decision Audit

## Bottom line

Docs9 improves the honesty of the dataset story and slightly improves the architecture story. It does not fully solve either. CASIA remains a narrow, synthetic, artifact-friendly benchmark, and U-Net/ResNet34 remains a convenience baseline that is only weakly justified even after the planned DeepLab comparison.

## 1. CASIA framing is finally corrected

This is one of the cleanest improvements in Docs9. `Docs9/01_Assignment_Alignment_Review.md` stops calling CASIA the expected dataset and explicitly labels it as a chosen baseline. That is the right correction.

It matters because the assignment says "examples include" CASIA, not "use CASIA or else." Docs9 gets that now.

## 2. CASIA limitations are acknowledged, but not solved

Docs9 knows the major CASIA problems:

- synthetic tampering artifacts,
- narrow benchmark behavior,
- possible duplicate or derivative leakage,
- weak transfer to real-world manipulations.

That is good. The problem is what comes next. The design still acts like a pHash pass and some stronger wording are enough to convert a brittle benchmark into a reliable basis for architectural claims. They are not.

If the dataset is narrow and artifact-rich, then any input feature that exploits compression or annotation habits must be treated with suspicion. Which brings us to ELA.

## 3. ELA is plausible, but Docs9 is overselling it

Docs9 approves ELA as a fourth channel because it is cheap, forensic-flavored, and easy to explain. Fine. The problem is the reasoning quality around it.

`Docs9/03_Feasible_Improvements.md` says ELA highlights "exactly the signal that copy-move boundaries produce." That is too strong and probably wrong in the general case. Copy-move often preserves same-image statistics better than splicing. ELA may help on some dataset-specific pasted or recompressed artifacts, but it is not some principled copy-move antidote.

So the real status of ELA is:

- reasonable hypothesis,
- attractive because it is cheap,
- risky because CASIA may reward it for the wrong reasons.

Docs9 presents it as closer to a justified gain than it really is.

## 4. The pHash leakage plan is not good enough as written

Docs9 correctly identifies that path-overlap checks are inadequate. Good. Then `Docs9/06_Notebook_V9_Implementation_Plan.md` proposes a pHash workflow that only groups exact hash strings. That is not a real near-duplicate detector.

This is the dataset equivalent of telling the truth about the problem and then under-engineering the fix.

If the dataset-leakage concern is serious enough to motivate a credibility repair, then the solution has to be real:

- define a near-duplicate threshold,
- compare hashes by distance,
- form groups before splitting.

Anything weaker is documentation therapy, not data integrity.

## 5. U-Net/ResNet34 is still only a baseline

Docs9 has not actually solved the architecture justification problem. It has merely improved the way it talks about it.

Current state:

- U-Net/ResNet34 remains the default baseline,
- DeepLabV3+ gets one comparison experiment,
- transformer and forensic multi-branch designs are pushed out of scope,
- ELA is used to patch the RGB limitation without really resolving it.

That can be a defensible strategy if the claim is kept narrow. But Docs9 keeps flirting with stronger implications than it has earned.

The correct architecture claim is:

"This is a practical, Colab-compatible baseline that we are trying to make assignment-complete."

That is much narrower than:

"This is now a well-reasoned solution to the tamper detection problem."

## 6. DeepLab comparison helps, but not as much as Docs9 implies

Approving a DeepLabV3+ comparison is directionally right. The problem is that Docs9 uses it as a kind of architecture absolution. One comparison run, especially at one seed, does not suddenly convert a convenience baseline into a deeply justified design.

Why not?

- It only compares one neighboring architecture.
- It does not address the RGB-versus-forensic-signal question.
- It does not prove the chosen model is "right," only maybe "good enough."

So yes, keep the comparison. No, do not pretend it closes the reasoning gap by itself.

## 7. One real strength in the architecture section

Docs9 does at least stop trying to solve every weakness with a giant research architecture. The deferred/rejected calls on transformers and multi-branch forensic networks are actually sensible for assignment scope.

That judgment is good. The problem is that the approved scope still remains too large overall.

## Verdict

Docs9 makes the dataset story more honest and the architecture story more disciplined. It still does not fully justify the proposed v9 design. CASIA remains a fragile benchmark, ELA is being sold too confidently, the pHash fix is undercooked, and U-Net/ResNet34 remains a baseline with paperwork, not a strongly proven forensic choice.
