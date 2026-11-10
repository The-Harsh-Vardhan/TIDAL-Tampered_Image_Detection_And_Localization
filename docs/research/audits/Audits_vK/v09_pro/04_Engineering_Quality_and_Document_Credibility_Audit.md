# Engineering Quality and Document Credibility Audit

## Bottom line

Docs9 is much more organized than earlier design sets. It is also still too willing to overstate maturity. The engineering plan has decent structure, but several details are under-specified, some difficulty estimates are fantasy, and the document set keeps treating planned runtime validation as if it already lowered risk.

## 1. The engineering structure is cleaner

Real strengths first:

- explicit decision log,
- ordered implementation plan,
- risk register,
- deferred versus rejected separation,
- Colab verification called out as a gate.

That is good engineering process hygiene. It is better than raw brainstorming.

## 2. The ELA implementation sketch is technically messy

This is one of the biggest engineering issues in Docs9.

`Docs9/06_Notebook_V9_Implementation_Plan.md` proposes:

- compute grayscale ELA,
- pass it through Albumentations as `additional_targets={'ela': 'image'}`,
- apply the same transform pipeline that includes `A.Normalize(...)`,
- then manually divide the ELA tensor by `255.0` and concatenate it to already normalized RGB.

That is under-specified and likely wrong.

Problems:

1. Treating grayscale ELA as a generic `image` target can apply photometric transforms to it in ways that may not make sense.
2. The normalization path is unclear and likely inconsistent with the RGB path.
3. The docs do not define whether ELA should be recomputed before or after compression augmentation.
4. The claim that this is easy is nonsense.

This is exactly the kind of detail that breaks notebook implementations while the markdown still sounds confident.

## 3. The pHash plan is under-engineered

The docs correctly identify content-level leakage as a credibility problem. Good. The proposed implementation is still weak:

- exact hash grouping is not near-duplicate search,
- the docs mention O(n²) but the code sketch does not do pairwise thresholding,
- the plan says "if duplicates found, group them into the same split" after the split logic is already established.

That is not a finished engineering design. It is a half-design with a reassuring label.

## 4. Difficulty estimates are too optimistic

Docs9 repeatedly labels substantial changes as easy:

- dual-head architecture,
- 4-channel pretrained adaptation,
- ELA integration,
- new multi-task loss,
- new evaluation branches,
- pHash integrity logic,
- three-seed experiment schedule.

The implementation plan itself contradicts those estimates by showing the real number of moving parts. That is a credibility smell. When docs consistently underestimate effort, schedule and scope drift are almost guaranteed.

## 5. Colab feasibility is not engineered tightly enough

Docs9 says Colab matters. Good. Then it approves a pipeline whose own risk assessment estimates roughly eleven hours of work on a T4 across the v9 experiment slate.

That is already too close to session limits for a project with:

- new dependencies,
- architectural changes,
- multi-seed runs,
- comparison experiments,
- ablations,
- visualization,
- potential runtime debugging.

If Colab feasibility is a hard constraint, the engineering plan should privilege:

- one stable submission configuration,
- one stable evaluation configuration,
- one verified end-to-end run.

Instead, Docs9 keeps a research-shaped workload attached to the same milestone.

## 6. The document credibility problem is smaller, not gone

Docs9 fixes a lot of the old lineage and framing issues. That is real progress. The remaining credibility issue is subtler:

- it no longer makes obviously false claims,
- but it still upgrades provisional state into "current status" too easily.

The clearest example is v8. The docs talk about v8 as a current implemented baseline with verified properties. The repo only proves that v8 notebook files exist and contain code. It does not prove that the notebooks were successfully executed and validated as stored artifacts.

That gap matters because Docs9 uses v8 as the baseline for estimating scope, benefits, and stability.

## 7. Checkpointing and tracking are not the problem

Docs9 retains W&B, checkpointing, and structured config. Those are fine in themselves. The problem is that the project keeps investing in scaffolding before fully locking down the minimal submission path. That is an engineering prioritization issue, not a tooling issue.

## Verdict

Docs9 has much better engineering organization than earlier versions, but its technical plan is still not fully credible. The two biggest issues are:

1. under-specified details in the ELA and leakage-handling plans,
2. overly optimistic maturity assumptions inherited from unexecuted v8 notebooks.

This is a better-managed plan, not yet a safer one.
