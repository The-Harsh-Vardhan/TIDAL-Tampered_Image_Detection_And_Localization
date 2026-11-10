# Dataset and Model Reasoning Audit

This note covers the two places where `Docs8` improved its honesty but still did not fully solve the underlying technical problem: dataset reasoning and model reasoning.

## 1. CASIA is still the easy answer, not the strong answer

`Docs8` is better than `Docs7` because it finally admits several things CASIA cannot do:

- it is a 2013 legacy benchmark
- it lacks modern manipulation types
- path-disjointness is not content-disjointness
- mask quality was not independently audited

That is all clearly documented (`Docs8/02_Dataset_Evolution.md:28-37`).

The problem is that the document then undercuts its own honesty with this line:

> "Keep CASIA. It is the assignment's expected dataset." (`Docs8/02_Dataset_Evolution.md:112`)

That is still wrong. `Assignment.md:14-17` says CASIA is an example alongside Coverage, CoMoFoD, and other relevant public datasets.

This is not just semantics. It reveals a persistent tendency to rewrite the assignment into "use CASIA" whenever defending the current pipeline becomes inconvenient.

## 2. Leakage handling is still too generous for the evidence available

`Docs8` says two incompatible things:

- `Docs8/00_Project_Evolution_Summary.md:78` says "No bugs or data leakage detected."
- `Docs8/01_Assignment_Requirement_Alignment.md:17-18` admits the split only has "0 leaks by path" and no content-based near-duplicate check.
- `Docs8/02_Dataset_Evolution.md:94-96` proposes a future pHash or CLIP-based near-duplicate audit because that check has not been run yet.

That means the real state is:

"No path overlap detected. Content leakage not yet ruled out."

Anything stronger is overclaiming.

## 3. Docs8 finally understands the copy-move problem, but it still does not solve it

This is one of the few places where `Docs8` is genuinely sharper than older docs.

It explicitly says:

- copy-move F1 is 0.3105
- copy-move is the majority tampered class
- same-image duplication kills many of the inter-region artifact cues the RGB model depends on

Evidence: `Docs8/00_Project_Evolution_Summary.md:51-54`, `Docs8/02_Dataset_Evolution.md:54-62`, `Docs8/11_Training_Failure_Cases.md:52-95`.

That diagnosis is strong.

The limitation is that the proposed response is still mostly training-side patchwork:

- `pos_weight`
- stronger augmentation
- per-type loss tracking

Those may help. They do not change the fact that the model still sees RGB only and still lacks the forensic signals copy-move most needs (`Docs8/11_Training_Failure_Cases.md:81-95`).

## 4. U-Net/ResNet34: the reasoning is cleaner, the answer is still conservative

The architecture section is one of the clearest signs of improvement in `Docs8`.

`Docs8/03_Model_Architecture_Evolution.md:76-89` explicitly says the model is retained because:

- training and calibration issues are higher priority
- changing architecture and loss together would muddy ablations
- U-Net/ResNet34 is a stable reference point

That is a serious improvement over the old "it is standard and fits on T4" defense.

But it also amounts to this:

"We know this is a generic baseline and not a strong forensic architecture, but we are postponing the hard comparison."

That is a valid engineering choice for sequencing work. It is not a strong final architecture justification.

## 5. Why U-Net?

The best version of the `Docs8` argument is:

- dense prediction is the right formulation
- U-Net is a stable baseline for mask prediction
- the current bottlenecks appear to be calibration, imbalance, and robustness rather than raw capacity

That is defensible.

The weak version - and the one that still lingers implicitly - is:

"U-Net is fine because it is common."

For a senior review, common is not enough. The real question is whether the feature extractor is aligned with forensic traces rather than generic semantic structure.

## 6. Why ResNet34?

The project's actual answer is still mostly resource-driven:

- manageable size
- pretrained
- T4-friendly

`Docs8` at least stops pretending those are forensic reasons (`Docs8/03_Model_Architecture_Evolution.md:22-27`, `Docs8/03_Model_Architecture_Evolution.md:88-89`).

But the downside remains obvious:

ResNet34 pretrained on ImageNet is good at semantics, not necessarily at noise residuals, seam artifacts, or resampling traces.

`Docs8` knows this. It directly admits RGB-only limitations and names SRM and ELA as later priorities (`Docs8/03_Model_Architecture_Evolution.md:45-49`, `Docs8/03_Model_Architecture_Evolution.md:112-115`, `Docs8/07_Shortcut_Learning_Risk_Assessment.md:141-148`).

## 7. Why not DeepLabV3+ or transformers?

`Docs8` gives a more disciplined answer than `Docs7`:

- architecture should be compared only after training is stabilized
- `DeepLabV3+` is a low-friction future comparison
- transformer variants are higher effort and later priority

Evidence: `Docs8/03_Model_Architecture_Evolution.md:110-115`, `Docs8/09_Future_Experiments.md:56-84`, `Docs8/09_Future_Experiments.md:115-139`.

That is reasonable as sequencing.

It is not enough as final reasoning because the alternatives are all deferred. The project still cannot say, from executed evidence, whether U-Net/ResNet34 is the right tradeoff or just the familiar one.

## 8. Research awareness is better, but still mostly aspirational

`Docs8/10_References.md:44-72` shows decent modern awareness:

- `DeepLabV3+`
- `SegFormer`
- `ManTraNet`
- `MVSS-Net`
- `ObjectFormer`
- SRM and ELA inputs

That is not the problem.

The problem is that the implementation position is still:

- baseline now
- serious forensic features later
- serious architecture comparison later
- real generalization tests later

That means the project is literature-aware without yet being literature-competitive.

## 9. The correct senior-level reading

The right reading of `Docs8` is not:

"The author has now justified the model."

The right reading is:

"The author now understands that the current model is a baseline with specific forensic blind spots, and they finally wrote that down honestly."

That is progress. It is not closure.

## 10. Bottom line

Dataset reasoning:

- improved honesty
- still one lingering assignment misread
- still too generous about leakage certainty

Model reasoning:

- much cleaner baseline framing
- still no executed comparison against obvious alternatives
- still no real fix for RGB-only blindness

So the answer is the same in both areas: `Docs8` acknowledges the problem more accurately than `Docs7`, but the technical gap is still open.
