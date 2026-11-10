# 01 - Assignment Compliance and Deliverable Gaps

## Bottom line

This submission fails the assignment as written in `Assignment.md`. The notebook contains a segmentation pipeline skeleton, but it does not provide trustworthy evidence of a completed experiment, it is not submitted as a real Colab-ready notebook, and it does not implement a proper image-level detector.

## Requirement-by-requirement audit

| Assignment requirement | Notebook status | Why it fails |
|---|---|---|
| Tampered image detection and localization | Partial at best | Localization is attempted. Detection is not modeled; Cell 33 derives image labels from `max(prob_map)`. |
| Model architecture that predicts tampered regions | Partial | Cell 20 builds a U-Net segmentation model. That covers region prediction, but the architecture rationale is shallow and outsourced. |
| Clear reasoning behind architecture choices | Weak | Cell 19 says the architecture is retained because `Docs8 Section 03` says it is not the bottleneck. That is not notebook-native reasoning. |
| Runnable on Google Colab or similar cloud environment | Fail as submitted | Cells 5, 7, 8, and 9 are Kaggle-specific. The notebook is not portable. |
| Single Google Colab notebook deliverable | Fail | The artifact is explicitly a Kaggle notebook and has no proof of execution. |

## The notebook has no execution evidence

This is the biggest compliance problem. The notebook has sections, code, and comments, but no execution counts and no preserved outputs. That means there is no proof that:

- the dataset was actually discovered correctly,
- the model instantiated successfully,
- the training loop ran,
- checkpoints were produced,
- the threshold sweep worked,
- the evaluation numbers existed,
- the plots were generated.

An internship submission is not graded on what the code might do in theory. It is graded on what the submitted artifact proves.

## The Colab requirement is not satisfied

The assignment says the entire implementation must be delivered in a single Google Colab notebook. The notebook ignores that and assumes Kaggle everywhere.

Evidence:

- Cell 5 sets `OUTPUT_DIR = '/kaggle/working'`.
- Cell 8 says the dataset is pre-mounted by Kaggle at `/kaggle/input/`.
- Cell 9 scans `/kaggle/input` for `Image/` and `Mask/`.
- Cell 7 imports `kaggle_secrets` and uses `UserSecretsClient` to fetch the W&B API key.

That is not "Colab or similar" portability. That is one environment with one storage layout and one secret provider. On Colab, the notebook would need path rewrites, dataset mounting logic, and a different secret flow before it even reaches the training loop.

## The notebook claims detection, but it only implements localization

This is the other major compliance failure. The assignment requires both image-level detection and pixel-level localization. The notebook only has a segmentation model:

- Cell 20 builds `smp.Unet(... classes=1, activation=None)`.
- Cell 23 defines segmentation losses.
- Cell 33 later invents an image score with `tamper_score = probs[i].view(-1).max().item()`.

That is not a dual-task model. That is a post hoc heuristic. One hot pixel in the probability map can flip the whole image-level decision. The notebook is quietly dodging the design problem instead of solving it.

## The reasoning is decorative, not convincing

Cell 19 says the architecture is retained from v6.5 because `Docs8 Section 03` determined the architecture is not the primary bottleneck. Cell 21 and Cell 30 do the same trick for training and evaluation. The notebook keeps borrowing authority from planning documents instead of showing reasoning inside the submitted artifact.

That matters because the assignment explicitly asks for clear reasoning behind architecture choices. "We kept it because our own notes say the problem is elsewhere" is not clear reasoning. It is self-citation.

## The notebook is more complicated than its evidence justifies

The submission includes:

- W&B integration in Cell 7,
- checkpoint resume in Cells 28 and 29,
- threshold sweeps in Cell 32,
- visualization in Cells 36 to 40,
- Grad-CAM in Cells 42 and 43,
- robustness tests in Cells 46 to 48,
- shortcut-learning checks in Cell 50,
- artifact inventory in Cell 55.

That would be fine if the core experiment were already proven. It is not. Right now the notebook has the structure of a mature submission and the evidentiary value of an outline.

## Verdict

This is a fail, not a partial pass. A principal engineer reviewing this in an interview would say the same thing: the notebook shows the author knows what a solid submission should contain, but the actual deliverable is still unproven, Kaggle-bound, and technically incomplete on the detection requirement.
