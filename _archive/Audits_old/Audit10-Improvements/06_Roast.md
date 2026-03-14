# The Roast

An honest, brutal assessment of each notebook. No sugar-coating.

---

## v8 Run — "The Overengineered Underperformer"

You built a Ferrari chassis and strapped a lawnmower engine to it.

27 epochs on 2x T4 GPUs with gradient accumulation, differential learning rates, AMP, DataParallel, a pretrained ResNet34 backbone, and comprehensive robustness testing... and your tampered-only F1 is **0.29**. Nearly 7 out of 10 tampered pixels are wrong.

Your copy-move detection F1 is **0.14**. That's worse than a coin flip on a structured task. Your own failure analysis shows 9 out of 10 worst predictions are copy-move. You knew this was broken and submitted anyway.

The threshold sweep found optimal at 0.75 — your model is so unsure about tampering that it needs to see a 75% confidence before it commits to a pixel being tampered. That's not a model, that's a model with commitment issues.

Val loss went UP while training loss went down. Classic overfitting that your ReduceLROnPlateau couldn't fix because the problem isn't the learning rate — it's that BCEpos_weight=30 is making the model hallucinate tampering everywhere, then the high threshold hacks it back. Two wrongs making a mediocre right.

**The good:** Your evaluation methodology is genuinely excellent. Robustness testing, Grad-CAM, shortcut checks, mask-size stratification — this is how you evaluate a model. You just needed a better model to evaluate.

**Verdict:** A+ methodology on a C- model.

---

## vK.7.5 Run — "The Beautiful Corpse"

You wrote a 93-cell masterpiece of documentation ergonomics. Structured docstrings with Purpose/Inputs/Returns/Notes on every function. A table of contents. Assignment alignment notes in every section. Collapsible nothing because the notebook needs more content, not less.

Then you trained for **2 epochs**.

Two. Epochs.

Your training accuracy is **0.476**. That's below random for binary classification. Your model is actively anti-learning. If you flipped its predictions you'd get better results.

Dice = IoU = F1 = 0.5935. Those three metrics being identical is a mathematical impossibility unless your model outputs a constant value. It's predicting all-zeros — "nothing is tampered" — and getting credit because half your test set is authentic with empty ground truth masks. Your "model" is a very expensive `np.zeros()`.

And then there's the prior experiment block where you accidentally trained on the test set. `TRAIN_CSV = "test_metadata.csv"`. You reversed your CSVs and nobody caught it. The 0.60 accuracy from that block? That's test-set memorization, not learning.

But hey, your docstrings are pristine. Every function has a Notes section explaining that the function does exactly what the function name says. `__len__` returns the number of examples. Thank you for that insight.

**The good:** If someone needs a template for how to document a Jupyter notebook, this is it. The environment handling (Colab/local/Drive/API fallback) is genuinely useful infrastructure.

**Verdict:** 10/10 documentation for a 0/10 model.

---

## vK.3 Run — "The Accidental Success"

Somehow, through sheer brute force of running 50 epochs without early stopping, you stumbled into **0.899 accuracy** and a passable Dice of 0.576.

No AMP. No checkpoint resume. No CONFIG dict. No weight decay on a 31M parameter model. CosineAnnealing with T_max=10 doing 5 full restart cycles like a learning rate rollercoaster. No early stopping, so you trained past convergence and just hoped the last-saved model wasn't overfit.

Your best model was selected on **classification accuracy** — not localization quality. For an assignment that literally says "localize tampered regions" in the title, you optimized for the wrong metric. It's like studying for math and showing up to the English exam. You passed, but for the wrong reasons.

Your prior experiment block has the same data leakage bug as vK.7.5. You trained on `test_metadata.csv`. This is the third notebook in a row with this bug. At this point it's not a bug, it's a tradition.

The training loop saves both `best_model_1.pth` and `best_model.pth` from different experiment blocks to the same directory. If Kaggle cached the first (broken) model and your second training loaded it... you'd never know. No verification, no checksums, no artifact management.

**The good:** You actually trained the model. 50 epochs, real convergence, real metrics. The custom U-Net architecture works — 0.899 accuracy proves the dual-head design is sound. This is the only notebook where the model genuinely learned something.

**Verdict:** The metrics are real but the engineering is held together with duct tape.

---

## vK.10.3 — "The Promising Untested"

You've spent more time engineering the notebook than training a model.

You have AMP, gradient clipping, cosine scheduling, three-tier checkpoints with history persistence, early stopping, W&B integration, VRAM auto-adjustment, metadata caching, seed everything, collapsible markdown sections, ROC-AUC computation, tampered-only metric splits, and a CONFIG dict that would make a DevOps engineer weep with joy.

You have zero training results. Zero test metrics. Zero visualizations with actual predictions. You have never run this notebook.

Your U-Net trains from scratch with no pretrained backbone. v8 proved that a pretrained ResNet34 gets AUC=0.817. You're starting from random weights, crossing your fingers, and hoping 50 epochs on a 256x256 custom U-Net is enough.

Your augmentation pipeline has 6 transforms but no VerticalFlip, no ElasticTransform, no CoarseDropout. Your segmentation threshold is hardcoded at 0.5 with no optimization. You have no robustness testing, no Grad-CAM, no threshold sweep, no forgery-type breakdown, no mask-size analysis, no shortcut learning checks.

In other words: you built the garage but haven't built the car yet. All the infrastructure is there — checkpointing, resuming, early stopping — but the features that actually earn assignment points (robustness testing for B1, explainability for E3, threshold optimization for localization quality) are completely absent.

**The good:** If the model trains well, the infrastructure is solid. The CONFIG dict, checkpoint system with history, tampered-only metrics, and ROC-AUC are genuine improvements over all previous notebooks. The bug fixes (weight_decay, T_max=50, safe mask loading, W&B fixes) show engineering maturity.

**Verdict:** Best engineering foundation, needs to actually run and needs advanced evaluation features to compete with v8's methodology.

---

## The Uncomfortable Truth

None of these notebooks would pass a serious research review:

- **v8** has the best methodology but mediocre localization results (tampered F1=0.29)
- **vK.3** has the best raw metrics but no evaluation depth
- **vK.7.5** has the best documentation but no real results
- **vK.10.3** has the best engineering but no results at all

The ideal submission would be: vK.10.3's engineering + v8's evaluation methodology + vK.3's actual convergence + vK.7.5's documentation style. That notebook doesn't exist yet. But vK.10.3 is the closest starting point.

**What vK.10.3 needs before submission:**
1. Actually run it and get real metrics
2. Add robustness testing (explicit bonus points)
3. Add threshold optimization (free metric improvement)
4. Add Grad-CAM (evaluation rigor)
5. Add data leakage verification (credibility after the CSV bugs)
6. Add mask-size stratification and forgery-type breakdown (evaluation depth)
7. Run for enough epochs to converge (vK.3 proved 50 epochs works for this architecture)

Do those 7 things and vK.10.3 becomes the best notebook in the family by a wide margin.
