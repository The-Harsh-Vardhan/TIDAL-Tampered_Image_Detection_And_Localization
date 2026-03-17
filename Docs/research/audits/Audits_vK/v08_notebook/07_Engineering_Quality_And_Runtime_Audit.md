# 07 - Engineering Quality and Runtime Audit

## Bottom line

The engineering quality is mixed. There are several good software instincts in the notebook, but they are undermined by environment coupling, default brittleness, and a habit of building infrastructure ahead of validated results.

## 1. Good instincts that are real

Credit where it is due:

- Cell 5 centralizes configuration.
- Cells 6 and 17 make an honest attempt at reproducibility with seeds and worker initialization.
- Cells 28 and 29 support checkpoint save and resume.
- Cell 13 writes a split manifest.
- Cell 53 writes a results summary.

Those are the bones of a decent experimental codebase. The author is not flailing.

## 2. Kaggle lock-in is the biggest engineering failure

The notebook is welded to Kaggle:

- `OUTPUT_DIR = '/kaggle/working'` in Cell 5,
- `KAGGLE_INPUT = '/kaggle/input'` in Cell 9,
- Kaggle secrets in Cell 7,
- markdown in Cell 8 explicitly assumes Kaggle mounting.

That is not just a portability issue. It is a design-quality issue. A notebook submitted for a Colab-centric assignment should not require environment surgery to start.

## 3. Default settings are brittle

Several defaults make the notebook more fragile than it needs to be:

- `use_wandb=True` by default means the notebook expects secrets and internet-backed logging behavior before proving the base run.
- `use_multi_gpu=True` is enabled even though the assignment centers on Colab-class hardware where multi-GPU is usually irrelevant.
- `train_ratio` exists in config but is ignored by the split code in Cell 12.
- `encoder_warmup_epochs` exists as a knob but defaults to zero and is never justified.

This is what happens when a config grows faster than discipline. The file looks flexible while some knobs do nothing important.

## 4. Checkpointing is useful, but resume behavior is a little too magical

Cell 29 auto-resumes from `last_checkpoint.pt` if it exists. That is convenient. It is also risky in notebook workflows because rerunning cells in the wrong order can continue an old state silently. There is no explicit resume flag, no hash check tying the checkpoint to the current config, and no guard against stale optimizer state.

That is not disastrous. It is just a reminder that notebook ergonomics are not the same thing as experiment hygiene.

## 5. Reproducibility is partial

The seeding work in Cell 6 and Cell 17 is decent. The reproducibility story is still incomplete because:

- package versions are only loosely pinned in Cell 3,
- dataset versioning is not locked beyond an assumed Kaggle attachment,
- notebook outputs are not preserved,
- there is no immutable run record in the submitted artifact,
- DataParallel behavior can differ from single-device behavior.

This is not reproducibility. This is reproducibility-themed code.

## 6. Data pipeline efficiency is acceptable, not impressive

The dataset is loaded with `cv2.imread` in `__getitem__`, which is normal. The pipeline then rescans all training masks in Cell 22 just to compute `pos_weight`. That extra pass is not catastrophic for a modest dataset, but it is another sign that the notebook is optimized for clarity over efficiency.

For an interview baseline, that is fine. For a submission that wants to look polished, it is mediocre.

## 7. Runtime feasibility: the compute likely fits, the notebook still does not

If you ignore the environment issues, the raw compute footprint is probably reasonable for a T4-class GPU:

- ResNet34 U-Net,
- `384 x 384` inputs,
- batch size `4`,
- AMP,
- gradient accumulation to effective batch `16`.

So the model choice itself is probably not the runtime blocker. The blocker is the notebook design:

- hard-coded Kaggle paths,
- Kaggle-specific secret handling,
- no Colab data-mount path,
- no proven outputs showing actual runtime success.

For Kaggle, the notebook is plausible. For Colab as submitted, it fails. For both, the absence of executed outputs means the runtime claim is still unproven.

## Verdict

The engineering quality is better than a throwaway notebook and worse than a serious submission. The code shows structure, but the structure is undermined by platform lock-in, default brittleness, partial reproducibility, and more infrastructure than evidence.

The clean summary is this: the author has decent ML notebook engineering habits, but not yet the discipline to turn them into a trustworthy deliverable.
