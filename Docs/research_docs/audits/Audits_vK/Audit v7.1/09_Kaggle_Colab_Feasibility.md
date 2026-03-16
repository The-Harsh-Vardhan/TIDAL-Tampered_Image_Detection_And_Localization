# 9. Kaggle / Colab Feasibility

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

This notebook is realistically feasible on Kaggle GPU. A full-width U-Net with a `1024`-channel bottleneck at `256x256` and batch size `8` is not lightweight, but it is generally manageable on common Kaggle GPUs such as T4 or P100-class hardware (cells 61, 63).

It is also probably feasible on Google Colab GPU, but less elegantly. The notebook does not really become native Colab code. Instead, it emulates Kaggle paths inside Colab by creating `/kaggle/input` and `/kaggle/working`, then normalizes the dataset into that structure (cells 6, 13, 15). That is workable, but it is clearly fallback behavior, not a first-class Colab implementation.

The main inefficiency is the lack of AMP. Without mixed precision, training will be slower and memory headroom tighter than necessary. Still, the dataset size and I/O burden appear manageable, and the pipeline is practical enough for an assignment notebook.
