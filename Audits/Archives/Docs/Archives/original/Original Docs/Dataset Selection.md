Selecting the right dataset is critical for training a robust image forensics model. For your
internship assignment, you should prioritize datasets that provide high-quality pixel-level ground
truth masks and a diverse set of manipulation types (splicing and copy-move).

# 1. Primary Recommendation: CASIA v2.

CASIA v2.0 is widely considered the most realistic and standard benchmark for general image
tampering detection. It is superior to CASIA v1 as it contains higher-resolution color images and
more complex forgeries.
● **Composition:** Approximately 4,975 images, including 1,701 authentic (Au) and 3,
tampered (Tp) images.
● **Forgery Types:** It includes both splicing (regions copied from different images) and
copy-move (regions copied from the same image).
● **Naming Convention:**
○ Au: Authentic images.
○ Tp_D: Spliced images (D for "Different" source).
○ Tp_S: Copy-move images (S for "Same" source).
● **Kaggle Access:** Use the dataset slug
divg07/casia-20-image-tampering-detection-dataset for easy integration into Google
Colab.

# 2. Specialized Dataset: COVERAGE

If you want to focus heavily on the "Copy-Move" requirement or bonus points, COVERAGE is
the industry standard for this specific attack.
● **Composition:** 100 original-tampered pairs (200 images total).
● **Unique Feature:** It is designed to address "self-similarity" ambiguity by using images
that contain multiple similar-but-genuine objects (SGOs), making it very difficult for
models to distinguish between a natural repetition and a forgery.
● **Mask Types:** Provides three mask types: copy source, forged region, and paste
destination.

# 3. Comprehensive Dataset: CoMoFoD

CoMoFoD is excellent for testing model robustness against "post-processing" attacks, which is
another bonus criteria for your assignment.
● **Composition:** 260 forged image sets in two categories: small (512x512) and large
(3000x2000).
● **Attack Diversity:** Images are grouped into translation, rotation, scaling, combination,
and distortion.
● **Robustness Traces:** All images have post-processing versions including JPEG


```
compression, blurring, and noise.
```
# Dataset Comparison Table Matrix

```
Dataset Total
Images
Forgery Types Key Strength
CASIA v2.0 ~4,975 Splicing &
Copy-Move
Large scale; high realism; widely
benchmarked.
COVERAGE 200 Copy-Move only Focuses on challenging "Similar
Genuine Objects".
CoMoFoD 260 sets Copy-Move Variety of transformations
(rotation/scaling).
IEEE
(Forensics)
~1,451 Splicing High-quality masks; 451 fake/
pristine.
```
# 4. Implementation & Cleaning Requirements

Regardless of the dataset chosen, you must address several known technical issues:
● **Resolution Misalignment:** A known issue in CASIA v2.0 (17 images) and COVERAGE
(9 images) is that the tampered image and its corresponding mask sometimes have
different resolutions or flipped dimensions. You must write a script to check image.size
== mask.size before training.
● **Mask Binarization:** Many masks in CASIA are not perfectly binary (0 and 255). You
should apply a threshold (e.g., mask > 128 ) during preprocessing to ensure a clean
binary ground truth.
● **Data Splitting:** A standard practice is a stratified split: 70% for training and 30% for
validation/testing to maintain the authentic-to-tampered ratio.
● **Class Balancing:** Because authentic images often outnumber tampered ones (or vice


versa in specific datasets), you may need to undersample the majority class to prevent
the model from getting "stuck" due to class imbalance.


