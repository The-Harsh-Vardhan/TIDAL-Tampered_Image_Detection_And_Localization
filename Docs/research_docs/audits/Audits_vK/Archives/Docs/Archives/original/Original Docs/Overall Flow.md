Completing an internship assignment on tampered image detection requires a systematic
approach that balances forensic theory with practical deep learning implementation. Below is a
stepwise guide to building your project in a single Google Colab notebook, optimized for the T
GPU.

# Step 1: Environment Setup and Dataset Acquisition

Your first priority is securing the data. CASIA v2.0 is the recommended starting point due to its
balance of splicing and copy-move examples.

1. **Configure Kaggle API:** Upload your kaggle.json to Colab and move it to ~/.kaggle/.
2. **Download Dataset:** Use !kaggle datasets download -d
    divg07/casia-20-image-tampering-detection-dataset.
3. **Install Libraries:** Ensure you have the following installed for segmentation and
    optimization:
       ○ segmentation-models-pytorch (SMP) for SOTA backbones.
       ○ albumentations for synchronized image/mask augmentation.
       ○ bitsandbytes if you intend to use 8-bit optimizers for memory efficiency.

# Step 2: Data Preprocessing and Alignment

Properly pairing images with their corresponding masks is critical, as the ground truth masks are
often stored in separate directories with specific naming conventions.

1. **Pairing Logic (CASIA v2):** Tampered images in the Tp folder typically follow a naming
    rule where the mask filename is the image stem plus _gt. For example,
    Tp_D_NRN_nat00025_11037.jpg maps to Tp_D_NRN_nat00025_11037_gt.png.
2. **Handle Imbalance:** Create a balanced subset (e.g., 1000 authentic and 2000 tampered)
    to prevent the model from becoming biased toward the "authentic" majority class.
3. **Train/Val/Test Split:** Standard practice for this dataset is a split of approximately 85%
    training, 7.5% validation, and 7.5% testing.

# Step 3: Forensic Feature Engineering

Forensic models perform better when semantic content is suppressed to highlight low-level
artifacts.

1. **SRM Filter Layer:** Initialize a non-trainable nn.Conv2d layer with 30 SRM high-pass
    kernels. This captures noise residuals that indicate local variance mismatches common
    in spliced regions.
2. **Bayar Convolution:** If you choose a learnable approach, implement a BayarConv2d
    layer. The kernels must be constrained so that the central weight $W_{0,0} = -1$ and the
    sum of all other weights $\sum_{i,j \neq 0,0} W_{i,j} = 1$.


# Step 4: Model Architecture Selection

For a Colab-friendly yet SOTA-aligned solution, use a dual-stream approach or a hierarchical
transformer.

1. **Backbone Selection:** Use segmentation_models_pytorch to create a U-Net or
    DeepLabV3+ with a pre-trained resnet34 or efficientnet-b1 encoder.
2. **Dual-Stream Integration:** Concatenate the RGB image with the output of your
    SRM/BayarConv filters before passing them into the encoder. This allows the model to
    "see" both high-level semantic edges and low-level noise artifacts.
3. **Decoder Configuration:** Ensure the final layer uses a sigmoid activation for binary
    mask generation.

# Step 5: Implementing Robust Loss Functions

To address the extreme class imbalance (where forgeries are small), avoid using only Binary
Cross-Entropy (BCE).

1. **Hybrid Loss:** Combine Dice Loss (which optimizes for region overlap) and BCE or Focal
    Loss (which focuses on "hard" pixels at boundaries) :
    $$L_{total} = L_{BCE} + L_{Dice}$$
2. **Implementation:** Use the direct implementation of Dice Loss: $1 - \frac{2 \sum P \cdot G
    + \text{smooth}}{\sum P + \sum G + \text{smooth}}$, where $P$ is the prediction and
    $G$ is the ground truth.

# Step 6: Training Optimization (T4 GPU)

Maximize the 16GB VRAM on the T4 GPU using PyTorch's performance tools.

1. **Mixed Precision (AMP):** Use torch.cuda.amp.autocast() and GradScaler() to perform
    computations in FP16. This reduces memory usage by up to 50%.
2. **Gradient Accumulation:** If a batch size of 8 or 16 is too large for the memory, use
    accumulation_steps (e.g., 4) to update the optimizer every 4 micro-batches, simulating a
    larger batch size.
3. **Preprocessing on GPU:** Use pin_memory=True in your DataLoader to speed up data
    transfer to the GPU.

# Step 7: Rigorous Testing and Robustness (Bonus Points)

To earn bonus points, you must demonstrate your model's limits under adversarial conditions.

1. **Standard Metrics:** Calculate Pixel-level F1-score, IoU (Intersection over Union), and
    Image-level accuracy.
2. **Robustness Transformations:** Use torchvision.transforms.v2 to subject your test set to
    :
       ○ **JPEG Compression:** Test performance at Quality Factors (QF) of 50, 70, and


## 90.

```
○ Gaussian Noise: Add noise with varying standard deviations ($\sigma = 0.01$ to
$0.1$).
○ Resizing: Test performance after downscaling by 0.5x.
```
# Step 8: Visualization and Documentation

Clear visuals are a core requirement of the assignment deliverable.

1. **Visual Comparison Grid:** Create a function to display a 4-column plot for sample test
    images:
       ○ **Column 1:** Original RGB image.
       ○ **Column 2:** Ground Truth mask.
       ○ **Column 3:** Predicted probability heatmap.
       ○ **Column 4:** Overlay of the predicted mask on the original image using alpha=0.
          for transparency.
2. **Confidence Maps:** If possible, visualize a "reliability map" where the intensity
    represents the model's confidence in its tampering prediction.
