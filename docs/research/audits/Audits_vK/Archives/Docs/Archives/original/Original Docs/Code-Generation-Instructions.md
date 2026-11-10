# Copilot Code Generation Instructions — PyTorch Implementation

You are assisting in writing PyTorch code for a **tampered image detection and localization system**.

Write code as if it will be reviewed by a **Principal AI Engineer**.

Focus on producing **clean, modular, research-quality PyTorch code**.

---

# 1. Code Structure

Do not place all code inside one training cell.

Organize code into logical sections:

dataset utilities
augmentation pipeline
model architecture
loss functions
training loop
evaluation functions
visualization utilities

Each component must be implemented as a **reusable function or class**.

---

# 2. Dataset Class

Always implement a custom PyTorch Dataset class.

Responsibilities:

• load image
• load mask
• apply transformations
• return tensor pairs

The dataset class should return:

(image_tensor, mask_tensor)

Masks must be **binary tensors** with values 0 or 1.

Never convert masks into floating grayscale values.

---

# 3. DataLoader

Use PyTorch DataLoader with:

shuffle=True for training
shuffle=False for validation and testing

Include:

batch_size
num_workers
pin_memory

Batch objects should contain:

batch_images
batch_masks

---

# 4. Model Implementation

Models must inherit from:

torch.nn.Module

Structure the model into clear components:

encoder
decoder
segmentation head
optional classification head

Prefer pretrained encoders from torchvision or timm.

The forward method must return:

segmentation_mask

Optionally return:

tamper_probability

Keep the forward pass clean and readable.

---

# 5. Residual Filtering Layer (Optional Enhancement)

If residual filtering is implemented:

apply a high-pass convolution layer before the encoder.

Purpose:

highlight manipulation artifacts
reduce semantic bias

Ensure this layer is implemented as a simple convolution module.

---

# 6. Loss Function Implementation

Implement segmentation loss as:

Binary Cross Entropy + Dice Loss

Dice Loss must be implemented explicitly rather than using external libraries.

Ensure numerical stability using epsilon values.

Return a scalar loss value.

---

# 7. Metric Functions

Implement separate functions for:

IoU calculation
Dice score
pixel accuracy

These functions must:

accept predicted mask and ground truth mask
handle thresholding of model outputs
avoid division by zero

Metrics must operate on tensors.

---

# 8. Training Loop

Write a clean training loop with:

model.train()
optimizer.zero_grad()
forward pass
loss calculation
backpropagation
optimizer.step()

Log:

training loss
validation loss
evaluation metrics

Use tqdm for progress visualization.

---

# 9. Validation Loop

The validation loop must:

use model.eval()
disable gradient calculation
compute validation metrics

Never update weights during validation.

---

# 10. Checkpointing

Include functionality to save the best model based on:

highest validation IoU or Dice score.

Save:

model state_dict
optimizer state
epoch number

This allows training to resume if needed.

---

# 11. Prediction Function

Implement a reusable prediction function that:

accepts an input image
runs model inference
applies sigmoid activation
thresholds output to create binary mask

Return:

predicted mask

---

# 12. Visualization Utilities

Implement a visualization function that displays:

original image
ground truth mask
predicted mask
overlay visualization

Overlay must highlight tampered pixels clearly.

Use matplotlib for plotting.

---

# 13. Robustness Testing Utilities

Implement functions that apply distortions:

JPEG compression
Gaussian noise
resizing

These should generate modified versions of test images.

Use these functions during robustness evaluation.

---

# 14. Logging

Print training progress per epoch.

Log:

epoch number
training loss
validation loss
IoU score
Dice score

Ensure logs are readable and formatted cleanly.

---

# 15. Code Quality Rules

When generating code:

avoid deeply nested functions
avoid long monolithic classes
prefer simple, readable logic
comment non-obvious operations

Code should be understandable to another engineer immediately.

---

# Final Principle

The goal is not just to train a model.

The goal is to produce a **clear, reliable machine learning pipeline** that demonstrates engineering maturity.
