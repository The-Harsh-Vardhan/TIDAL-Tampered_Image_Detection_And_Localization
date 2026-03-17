# 17. Training Optimisation — Advanced Techniques Guide

## 17.1 Scope

This document covers optimisation techniques **beyond the basics** (AMP, gradient accumulation) already covered in Doc 05. Here we explore parallelism, NVIDIA-specific optimisations, advanced batching strategies, and I/O pipeline tuning.

---

## 17.2 Optimisation Landscape

```
┌──────────────────────────────────────────────────────────────┐
│                 Training Speed Bottlenecks                    │
│                                                              │
│  Data Loading ──→ Preprocessing ──→ GPU Compute ──→ Logging  │
│  (CPU + I/O)     (CPU or GPU)      (Forward+Back)   (Sync)  │
│                                                              │
│  Each bottleneck has different solutions:                     │
│  - I/O bound → prefetching, caching, faster storage          │
│  - CPU bound → more workers, GPU preprocessing, DALI         │
│  - GPU bound → AMP, larger batch, model pruning, compilation │
│  - Sync bound → async logging, non-blocking transfers        │
└──────────────────────────────────────────────────────────────┘
```

---

## 17.3 Technique 1: NVIDIA DALI — GPU Data Pipeline

### What Is DALI?
NVIDIA Data Loading Library (DALI) moves the **entire data pipeline** (decode → resize → augment → normalize) to the GPU, eliminating the CPU-to-GPU data transfer bottleneck.

### Traditional Pipeline vs. DALI

```
Traditional:
  Disk → CPU (decode JPEG) → CPU (resize) → CPU (augment) → CPU→GPU transfer → GPU (model)
  
DALI:
  Disk → GPU (decode + resize + augment + normalize) → GPU (model)
  ^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Only bottleneck is disk read speed. All processing on GPU.
```

### Implementation

```python
!pip install -q nvidia-dali-cuda120  # Match your CUDA version

from nvidia.dali.pipeline import Pipeline
from nvidia.dali import fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class TamperPipeline(Pipeline):
    def __init__(self, image_dir, mask_dir, batch_size, num_threads, device_id, 
                 image_size=512):
        super().__init__(batch_size, num_threads, device_id)
        self.image_input = fn.readers.file(file_root=image_dir, random_shuffle=True)
        self.mask_input = fn.readers.file(file_root=mask_dir, random_shuffle=True)
        self.image_size = image_size
    
    def define_graph(self):
        images, _ = self.image_input
        masks, _ = self.mask_input
        
        # Decode on GPU
        images = fn.decoders.image(images, device='mixed', output_type=types.RGB)
        masks = fn.decoders.image(masks, device='mixed', output_type=types.GRAY)
        
        # Resize on GPU
        images = fn.resize(images, size=(self.image_size, self.image_size))
        masks = fn.resize(masks, size=(self.image_size, self.image_size), 
                         interp_type=types.INTERP_NN)  # Nearest for masks
        
        # Augmentations on GPU
        images = fn.flip(images, horizontal=fn.random.coin_flip())
        
        # Normalize on GPU
        images = fn.crop_mirror_normalize(
            images,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=types.FLOAT,
            output_layout='CHW'
        )
        
        return images, masks

# Create pipeline
pipe = TamperPipeline(
    image_dir='./data/train/images',
    mask_dir='./data/train/masks',
    batch_size=4,
    num_threads=2,
    device_id=0
)
pipe.build()

# PyTorch iterator
train_loader = DALIGenericIterator(pipe, ['images', 'masks'])
```

### Should You Use DALI?

| Factor | Assessment |
|--------|-----------|
| **Speed gain** | 30-50% for I/O-bound pipelines with heavy augmentation |
| **Complexity** | High — different API from PyTorch DataLoader/albumentations |
| **Compatibility** | Limited augmentation library vs. albumentations |
| **Colab support** | Works, but requires matching CUDA version |
| **Our bottleneck** | GPU compute (not data loading) — 5K images fit in memory |

**Verdict: Skip for this project.** DALI shines with 100K+ images and heavy augmentation. With 5K images and `num_workers=2`, our DataLoader isn't the bottleneck. The complexity cost isn't justified.

---

## 17.4 Technique 2: torch.compile() (PyTorch 2.0+)

### What Is It?
`torch.compile()` uses TorchDynamo + Triton to JIT-compile your model into optimised fused kernels. Free 10-30% speedup with one line of code.

### Implementation

```python
# PyTorch 2.0+ (available in Colab)
model = TamperingDetector(encoder_name='efficientnet-b1').to(device)

# Compile the model — one line!
model = torch.compile(model, mode='reduce-overhead')
# Modes:
#   'default'         — balanced (good for most cases)
#   'reduce-overhead' — minimises CPU overhead (best for smaller models)
#   'max-autotune'    — tries many kernel variants (slower first run, fastest steady-state)
```

### How It Works
1. TorchDynamo captures the Python execution graph
2. Converts it to FX intermediate representation
3. Triton compiler generates fused CUDA kernels
4. First batch is slow (compilation); subsequent batches are 10-30% faster

### Caveats
- First forward pass triggers compilation (~30-60 seconds)
- Dynamic shapes (different batch sizes) trigger recompilation
- Not all ops are supported (custom CUDA kernels may fall back to eager)
- Debugging is harder (stack traces reference compiled code)

### Should You Use torch.compile()?

**Verdict: Yes, if on PyTorch 2.0+.** One line of code, free 10-30% speedup, minimal risk. Add it after model creation and before the training loop.

```python
# Safe pattern: try compile, fall back gracefully
try:
    model = torch.compile(model, mode='reduce-overhead')
    print("Model compiled successfully (PyTorch 2.0+ optimisation)")
except Exception as e:
    print(f"torch.compile not available: {e}. Running in eager mode.")
```

---

## 17.5 Technique 3: Optimised DataLoader Configuration

### Beyond Basic Settings

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=2,
    # Advanced:
    multiprocessing_context='fork',  # 'spawn' is safer but slower on Linux
)
```

### Non-Blocking GPU Transfers

```python
# In training loop — overlap data transfer with computation
for images, masks in train_loader:
    # non_blocking=True: starts CPU→GPU transfer asynchronously
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    
    # GPU transfer happens in parallel with any remaining CPU work
    # By the time forward() runs, data is on GPU
    with autocast('cuda'):
        logits = model(images)
```

`non_blocking=True` only works with `pin_memory=True`. The data transfer is overlapped with CPU work — saves ~5% per batch.

---

## 17.6 Technique 4: Efficient Memory Management

### set_to_none vs. zero_grad

```python
# Slow: allocates zero tensor and copies
optimizer.zero_grad()

# Fast: sets gradients to None (garbage collected)
optimizer.zero_grad(set_to_none=True)  # ~5% memory savings
```

### Inference Optimisations

```python
@torch.no_grad()          # Disables gradient computation entirely
@torch.inference_mode()    # Even more aggressive — disables autograd tracking
def predict(model, images):
    return model(images)
```

`@torch.inference_mode()` is faster than `@torch.no_grad()` because it also disables version counting and tensor metadata tracking.

### Selective Gradient Computation

```python
# Freeze encoder for first N epochs (transfer learning warmup)
for param in model.segmentation_model.encoder.parameters():
    param.requires_grad = False

# After N epochs, unfreeze
for param in model.segmentation_model.encoder.parameters():
    param.requires_grad = True
```

---

## 17.7 Technique 5: Efficient Loss Computation

### Avoid Redundant Sigmoid Calls

```python
# BAD: sigmoid computed twice (loss + metrics)
loss_bce = F.binary_cross_entropy(torch.sigmoid(logits), target)
pred_probs = torch.sigmoid(logits)  # Recomputed!

# GOOD: Use BCEWithLogitsLoss (fused sigmoid + BCE)
loss_bce = F.binary_cross_entropy_with_logits(logits, target)
pred_probs = torch.sigmoid(logits)  # Only compute once, for metrics
```

### In-Place Operations

```python
# BAD: creates new tensor
x = x + residual

# GOOD: modifies in-place (saves memory allocation)
x += residual
# or
x.add_(residual)
```

---

## 17.8 Technique 6: Parallel API Calls (Non-Training)

For tasks like dataset download, metric computation, or logging — use async/parallel execution:

```python
import concurrent.futures

def download_with_parallel(urls, max_workers=4):
    """Download multiple files in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            result = future.result()
            print(f"Downloaded: {url}")

# Parallel metric computation across test images
from multiprocessing import Pool

def compute_f1_for_image(args):
    pred, gt, threshold = args
    return compute_pixel_f1_safe(pred, gt, threshold)[0]

with Pool(processes=2) as pool:
    f1_scores = pool.map(
        compute_f1_for_image,
        [(p, g, 0.5) for p, g in zip(all_preds, all_gts)]
    )
```

---

## 17.9 Technique 7: Mixed-Precision Nuances

### BFloat16 vs. Float16

| Type | Range | Precision | T4 Support |
|------|-------|-----------|------------|
| FP16 | ±65,504 | 3-4 decimal digits | ✅ Tensor Cores |
| BF16 | ±3.4×10³⁸ | 2-3 decimal digits | ❌ T4 doesn't support BF16 |

On T4, use FP16 (the default for `autocast('cuda')`). BF16 requires Ampere+ (A100, RTX 3090+).

### Channel-Last Memory Format

```python
# Convert model and data to channels-last format
# PyTorch's default: NCHW (batch, channels, height, width)
# Channels-last: NHWC — matches cuDNN's preferred format for convolutions

model = model.to(memory_format=torch.channels_last)

# In training loop:
images = images.to(device, memory_format=torch.channels_last)
```

**Effect**: 10-20% speedup for convolutional models on NVIDIA GPUs. The data layout matches cuDNN's internal format, avoiding costly memory reformatting.

---

## 17.10 Optimisation Priority for This Project

Ranked by **effort vs. impact**:

| Priority | Technique | Effort | Speed Gain | Use? |
|----------|----------|--------|------------|------|
| 1 | AMP (already in Doc 05) | Low | 40-50% | ✅ Yes |
| 2 | `torch.compile()` | 1 line | 10-30% | ✅ Yes |
| 3 | Channels-last memory | 2 lines | 10-20% | ✅ Yes |
| 4 | `non_blocking=True` transfers | 1 change | 5% | ✅ Yes |
| 5 | `set_to_none=True` | Already done | 5% | ✅ Yes |
| 6 | `@torch.inference_mode()` for eval | 1 line change | 5% | ✅ Yes |
| 7 | Freeze encoder for warmup epochs | 5 lines | Convergence, not speed | ⚠️ Optional |
| 8 | DALI GPU pipeline | Major refactor | 30-50% | ❌ No (overkill) |
| 9 | Gradient checkpointing | 1 line | Negative (trades speed for memory) | ❌ Only if OOM |

### Quick Wins to Add to Your Notebook

```python
# Add these 4 lines after model creation — free performance boost
model = model.to(device)
model = model.to(memory_format=torch.channels_last)  # Channels-last
try:
    model = torch.compile(model, mode='reduce-overhead')  # JIT compile
except Exception:
    pass  # Graceful fallback for older PyTorch

# And in training loop:
images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
masks = masks.to(device, non_blocking=True)
```
