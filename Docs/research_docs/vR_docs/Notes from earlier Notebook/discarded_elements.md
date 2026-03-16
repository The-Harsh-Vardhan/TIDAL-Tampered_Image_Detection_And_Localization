# Discarded Elements from vK.12.0 — What NOT to Reuse

| Field | Value |
|-------|-------|
| Source | vK.12.0 Image Detection and Localisation.ipynb |
| Reason | Architecture failure, PyTorch-specific APIs, overengineering, or disproven by experiments |
| Date | 2026-03-15 |

---

## Category 1: Architecture Decisions That Caused the Failure

### 1.1 Dual-Head Architecture (Segmentation + Classification)

| Field | Value |
|-------|-------|
| What | UNet decoder for pixel masks + classification head for binary label, sharing a single ResNet34 encoder |
| Why it failed | Multi-task loss balancing was never solved. The classification head dominated training (trivial task), starving the segmentation head of gradient signal. vK.11.4 through vK.12.0b all produced near-constant segmentation output (pixel-AUC=0.50 = random). |
| Why not reuse | vR.P.0 correctly uses a single-task UNet for segmentation only. Classification is derived from mask prediction (any nonzero mask = tampered). |
| Lesson | Do not combine fundamentally different tasks (dense prediction + global classification) in a shared-encoder architecture without extensive loss-weight tuning. |
| Reconsider | Never for this project. |

---

### 1.2 Complex Multi-Loss Combinations

| Field | Value |
|-------|-------|
| What | BCE + Dice + Edge loss + Classification BCE, with manual loss weights (alpha=1.5, beta=1.0, gamma=0.3) |
| Why it failed | Four loss terms with different scales and gradients created conflicting optimization objectives. The model could not satisfy all losses simultaneously. |
| Why not reuse | vR.P.0 uses BCEDiceLoss only (proven in earlier experiments). Track 1 uses categorical_crossentropy only. Single-loss simplicity. |
| Lesson | More losses ≠ better training. Each additional loss is a hyperparameter to tune. |
| Reconsider | Never. |

---

### 1.3 Edge Detection Auxiliary Task

| Field | Value |
|-------|-------|
| What | Sobel-based edge supervision: used Sobel-X and Sobel-Y filters via F.conv2d to compute GT mask edges, trained a separate prediction branch with BCE loss |
| Why it failed | Edge loss competed with segmentation loss. Edges of tampered regions are often subtle and ambiguous in ELA maps. The edge output learned nothing useful (edge-F1 near 0). |
| Why not reuse | Boundary refinement should be done post-hoc (CRF, morphological operations) not as a training objective. |
| Reconsider | Never. |

---

## Category 2: PyTorch-Specific APIs (Cannot Port to TF/Keras Track 1)

### 2.1 Gradient Accumulation with AMP

| Field | Value |
|-------|-------|
| What | `torch.cuda.amp.autocast()` + `GradScaler` for mixed-precision training with gradient accumulation across mini-batches |
| Why not reuse | TF has `tf.keras.mixed_precision` but with a different API. Gradient accumulation requires custom training loops in TF. Not worth the complexity for Track 1 (batch=32 fits T4 easily). |
| Track 2 note | Could use in Track 2 if VRAM is tight at 384×384. |

---

### 2.2 seed_worker + Generator for Reproducible DataLoaders

| Field | Value |
|-------|-------|
| What | PyTorch DataLoader with `worker_init_fn=seed_worker` and `generator=g` for deterministic multi-worker data loading |
| Why not reuse | TF/Keras uses `tf.random.set_seed()` and `PYTHONHASHSEED` for reproducibility. No equivalent API needed. |
| Alternative | Already handled: `np.random.seed(42); tf.random.set_seed(42); random.seed(42)` in current notebooks. |

---

### 2.3 Albumentations additional_targets

| Field | Value |
|-------|-------|
| What | Albumentations pipeline with `additional_targets={'ela': 'image'}` to apply identical transforms to both RGB and ELA inputs |
| Why not reuse | Albumentations works with PyTorch. For TF/Keras, use `tf.image` operations within a `tf.data.Dataset.map()` call if augmentation is needed. |
| Track 2 note | Directly usable in Track 2 (PyTorch). |

---

## Category 3: Overengineered for Current Project Scale

### 3.1 Multi-Source Dataset Discovery

| Field | Value |
|-------|-------|
| What | Cascade: Kaggle attached → Google Drive → Kaggle API download → manual upload prompt (cells 20–21, ~50 lines) |
| Why not reuse | Project runs exclusively on Kaggle. The auto-discovery in vR.1.1 (`os.walk` looking for Au/ and Tp/) is sufficient. The elaborate fallback system never executed. |
| Lesson | Build for the platform you use, not every possible platform. |

---

### 3.2 Metadata CSV Caching with Staleness Detection

| Field | Value |
|-------|-------|
| What | Save dataset metadata to CSV; on next run, compare row counts to detect staleness; regenerate if stale |
| Why not reuse | CASIA v2.0 is static (never changes). CSV caching saves ~5 seconds on a 10-minute run. Not worth the code complexity. |

---

### 3.3 VRAM-Based Batch Size Auto-Scaling

| Field | Value |
|-------|-------|
| What | Query GPU VRAM → estimate model memory → compute max feasible batch size (cell 26) |
| Why not reuse | Batch size is **FROZEN** in the ablation plan (32 for Track 1, 16 for Track 2). Auto-scaling would violate the single-variable rule by making batch size non-deterministic across GPUs. |
| Lesson | In an ablation study, reproducibility trumps convenience. |

---

### 3.4 W&B Experiment Tracking

| Field | Value |
|-------|-------|
| What | Weights & Biases integration with Kaggle Secrets key lookup, offline fallback mode (cell 60) |
| Why not reuse | Adds an external dependency that can crash (vK.7.5 actually crashed due to W&B). The manual tracking table in ablation_master_plan.md is sufficient for ~10 experiments. W&B is appropriate for 50+ experiments. |
| Lesson | External dependencies are liabilities in time-constrained projects. |

---

## Category 4: Ideas Disproven by Project History

### 4.1 Geometric Augmentation on ELA Maps

| Field | Value |
|-------|-------|
| What | Horizontal flip, vertical flip, rotation ±15° applied to ELA-preprocessed images (cell 44 transforms) |
| Disproven by | **vR.1.2** — accuracy dropped 2.85pp (88.38% → 85.53%), best epoch was epoch 1. Definitively rejected. |
| Root cause | Flatten→Dense(256) memorizes pixel-exact patterns; augmented images appear as entirely new data. ELA signal may also be fragile to rotation fill artifacts. |
| Reconsider | ONLY after vR.1.7 (GAP replaces Flatten), which adds spatial invariance. Even then, prefer JPEG augmentation (Candidate B) over geometric transforms. |

---

### 4.2 Model Complexity Analysis with VRAM Estimation

| Field | Value |
|-------|-------|
| What | Compute per-layer parameter counts, estimate activation memory, project total VRAM via torchinfo (cell 58) |
| Why not reuse | The ETASR CNN is fully characterized (~29.5M params, fits T4 easily). VRAM estimation was valuable for vK.12.0's complex multi-head architecture but is unnecessary for a simple CNN with 8 layers. |

---

## Summary Table

| Element | Category | Reason | Reconsider When |
|---------|----------|--------|-----------------|
| Dual-head architecture | Failure | Multi-task loss balancing unsolved | Never |
| Complex multi-loss (BCE+Dice+Edge+Cls) | Failure | Conflicting gradients | Never |
| Edge detection auxiliary task | Failure | Edge-F1 near 0 | Never |
| AMP + gradient accumulation | PyTorch-specific | Different API in TF | Track 2, if VRAM-limited |
| seed_worker + Generator | PyTorch-specific | TF has different mechanisms | Track 2 only |
| Albumentations additional_targets | PyTorch-specific | Use tf.image for Track 1 | Track 2 only |
| Multi-source dataset discovery | Overengineered | Project uses Kaggle only | Never |
| CSV metadata caching | Overengineered | 5-second savings, static dataset | Never |
| VRAM auto-scaling | Ablation violation | Batch size must be frozen | Never in ablation |
| W&B tracking | Timeline risk | Manual tracking sufficient | After 50+ experiments |
| Geometric augmentation on ELA | Disproven (vR.1.2) | -2.85pp, Flatten memorizes position | After vR.1.7 (GAP) |
| VRAM estimation | Unnecessary | Architecture is simple and known | Never for Track 1 |
