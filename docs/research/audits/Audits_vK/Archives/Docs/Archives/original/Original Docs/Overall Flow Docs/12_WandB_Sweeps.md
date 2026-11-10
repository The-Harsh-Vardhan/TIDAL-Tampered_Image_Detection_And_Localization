# 12. W&B Sweeps — Hyperparameter Search Guide

## 12.1 What Are Sweeps?

W&B Sweeps is an **automated hyperparameter optimisation** system. Instead of manually trying LR=1e-3, then LR=1e-4, then LR=5e-4, you define a search space and a strategy, and Sweeps runs the experiments for you — logging everything to the same dashboard.

### How It Works

```
┌─────────────────────┐
│  Sweep Controller    │ ← Hosted by W&B (cloud)
│  (decides next run)  │
└─────────┬───────────┘
          │ Suggests hyperparams
          ▼
┌─────────────────────┐
│  Sweep Agent         │ ← Runs on YOUR Colab GPU
│  (trains one model)  │
│  Reports metrics     │──→ W&B Dashboard
└─────────────────────┘
          │
          ▼ Repeat with new params
```

1. You define a **sweep configuration** (search space + strategy)
2. W&B creates a **sweep controller** in the cloud
3. You start a **sweep agent** on your GPU — it fetches the next hyperparameter combo, trains, reports metrics
4. The controller uses reported metrics to decide the next combo (if Bayesian)
5. Repeat until budget exhausted

---

## 12.2 Why Use Sweeps for This Project?

### The Manual Tuning Problem
With our model, the key hyperparameters are:
- Encoder LR (1e-5 to 1e-3)
- Decoder LR (1e-4 to 1e-2)
- Loss weights (BCE, Dice, Edge — 3 values)
- Weight decay (1e-5 to 1e-3)
- Batch size / accumulation steps
- Threshold for binarisation

That's a 6+ dimensional space. Manual grid search with 3 values each = 729 combinations. Even 2 values each = 64 runs × 3.5 hours = 224 hours. Not feasible.

### What Sweeps Gives You
- **Bayesian optimisation**: Learns from completed runs → suggests smarter next runs
- **Early termination**: Kill bad runs after 5 epochs instead of wasting 50
- **Parallel agents**: Run on multiple Colab instances simultaneously
- **Visualisation**: Parallel coordinates plot, importance ranking, correlation analysis

---

## 12.3 Sweep Configuration

### Define the Search Space

```python
sweep_config = {
    "method": "bayes",                # bayes | random | grid
    "metric": {
        "name": "val/pixel_f1",       # What to optimise
        "goal": "maximize"
    },
    "parameters": {
        "lr_encoder": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3
        },
        "lr_decoder": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 5e-3
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3
        },
        "loss_bce_weight": {
            "values": [0.5, 1.0, 1.5]
        },
        "loss_dice_weight": {
            "values": [0.5, 1.0, 1.5, 2.0]
        },
        "loss_edge_weight": {
            "values": [0.0, 0.25, 0.5, 1.0]
        },
        "accumulation_steps": {
            "values": [2, 4, 8]       # Effective BS = 4 × this
        },
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,                # Min epochs before killing
        "eta": 3,                     # Aggressiveness (3 = moderate)
        "s": 2
    }
}
```

### Search Strategy Comparison

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| **`grid`** | Exhaustive: tries every combination | Small discrete spaces (≤50 combos) |
| **`random`** | Uniform random from search space | Large spaces; good baseline |
| **`bayes`** | Gaussian Process models metric surface; picks promising regions | **Best for our case** — learns from partial results |

---

## 12.4 Training Function for Sweeps

The sweep agent calls your training function with different configs each time. Structure it as follows:

```python
def train_sweep():
    """
    Single training run — called by the sweep agent.
    Hyperparameters come from wandb.config.
    """
    # Init run (config is auto-populated by the sweep agent)
    run = wandb.init()
    config = wandb.config
    
    # ========== Setup ==========
    # Model (same every time — architecture isn't swept)
    model = TamperingDetector(
        encoder_name='efficientnet-b1',
        encoder_weights='imagenet'
    ).to(device)
    
    # Loss (weights from sweep config)
    criterion = HybridLoss(
        bce_weight=config.loss_bce_weight,
        dice_weight=config.loss_dice_weight,
        edge_weight=config.loss_edge_weight,
    )
    
    # Optimizer (LRs from sweep config)
    optimizer = torch.optim.AdamW([
        {'params': model.channel_reducer.parameters(), 'lr': config.lr_decoder},
        {'params': model.segmentation_model.encoder.parameters(), 'lr': config.lr_encoder},
        {'params': model.segmentation_model.decoder.parameters(), 'lr': config.lr_decoder},
        {'params': model.segmentation_model.segmentation_head.parameters(), 'lr': config.lr_decoder},
    ], weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SWEEP_EPOCHS, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    accumulation_steps = config.accumulation_steps
    
    # ========== Training Loop ==========
    SWEEP_EPOCHS = 15  # Shorter than full training — enough to compare
    best_f1 = 0
    
    for epoch in range(SWEEP_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            with autocast('cuda'):
                logits = model(images)
                loss, loss_dict = criterion(logits, masks)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        
        # Validate
        val_loss, val_f1, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        best_f1 = max(best_f1, val_f1)
        
        # Log to W&B (sweep controller monitors these)
        wandb.log({
            "epoch": epoch,
            "val/loss": val_loss,
            "val/pixel_f1": val_f1,
            "val/pixel_iou": val_iou,
            "val/best_f1": best_f1,
        })
    
    wandb.finish()
```

---

## 12.5 Running the Sweep

### Step 1: Create the Sweep

```python
sweep_id = wandb.sweep(
    sweep_config,
    project="bigvision-tampering-detection"
)
print(f"Sweep ID: {sweep_id}")
```

### Step 2: Start the Agent

```python
# Run N trials
wandb.agent(
    sweep_id,
    function=train_sweep,
    count=20           # Number of runs to try
)
```

### Step 3: Monitor

Open `https://wandb.ai/YOUR_USERNAME/bigvision-tampering-detection/sweeps/SWEEP_ID` to see:
- **Parallel coordinates plot** — which param ranges produce high F1
- **Parameter importance** — ranked by correlation with metric
- **Run table** — all trials sorted by performance

---

## 12.6 Practical Sweep Strategy for This Project

### Phase 1: Coarse Search (Random, 10 runs × 15 epochs)
- Wide parameter ranges
- Goal: eliminate clearly bad regions

### Phase 2: Fine Search (Bayesian, 10 runs × 15 epochs)
- Narrow ranges around Phase 1 winners
- Goal: find near-optimal combo

### Phase 3: Full Training (1 run × 50 epochs)
- Use the best hyperparameters from Phase 2
- Full training with early stopping and checkpointing
- This is the final submission model

### Time Budget

| Phase | Runs | Epochs/Run | Time/Run | Total |
|-------|------|-----------|----------|-------|
| Coarse | 10 | 15 | ~50 min | ~8 hr |
| Fine | 10 | 15 | ~50 min | ~8 hr |
| Final | 1 | 50 | ~3.5 hr | ~3.5 hr |
| **Total** | **21** | — | — | **~19.5 hr** |

This fits within 2-3 Colab sessions. With Colab Pro (A100), the time halves.

---

## 12.7 Interpreting Sweep Results

### Parallel Coordinates
W&B generates a parallel coordinates plot showing how each hyperparameter value maps to the target metric. Look for:
- **Convergence bands**: if all top runs have `lr_encoder` in [5e-5, 2e-4], that's your range
- **Uncorrelated params**: if `edge_weight` shows no pattern, it doesn't matter much
- **Conflicting signals**: if high LR is good early but bad late, your training is unstable

### Parameter Importance
W&B automatically ranks parameters by their correlation with the metric:
```
1. lr_encoder:      0.72 importance
2. loss_dice_weight: 0.54 importance
3. lr_decoder:      0.38 importance
4. weight_decay:    0.12 importance   ← Doesn't matter much
```

---

## 12.8 Should You Use Sweeps for This Project?

| Factor | Assessment |
|--------|-----------|
| **Time available** | 1 week — tight, but 2 Colab sessions for sweeps is feasible |
| **Effort** | Low — config is ~20 lines, agent is ~5 lines |
| **Impact on results** | Medium — the right LR alone can swing F1 by 5-10% |
| **Evaluator impression** | Very strong — shows systematic engineering, not trial-and-error |
| **Risk** | Low — if sweeps take too long, use the best result found so far |

**Verdict: Do it if time permits.** At minimum, do a 5-run random sweep (15 epochs each = ~4 hours). The insight into which hyperparameters matter most is valuable even with few runs.

---

## 12.9 Quick Alternative: Manual Mini-Sweep

If W&B Sweeps feels like overkill, do a structured manual search:

```python
# 3 priority experiments (run sequentially)
experiments = [
    {"name": "baseline",    "lr_enc": 1e-4, "lr_dec": 1e-3, "edge_w": 0.5},
    {"name": "high-lr",     "lr_enc": 5e-4, "lr_dec": 3e-3, "edge_w": 0.5},
    {"name": "no-edge",     "lr_enc": 1e-4, "lr_dec": 1e-3, "edge_w": 0.0},
]
# Run each for 15 epochs, compare val F1, pick winner for full 50-epoch run
```

This takes ~2.5 hours and answers the two biggest questions: "Is my LR right?" and "Does edge loss help?"
