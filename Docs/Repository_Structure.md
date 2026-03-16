# Repository Structure — Tampered Image Detection & Localization

**Project:** Tampered Image Detection & Localization using UNet + Pretrained Encoders with ELA Preprocessing
**Best Model:** vR.P.19 — Multi-Quality RGB ELA 9-Channel (Pixel F1 = 0.7965)
**Dataset:** CASIA 2.0 Image Tampering Detection Dataset (12,614 images)

---

## Quick Start — Submission Artifacts

For reviewers, everything needed is in one folder:

```
submission/
├── final_notebook.ipynb          # vR.P.19 — best performing model (Pixel F1 = 0.7965)
├── submission_report.md          # Full submission report
├── model_weights_link.txt        # How to load trained weights
└── Internship Assignment...pdf   # Original assignment brief
```

---

## Directory Layout

### `submission/`
**Purpose:** Final deliverables for review. Contains the best notebook (vR.P.19), submission report, and model weights reference.

### `Notebooks/`
All experiment notebooks organized by research track.

```
Notebooks/
├── final/
│   ├── tampered_detection_final.ipynb    # Copy of best model for easy access
│   └── best_notebooks/                   # Top 5 notebooks with docs
│       ├── 1. vR.P.10 — CBAM Attention (Best Localization)/
│       ├── 2. vR.P.7 — Extended Training (2nd Best Localization)/
│       ├── 3. vR.P.8 — Progressive Unfreeze (Best Image Accuracy)/
│       ├── 4. vR.P.3 — ELA Breakthrough (Most Impactful Innovation)/
│       └── 5. vR.1.6 — Deeper CNN (Best Classification)/
│
└── research_tracks/
    ├── v0x/documentation_experiments/    # Early approach exploration (Approaches 1-4)
    ├── vK/kaggle_baseline_experiments/   # Kaggle baseline track (vK.1–vK.12)
    │   ├── source/                       # Source notebooks (25 files)
    │   ├── runs/                         # Kaggle run outputs (22 files)
    │   ├── reference/                    # Reference notebooks from Kaggle (16 files)
    │   ├── scripts/                      # vK build scripts
    │   └── helper_functions/             # vK helper functions
    ├── vR/research_paper_experiments/    # Research paper CNN track (vR.0–vR.1.7)
    │   ├── source/                       # Source notebooks (11 files)
    │   ├── runs/                         # Kaggle run outputs (15+ files)
    │   └── reference/                    # Reference ELA/CNN notebooks
    └── vR.P/pretrained_ablation_experiments/  # Pretrained ablation track (vR.P.0–vR.P.41)
        ├── source/                       # Source notebooks (41 files)
        ├── runs/                         # Kaggle run outputs (22 files)
        ├── per_version_docs/             # Per-experiment documentation (36 dirs)
        └── best_notebooks/               # Curated best from vR.P track
```

### `experiments/`
Experiment tracking and W&B integration.

```
experiments/
├── wandb_runs/                     # 37 W&B exported run notebooks
│   ├── vr-p-0-pretrained-resnet-34-unet.ipynb
│   ├── vr-p-19-multi-quality-rgb-ela-9-channels.ipynb
│   ├── vr-p-30-1-multi-quality-ela-cbam-attention.ipynb
│   └── ... (37 total)
└── wandb_tracking/                 # W&B tracking infrastructure
    ├── configs/                    # W&B config files
    ├── datasets/                   # Kaggle dataset runners + source notebooks
    ├── experiment_lists/           # Experiment lists
    ├── leaderboard/                # Leaderboard notebooks
    ├── runners/                    # W&B runner notebooks
    └── utilities/                  # Tracking utilities
```

### `Docs/`
All documentation, split into submission and research tiers.

```
Docs/
├── submission_report/              # Clean submission documents
│   ├── Submission_Report.md        # Final submission report
│   ├── Submission_Report.latex     # LaTeX version
│   ├── Report_reference.latex      # Reference LaTeX
│   ├── Detailed_Technical_Report.md
│   ├── Internship Assignment...pdf # Original assignment brief
│   └── figures/                    # Report figures
│
└── research_docs/                  # Full research documentation
    ├── Research_Report.md          # Academic-style research paper
    ├── Research_Journey_and_Experimentation.md  # Full experiment narrative
    ├── Assignment.md               # Assignment analysis
    ├── Project_Experiment_Evolution_Report.md
    │
    ├── ablation_study/             # Ablation study documents
    │   ├── WandB_Run_Audit.md      # Comprehensive run audit with /100 scores
    │   ├── ablation_master_plan.md # Full experiment roadmap
    │   ├── experiment_leaderboard.md
    │   ├── experiment_tracking_table.md
    │   └── ... (14 files)
    │
    ├── audits/                     # Audit reports across all tracks
    │   ├── Audits_vK/              # vK track audits
    │   ├── Audit_new_runs_vK11/    # vK.11 run audits
    │   └── vR ETASR audit files    # vR track audit files
    │
    ├── papers/                     # Research paper analyses
    │   ├── Paper_Analyses/
    │   ├── Papers/
    │   └── Research_Paper_Analysis_Report.md
    │
    ├── vR_docs/                    # vR track documentation
    │   ├── DocsR1_ETASR/           # Master ETASR docs (6 files)
    │   ├── docs-vR.1.1/ to docs-vR.1.7/
    │   └── docs-vR.P.0/ to docs-vR.P.3/
    │
    ├── vK_docs/                    # vK track documentation
    │   ├── Docs9_Notebook_Roast/
    │   ├── Docs_vK4_Kill_Review/
    │   ├── Docs_vK4_Notebook_Audit/
    │   ├── Interview Prep/
    │   └── v11/
    │
    └── reference/                  # External reference documents
```

### `scripts/`
All Python scripts used for notebook generation and utilities.

```
scripts/
├── build_research_notebook.py      # Notebook builder
├── create_p41.py                   # P.41 generator
├── fix_ela_quality.py              # ELA quality fix script
├── gen_docs.py                     # Documentation generator
├── gen_runners.py                  # W&B runner generator
├── upgrade_etasr_notebooks.py      # ETASR notebook upgrader
├── reference_code/                 # External reference implementations
│   └── DocAuth-master/             # Document authentication reference
└── peerj_review_codes/             # PeerJ review code reference
```

### `models/`
Model weights and analysis.

```
models/
├── pretrained_weights/             # Pretrained model analysis docs
│   ├── 01_Feasibility_Analysis.md
│   ├── 02_Architecture_Comparison.md
│   ├── 03_Implementation_Plan.md
│   └── 04_ELA_Compatibility_Analysis.md
└── checkpoints/
    └── results/                    # Training results/outputs
```

### `configs/`
Training configurations and hyperparameter sweep definitions.

```
configs/
└── training_configs/
    ├── sweep.yaml                  # W&B sweep configuration
    ├── sweep_runner.py             # Sweep runner script
    ├── convert_notebooks.py        # Notebook conversion utility
    ├── path_config.py              # Path configuration
    └── modules/                    # Sweep modules
```

### `data_access/`
Dataset information and access utilities.

```
data_access/
├── Datasets-Comparison.md          # Dataset comparison analysis
├── Image Forgery Datasets.md       # Dataset catalog
├── Links-to-Datasets.md            # Dataset download links
├── dataset_download_scripts/       # Download automation
└── kaggle_workspace/               # Kaggle workspace template
    ├── input/
    └── working/
```

### `figures/`
Visual assets for reports and documentation.

```
figures/
└── visual_results/                 # Model prediction visualizations
```

### `_archive/`
Historical artifacts preserved for lineage. Contains older documentation versions (v1-v9), archived notebooks, old audit reports, and early project files.

### `Approach 5 FakeShield Simplification/`
Separate git repository for FakeShield approach exploration. Cannot be moved due to independent .git history.

---

## Research Tracks

### Track 1: v0x — Early Exploration
Early approaches exploring different strategies (literature review, Kaggle baselines, research paper recreation, pretrained models, FakeShield).

### Track 2: vK — Kaggle Baseline (vK.1–vK.12)
Started from a Kaggle notebook baseline. Iteratively improved through documentation, code quality, architecture changes, and training strategy. **Best: vK.12.0**

### Track 3: vR — Research Paper CNN (vR.0–vR.1.7)
Reproduced the ETASR paper's ELA+CNN classification approach. Systematic ablation study varying evaluation, augmentation, class weights, BatchNorm, LR scheduler, architecture depth, and pooling. **Best: vR.1.6 (90.23% test accuracy)**

### Track 4: vR.P — Pretrained Ablation (vR.P.0–vR.P.41)
The primary research track. UNet with pretrained ResNet-34 encoder for pixel-level localization. 41+ experiments systematically ablating:
- Input preprocessing (RGB → ELA → Multi-Q RGB ELA)
- Encoder architecture (ResNet-34/50, EfficientNet-B0)
- Freeze strategy (frozen → progressive unfreeze)
- Loss functions (BCE+Dice → Focal+Dice)
- Attention mechanisms (CBAM)
- Training schedule (25 → 50 epochs)
- Alternative forensic features (DCT, YCbCr, Noiseprint)

**Best: vR.P.19 (Pixel F1 = 0.7965, Multi-Quality RGB ELA 9ch)**

---

## Experiment Lineage

```
v0x (Exploration)
 ├── Approach 1: Literature Review
 ├── Approach 2: Kaggle Baseline → vK track
 ├── Approach 3: Research Paper CNN → vR track
 ├── Approach 4: Pretrained Model → vR.P track
 └── Approach 5: FakeShield (abandoned)

vK track: vK.1 → vK.3 → vK.7 → vK.10 → vK.11 → vK.12

vR track: vR.0 → vR.1.1 → vR.1.2(X) → vR.1.3 → vR.1.4 → vR.1.5 → vR.1.6(BEST) → vR.1.7

vR.P track: P.0 → P.1 → P.2 → P.3(ELA breakthrough) → P.4–P.18 (ablations)
            → P.19(BEST) → P.20–P.28 (feature experiments)
            → P.30–P.30.4 (combination experiments)
            → P.40.1–P.41 (custom encoder experiments)
```

---

## Key Files for Reviewers

| File | Location | Purpose |
|------|----------|---------|
| Final notebook | `submission/final_notebook.ipynb` | Best model (vR.P.19) |
| Submission report | `submission/submission_report.md` | Summary for reviewers |
| Model weights | `submission/model_weights_link.txt` | How to load weights |
| Run audit | `Docs/research_docs/ablation_study/WandB_Run_Audit.md` | All runs scored /100 |
| Ablation plan | `Docs/research_docs/ablation_study/ablation_master_plan.md` | Full experiment roadmap |
| Research paper | `Docs/research_docs/Research_Report.md` | Academic-style report |
| Experiment journey | `Docs/research_docs/Research_Journey_and_Experimentation.md` | Full narrative |
