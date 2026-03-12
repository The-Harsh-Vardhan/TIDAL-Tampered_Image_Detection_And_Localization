# 20. Databricks — What, Why, How & Should We Use It?

## 20.1 What Is Databricks?

Databricks is a **unified data analytics and AI platform** built on top of Apache Spark. Originally created by the founders of Spark at UC Berkeley, it's now an enterprise-grade cloud service.

### Core Components

| Component | What It Does |
|-----------|-------------|
| **Apache Spark** | Distributed compute engine — process TB/PB of data across clusters |
| **Delta Lake** | ACID-compliant data lake storage (versioned Parquet tables) |
| **MLflow** | Experiment tracking, model registry, deployment (open-source) |
| **Databricks Notebooks** | Collaborative notebooks (Python, SQL, R, Scala) with cluster-backed execution |
| **Unity Catalog** | Governance: permissions, lineage, auditing across data & ML assets |
| **Workflows** | Orchestration: schedule and chain jobs (ETL → training → deployment) |

### How It Fits in the ML Landscape

```
┌──────────────────────────────────────────────────────────┐
│                    Databricks Platform                     │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Data Lake   │  │  Spark       │  │  MLflow        │  │
│  │  (Delta)     │→ │  (Compute)   │→ │  (Track/Deploy)│  │
│  │  S3/ADLS/GCS │  │  Distributed │  │  Model Registry│  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Notebooks   │  │  Workflows   │  │  Unity Catalog │  │
│  │  (IDE)       │  │  (Orchestrate)│  │  (Governance)  │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 20.2 What Databricks Excels At

### 1. Massive-Scale Data Processing
```python
# Processing 10M+ images' metadata in seconds:
df = spark.read.format("delta").load("s3://company-data-lake/images/metadata")

# Filter, aggregate, and analyze across 10M rows
tampered_stats = (
    df.filter(df.label == "tampered")
      .groupBy("tampering_type")
      .agg(
          count("*").alias("count"),
          avg("image_size_kb").alias("avg_size"),
          avg("confidence_score").alias("avg_confidence"),
      )
      .orderBy("count", ascending=False)
)
tampered_stats.display()  # Rich visualization in Databricks notebooks
```

### 2. Distributed Model Training (Multi-GPU / Multi-Node)
```python
# Databricks integrates with PyTorch Lightning + Horovod for distributed training
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

# On a Databricks cluster with 4× A100 GPUs:
trainer = Trainer(
    accelerator="gpu",
    devices=4,
    strategy=DDPStrategy(),
    max_epochs=50,
)
trainer.fit(model, train_loader, val_loader)
```

### 3. MLflow Integration (Built-In)
```python
import mlflow
import mlflow.pytorch

# MLflow is pre-installed and auto-configured in Databricks
with mlflow.start_run():
    mlflow.log_params({"encoder": "efficientnet-b1", "lr": 1e-3})
    
    # ... training loop ...
    
    mlflow.log_metrics({"val_f1": 0.72, "val_iou": 0.58})
    mlflow.pytorch.log_model(model, "tampering-detector")
    
    # Model automatically appears in Databricks Model Registry
```

### 4. Feature Store & Data Versioning
```python
# Delta Lake gives you Git-like versioning for datasets
df.write.format("delta").mode("overwrite").save("/delta/casia_v2_clean")

# Time travel: query previous versions
spark.read.format("delta").option("versionAsOf", 3).load("/delta/casia_v2_clean")
```

---

## 20.3 How It Compares to Our Setup

| Dimension | Our Setup (Colab + Drive) | Databricks |
|-----------|--------------------------|-----------|
| **Compute** | 1× T4 GPU (free) | Multi-node clusters (A100s, etc.) |
| **Dataset Size** | ~5K images, < 2 GB | TB to PB scale |
| **Data Storage** | Google Drive / Kaggle | Delta Lake (S3/ADLS/GCS) |
| **Experiment Tracking** | W&B (Doc 11) | MLflow (built-in) |
| **Notebooks** | Google Colab | Databricks Notebooks |
| **Cost** | Free (Colab) | $0.40-$2.00+/DBU-hour (cloud provider + Databricks fee) |
| **Team Size** | Solo | Teams of 5-500+ |
| **Setup Time** | 0 minutes | Hours-days (workspace, cluster config, IAM) |
| **Orchestration** | Manual notebook runs | Workflows (scheduled, chained) |

---

## 20.4 When Databricks Makes Sense

### Use Databricks When:

1. **Dataset is massive** (100K+ images, TB+ of data)
   - Distributed Spark processing is genuinely needed
   - Complex ETL pipelines across multiple data sources

2. **Team collaboration** (5+ data scientists/engineers)
   - Shared notebooks, shared clusters, shared model registry
   - Unity Catalog for access control and governance

3. **Production ML pipelines**
   - Automated: ingest → preprocess → train → evaluate → deploy
   - Scheduled retraining with Workflows
   - Model serving with real-time endpoints

4. **Regulatory/compliance needs**
   - Audit trails, data lineage, role-based access
   - Unity Catalog governance layer

5. **Multi-cloud enterprise**
   - Available on AWS, Azure, GCP with consistent API

### Don't Use Databricks When:

1. **Small dataset** (< 50K samples) — Spark overhead hurts more than helps
2. **Solo project** — collaboration features are wasted
3. **One-off experiment** — setup cost isn't justified
4. **Budget-constrained** — minimum useful cluster costs ~$1-2/hour
5. **Simple training loop** — single GPU is sufficient

---

## 20.5 Should We Use Databricks for This Project?

### Assessment

| Criterion | Our Project | Databricks Designed For |
|-----------|------------|------------------------|
| Dataset | ~5K images, ~1.5 GB | 1M+ images, TB+ |
| Team | 1 person | 5-500 people |
| Compute need | 1× T4 GPU | Multi-node GPU clusters |
| Pipeline complexity | Load → Train → Evaluate | ETL → Feature Store → Train → Deploy → Monitor |
| Budget | $0 (free Colab) | Enterprise budgets |
| Duration | 1-week assignment | Ongoing production system |

### Verdict: **No — Not for This Project**

Databricks is a powerful enterprise platform, but it solves problems we don't have:

- **We don't need distributed computing** — 5K images fits in a single GPU's memory
- **We don't need Spark** — pandas + PyTorch DataLoader handles our data volume trivially
- **We don't need Delta Lake** — our dataset is static and small
- **We don't need MLflow** — W&B (Doc 11) is more feature-rich for experiment tracking
- **We don't need orchestration** — we run one notebook once
- **We don't need governance** — solo project, no compliance requirements
- **Setup overhead is massive** — hours of configuration for a 1-week deadline

---

## 20.6 What We Use Instead (And Why It's Better for Us)

| Databricks Feature | Our Equivalent | Why It's Better for Us |
|-------------------|---------------|----------------------|
| Spark DataFrames | pandas / PyTorch Dataset | No overhead, simpler API |
| Delta Lake | Google Drive + Kaggle | Free, zero config |
| MLflow | W&B (Doc 11) | Better UI, sweep support, free tier |
| MLflow Model Registry | HF Hub (Doc 14) | Better sharing, model cards, community |
| Databricks Notebooks | Google Colab | Free T4 GPU, assignment-specified |
| Workflows | Manual runs | We only run once |
| Unity Catalog | Not needed | Solo project |

---

## 20.7 When You WOULD Learn Databricks

If you move beyond this assignment and into a role that involves:

1. **Data engineering at scale**: ETL pipelines processing millions of records daily
2. **Enterprise ML platforms**: Deploying models in production with monitoring, A/B testing
3. **Team-based ML**: Sharing experiments, models, and datasets across a team
4. **Consulting / cloud architecture**: Many enterprises standardize on Databricks

### Learning Path (Post-Assignment)

```
Phase 1: Fundamentals
├── Databricks Community Edition (free tier)
├── "Databricks Certified Data Engineer Associate" course
├── Build a small ETL pipeline (CSV → Delta → SQL analytics)
└── Run MLflow tracking for a simple model

Phase 2: ML on Databricks
├── Convert a PyTorch training script to Databricks notebook
├── Use MLflow for experiment tracking and model registry
├── Set up a Workflow: data_prep → train → evaluate
└── Deploy a model serving endpoint

Phase 3: Production
├── Feature Store integration
├── Unity Catalog setup
├── Multi-node distributed training
└── Real-time inference pipeline
```

### Databricks Community Edition

For free experimentation:
- Single-node cluster (1 driver, no workers)
- 15 GB memory, limited compute
- MLflow built-in
- No cloud storage (local DBFS only)
- Good for learning, not for production

---

## 20.8 Quick Comparison: Databricks vs Our Full Stack

```
OUR STACK (This Project)                DATABRICKS EQUIVALENT
========================                =====================
Google Colab (notebook)       ←→        Databricks Notebook
T4 GPU (free)                 ←→        GPU Cluster ($$$)
Kaggle (dataset download)     ←→        Unity Catalog + Delta Lake
W&B (experiment tracking)     ←→        MLflow
W&B Sweeps (HPO)              ←→        Hyperopt on Spark
HF Hub (model storage)        ←→        MLflow Model Registry
HF Spaces (deployment)        ←→        Model Serving Endpoints
Google Drive (persistence)    ←→        DBFS / Cloud Storage (S3/ADLS)
```

Our stack is **free**, **simple**, and **purpose-built for this assignment**.
Databricks is **powerful**, **enterprise-grade**, and **overkill for a 5K-image solo project**.

---

## 20.9 One Sentence Summary

> Databricks is the right tool when you have big data, big teams, and big budgets — none of which apply to this 1-week solo Colab assignment, so we skip it and use Colab + W&B + HF Hub instead.
