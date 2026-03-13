# 16. DuckDB, Cache Systems & DynamoDB — Database Technologies Assessment

## 16.1 Overview

This document evaluates three database/data-management technologies and whether they belong in your tampering detection project:

| Technology | Type | Primary Use |
|-----------|------|-------------|
| **DuckDB** | Embedded analytical database (OLAP) | Fast SQL on local files (CSV, Parquet, Arrow) |
| **Cache Systems** (Redis, SQLite, filesystem) | Key-value / local storage | Avoid recomputation; speed up data loading |
| **DynamoDB** | Cloud NoSQL database (AWS managed) | Serverless key-value/document store at scale |

---

## 16.2 DuckDB

### What Is It?
DuckDB is an in-process SQL OLAP database — think "SQLite for analytics." It runs inside your Python process with zero server setup and can query Parquet, CSV, Arrow, and Pandas DataFrames directly.

### Strengths

| Feature | Benefit |
|---------|---------|
| Zero installation | `pip install duckdb`; no server, no config |
| SQL on anything | Query Parquet/CSV/Arrow/Pandas with plain SQL |
| Columnar engine | Blazing fast aggregations, filtering, joins |
| In-process | No network latency; embedded in your Python script |
| Integrations | Native Arrow, Pandas, Polars interop |

### Potential Use Cases for This Project

```python
import duckdb

# Example: Analyse dataset metadata with SQL
conn = duckdb.connect()

# Directly query a CSV/Parquet of image metadata
conn.execute("""
    SELECT 
        tampering_type,
        COUNT(*) as count,
        AVG(height) as avg_height,
        AVG(width) as avg_width,
        AVG(mask_coverage_pct) as avg_tampered_area
    FROM 'data/metadata.parquet'
    GROUP BY tampering_type
""").fetchdf()
```

```python
# Query experiment logs
conn.execute("""
    SELECT 
        run_name, 
        MAX(val_f1) as best_f1, 
        argmax(epoch, val_f1) as best_epoch
    FROM 'experiments/logs.parquet'
    GROUP BY run_name
    ORDER BY best_f1 DESC
""").fetchdf()
```

### Should You Use DuckDB in This Project?

| Factor | Assessment |
|--------|-----------|
| **Dataset analysis** | Marginally useful — CASIA has ~5K images; Pandas handles it in milliseconds |
| **Experiment tracking** | W&B already does this better with a GUI |
| **Complexity added** | Another dependency + SQL queries for what's a 3-line Pandas operation |
| **When DuckDB shines** | Datasets with 1M+ rows; multi-GB CSV/Parquet files; ad-hoc SQL exploration |

**Verdict: Skip for this project.** DuckDB is excellent for large-scale data exploration (100K+ rows), but our dataset has ~5K images. Pandas + basic Python is sufficient. However, if you wanted to build a **metadata exploration** cell in the notebook that shows off SQL analytics skills, it could be a nice flourish.

### When You WOULD Use DuckDB
- Analysing training logs across 100+ experiments
- Querying a 50GB dataset of image metadata
- Building a data quality dashboard
- Replacing a Spark setup for single-machine analytics

---

## 16.3 Cache Systems

### Types of Caching Relevant to ML

| Cache Type | How It Works | Where It Lives |
|-----------|-------------|----------------|
| **Filesystem cache** | Save processed tensors to disk; load if exists | Local SSD / Google Drive |
| **In-memory cache** | `dict` or `lru_cache` — keep data in RAM | Process memory |
| **SQLite** | Lightweight embedded DB for key-value lookups | Local file (`.db`) |
| **Redis** | In-memory key-value store (networked) | Requires server process |

### Use Case 1: Cache Preprocessed Images

Loading + resizing + augmenting images from disk every epoch is slow. Cache the preprocessed versions:

```python
import os
import pickle
import hashlib

class CachedDataset(Dataset):
    """
    Wraps a dataset with filesystem caching.
    First epoch: process from raw → save to cache.
    Subsequent epochs: load from cache (10× faster).
    """
    def __init__(self, base_dataset, cache_dir='/content/cache'):
        self.base = base_dataset
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        cache_path = os.path.join(self.cache_dir, f'{idx}.pt')
        
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)
        
        # Process from raw
        item = self.base[idx]
        
        # Save to cache
        torch.save(item, cache_path)
        
        return item
```

### Use Case 2: Cache SRM Features

SRM filters are fixed (non-trainable). Computing them every batch is wasted work:

```python
class SRMCachedDataset(Dataset):
    """Pre-compute and cache SRM noise features."""
    
    def __init__(self, image_paths, mask_paths, transform, srm_layer, cache_dir):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.srm = srm_layer
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def __getitem__(self, idx):
        cache_path = os.path.join(self.cache_dir, f'{idx}_srm.pt')
        
        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=True)
            return data['combined'], data['mask']
        
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L')) / 255.0
        
        # Apply augmentation
        transformed = self.transform(image=image, mask=mask.astype(np.float32))
        image_tensor = transformed['image']       # (3, H, W)
        mask_tensor = transformed['mask'].unsqueeze(0)  # (1, H, W)
        
        # Compute SRM (no grad needed)
        with torch.no_grad():
            noise = self.srm(image_tensor.unsqueeze(0))    # (1, 30, H, W)
            noise_reduced = self.reducer(noise).squeeze(0)  # (3, H, W)
        
        combined = torch.cat([image_tensor, noise_reduced], dim=0)  # (6, H, W)
        
        # Cache it
        torch.save({'combined': combined, 'mask': mask_tensor}, cache_path)
        
        return combined, mask_tensor
```

> **Warning**: Don't cache training data if using random augmentation — each epoch should produce different augmented versions. Cache only for validation/test sets, or for the non-augmented base preprocessing.

### Should You Use Caching in This Project?

| Factor | Assessment |
|--------|-----------|
| **Dataset size** | 5K images at 512×512 = ~15 GB cached tensors → exceeds Colab disk |
| **Speed gain** | ~20% for data loading (from 4min to 3.2min per epoch) |
| **Complexity** | Moderate — cache invalidation, disk management |
| **Colab disk** | ~100 GB available, but shared with dataset + model |

**Verdict: Selective use only.** Cache the validation/test sets (small, no augmentation) for faster evaluation. Don't cache training data due to random augmentation.

### When You WOULD Use Full Caching
- Large datasets (>50K images) where I/O is the bottleneck
- Expensive preprocessing (super-resolution, optical flow computation)
- Production inference servers (Redis for feature caching)

---

## 16.4 DynamoDB

### What Is It?
Amazon DynamoDB is a fully managed NoSQL cloud database. It provides single-digit millisecond read/write at any scale, with automatic scaling and zero server management.

### Key Characteristics

| Feature | Details |
|---------|---------|
| **Type** | Key-value + document store |
| **Hosting** | AWS managed (serverless) |
| **Pricing** | Pay-per-request or provisioned capacity |
| **Latency** | <10ms reads/writes |
| **Scale** | Virtually unlimited (designed for >1M req/sec) |
| **SDK** | `boto3` (Python AWS SDK) |

### Potential Use Cases for ML Projects

```python
import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('experiment-runs')

# Store experiment metadata
table.put_item(Item={
    'run_id': 'unet-effb1-srm-v1-20240115',
    'config': {'lr': 1e-4, 'batch_size': 4},
    'metrics': {'best_f1': 0.65, 'best_epoch': 35},
    'status': 'completed',
    'timestamp': '2024-01-15T10:30:00Z'
})

# Query all runs with F1 > 0.60
response = table.scan(
    FilterExpression=Attr('metrics.best_f1').gte(0.60)
)
```

### Should You Use DynamoDB in This Project?

| Factor | Assessment |
|--------|-----------|
| **Requires AWS account** | Yes — setup overhead, possible billing surprises |
| **Latency needs** | None — this is a Colab notebook, not a production API |
| **Data volume** | ~5K images, ~20 experiments — a JSON file handles this |
| **Team access** | W&B dashboard already provides this for experiment tracking |
| **Cost** | Free tier: 25 GB + 25 write/read capacity units. But AWS billing is complex. |

**Verdict: Don't use for this project.** DynamoDB solves problems at scale (millions of requests, multi-region, serverless) that don't exist in a solo intern project on Colab. W&B + Google Drive covers all your storage and sharing needs.

### When You WOULD Use DynamoDB
- Production ML serving: cache model predictions for repeat queries
- Multi-user inference APIs: track request metadata, rate limiting
- Feature stores: serve pre-computed features at low latency
- ML pipeline orchestration: track job states across distributed workers

---

## 16.5 Decision Matrix

| Technology | Our Project Needs It? | When It Makes Sense |
|-----------|----------------------|-------------------|
| **DuckDB** | ❌ No | 100K+ row analytics; replacing heavy Spark setups |
| **Filesystem Cache** | ⚠️ Partial (val/test only) | Expensive preprocessing; large datasets; inference caching |
| **Redis** | ❌ No | Production inference servers; real-time feature serving |
| **SQLite** | ❌ No | Local metadata management; embedded applications |
| **DynamoDB** | ❌ No | Serverless APIs; multi-region scale; production ML systems |

---

## 16.6 What TO Use Instead

For this project, these simpler tools cover all data management needs:

| Need | Tool | Why |
|------|------|-----|
| Dataset loading | PyTorch `DataLoader` with `num_workers=2` | Parallel data loading, built-in |
| Experiment tracking | W&B | Purpose-built for ML experiments |
| Model storage | Google Drive + HF Hub | Free, accessible, versioned |
| Metadata | Python dicts + JSON files | 5K images don't need a database |
| Results sharing | W&B dashboard + Colab link | Browser-accessible, no setup |

**KISS principle**: Use the simplest tool that solves the problem. Introduce databases when the problem demands them, not before.
