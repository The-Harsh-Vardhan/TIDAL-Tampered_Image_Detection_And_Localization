#!/usr/bin/env python3
"""
Data Loading Performance Benchmark
====================================
Benchmarks different num_workers configurations for optimal throughput.
"""

import os
import sys
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Configuration
BATCH_SIZE = 32
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS_TO_TEST = [0, 2, 4, 8]

# Assuming cache directory exists with ELA data
CACHE_DIR = Path("cache/ela_tensors_v2")

# Simple cached ELA dataset
class CachedELADataset(Dataset):
    """Loads pre-computed ELA tensors from disk cache."""
    def __init__(self, image_paths, labels, cache_dir, ela_mean, ela_std, img_size=384):
        self.image_paths = image_paths
        self.labels = labels
        self.cache_dir = cache_dir
        self.ela_mean = ela_mean.to('cpu')
        self.ela_std = ela_std.to('cpu')
        self.img_size = img_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        cache_file = os.path.join(self.cache_dir, f'ela_{idx:06d}.npy')
        
        try:
            ela_data = np.load(cache_file).astype(np.float32)
        except Exception as e:
            print(f'Cache error {idx}: {e}')
            ela_data = np.zeros((self.img_size, self.img_size, 9), dtype=np.float32)
        
        # Normalize
        ela_data = ela_data / 255.0
        ela_tensor = torch.from_numpy(ela_data).permute(2, 0, 1)  # (9, H, W)
        for c in range(9):
            ela_tensor[c] = (ela_tensor[c] - self.ela_mean[c]) / self.ela_std[c]
        
        # Simplified mask for this benchmark
        label = self.labels[idx]
        mask = torch.ones(1, self.img_size, self.img_size) if label == 1 else torch.zeros(1, self.img_size, self.img_size)
        
        return ela_tensor, mask, label


def main():
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Check if cache directory exists
    if not CACHE_DIR.exists():
        print(f"\n⚠️  Cache directory does not exist: {CACHE_DIR}")
        print("Creating sample data for demonstration...")
        
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create some sample ELA cache files
        n_samples = 100
        for i in range(n_samples):
            cache_file = CACHE_DIR / f'ela_{i:06d}.npy'
            if not cache_file.exists():
                # Create random ELA tensor for demo
                ela_data = np.random.randint(0, 256, (384, 384, 9), dtype=np.uint8)
                np.save(cache_file, ela_data)
        
        print(f"Created {n_samples} sample cache files.")
    
    # Get list of cache files
    cache_files = list(CACHE_DIR.glob('ela_*.npy'))
    if len(cache_files) == 0:
        print(f"❌ No cache files found in {CACHE_DIR}")
        return
    
    print(f"✓ Found {len(cache_files)} cache files")
    
    # Create sample image paths and labels
    image_paths = [f"image_{i}" for i in range(len(cache_files))]
    labels = np.random.randint(0, 2, len(cache_files))
    
    # Use default statistics for efficiency
    ELA_MEAN = torch.zeros(9)
    ELA_STD = torch.ones(9)
    print("\nUsing default ELA statistics (mean=0, std=1)")
    
    # Benchmark DataLoaders
    print(f"\n{'='*70}")
    print(f"  DATA LOADING BENCHMARK")
    print(f"{'='*70}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Testing num_workers: {NUM_WORKERS_TO_TEST}")
    print(f"Device: {DEVICE}")
    print(f"Total images available: {len(image_paths)}")
    print(f"{'='*70}\n")
    
    np.random.seed(SEED)
    # Use available samples for testing
    n_bench_samples = min(100, len(image_paths))
    train_sample_idx = np.arange(n_bench_samples)
    bench_paths = [image_paths[i] for i in train_sample_idx]
    bench_labels = [labels[i] for i in train_sample_idx]
    bench_dataset = CachedELADataset(bench_paths, bench_labels, CACHE_DIR, ELA_MEAN, ELA_STD)
    
    benchmark_results = {}
    for n_workers in NUM_WORKERS_TO_TEST:
        print(f"Testing with num_workers={n_workers}...")
        loader = DataLoader(
            bench_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            persistent_workers=(n_workers > 0),
            drop_last=True
        )
        
        latencies = []
        iterations = 0
        for batch_idx, (images, masks, batch_labels) in enumerate(loader):
            if batch_idx == 0:
                continue  # Skip first batch (warmup)
            if batch_idx > 20:  # More iterations for better averaging
                break
            iterations += 1
            
            t0 = time.perf_counter()
            _ = images.to(DEVICE, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
        
        if latencies:
            latencies = np.array(latencies)
            result = {
                'mean_ms': latencies.mean(),
                'std_ms': latencies.std(),
                'min_ms': latencies.min(),
                'max_ms': latencies.max(),
                'throughput': BATCH_SIZE / (latencies.mean() / 1000)
            }
            benchmark_results[n_workers] = result
            print(f'  num_workers={n_workers}:')
            print(f'    Mean: {result["mean_ms"]:7.2f}ms, Std: {result["std_ms"]:6.2f}ms')
            print(f'    Min:  {result["min_ms"]:7.2f}ms, Max: {result["max_ms"]:6.2f}ms')
            print(f'    Throughput: {result["throughput"]:6.0f} img/s')
            print()
    
    if benchmark_results:
        optimal_workers = min(benchmark_results.keys(), key=lambda w: benchmark_results[w]['mean_ms'])
        print(f'\n→ Optimal: num_workers={optimal_workers} ({benchmark_results[optimal_workers]["mean_ms"]:.1f}ms/batch)')
    print(f'{'='*70}')
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
