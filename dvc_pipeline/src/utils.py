"""Shared utilities for DVC pipeline."""
import json, os, random
from pathlib import Path
import numpy as np, torch, yaml

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_params(path="params.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def save_metrics(metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f: json.dump(metrics, f, indent=2)

def get_device():
    d = os.environ.get("DEVICE","auto")
    if d == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)

def compute_pixel_f1(preds, targets, threshold=0.5):
    pb = (preds>threshold).astype(np.float32).flatten()
    tf = targets.astype(np.float32).flatten()
    tp=(pb*tf).sum(); fp=(pb*(1-tf)).sum(); fn=((1-pb)*tf).sum()
    p=tp/(tp+fp+1e-8); r=tp/(tp+fn+1e-8); f1=2*p*r/(p+r+1e-8); iou=tp/(tp+fp+fn+1e-8)
    return {"pixel_precision":float(p),"pixel_recall":float(r),"pixel_f1":float(f1),"pixel_iou":float(iou)}
