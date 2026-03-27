"""Compute ELA normalization statistics."""
import json, logging
from pathlib import Path
import numpy as np, torch
from dataset import CASIASegmentationDataset, collect_image_paths
from utils import load_params, save_metrics, seed_everything
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess(params):
    seed_everything(params["seed"])
    dc, pc = params["data"], params["preprocessing"]
    au = collect_image_paths(f"{dc['dataset_root']}/Au")
    tp = collect_image_paths(f"{dc['dataset_root']}/Tp")
    all_p = au+tp; n = min(pc["normalization_samples"], len(all_p))
    idx = np.random.choice(len(all_p), n, replace=False)
    ds = CASIASegmentationDataset([all_p[i] for i in idx],[None]*n,[0]*n,torch.zeros(9),torch.ones(9),pc["image_size"],pc["ela_qualities"])
    pixels = []
    for i in range(len(ds)):
        try: t,_,_ = ds[i]; pixels.append(t.numpy().reshape(9,-1))
        except: continue
    if not pixels: return
    px = np.concatenate(pixels,axis=1)
    mean, std = px.mean(axis=1).tolist(), [max(s,1e-6) for s in px.std(axis=1).tolist()]
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/ela_statistics.json","w") as f: json.dump({"mean":mean,"std":std,"n_samples":n},f,indent=2)
    save_metrics({"n_images":len(all_p),"n_samples":n}, "metrics/preprocess_metrics.json")
    logger.info("ELA stats saved. Mean: %s", [f"{v:.4f}" for v in mean])

if __name__=="__main__": preprocess(load_params())
