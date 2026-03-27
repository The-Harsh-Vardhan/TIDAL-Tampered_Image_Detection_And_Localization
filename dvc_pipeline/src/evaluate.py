"""Evaluate trained model on test set."""
import json, logging
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from dataset import CASIASegmentationDataset, collect_image_paths
from models import build_model
from utils import compute_pixel_f1, get_device, load_params, save_metrics, seed_everything
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(params):
    seed_everything(params["seed"]); device=get_device()
    mc,ec,pc,dc = params["model"],params["evaluation"],params["preprocessing"],params["data"]
    model = build_model(mc["encoder"],None,mc["in_channels"],mc["num_classes"]).to(device)
    ckpt = torch.load("../models/best_model.pt",map_location=device,weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict",ckpt),strict=False); model.eval()
    with open("artifacts/ela_statistics.json") as f: s=json.load(f)
    em,es = torch.tensor(s["mean"],dtype=torch.float32), torch.tensor(s["std"],dtype=torch.float32)
    from sklearn.model_selection import train_test_split
    au=collect_image_paths(f"{dc['dataset_root']}/Au"); tp=collect_image_paths(f"{dc['dataset_root']}/Tp")
    all_p,all_l = au+tp, [0]*len(au)+[1]*len(tp)
    _,tmp = train_test_split(range(len(all_p)),test_size=0.30,stratify=all_l,random_state=params["seed"])
    _,test_i = train_test_split(tmp,test_size=0.50,stratify=[all_l[i] for i in tmp],random_state=params["seed"])
    ds=CASIASegmentationDataset([all_p[i] for i in test_i],[None]*len(test_i),[all_l[i] for i in test_i],em,es,pc["image_size"],pc["ela_qualities"])
    dl=DataLoader(ds,batch_size=16,shuffle=False)
    ap,am=[],[]
    with torch.no_grad():
        for imgs,masks,_ in dl:
            ap.append(torch.sigmoid(model(imgs.to(device)).float()).cpu().numpy()); am.append(masks.numpy())
    mets = compute_pixel_f1(np.concatenate(ap),np.concatenate(am))
    th = ec["thresholds"]; mets["quality_gate_passed"] = mets["pixel_f1"]>=th["min_pixel_f1"] and mets["pixel_iou"]>=th["min_iou"]
    mets["test_samples"]=len(test_i)
    Path("evaluation_results").mkdir(exist_ok=True)
    save_metrics(mets,"metrics/eval_metrics.json")
    logger.info("Results: %s Gate: %s", mets, "PASS" if mets["quality_gate_passed"] else "FAIL")

if __name__=="__main__": evaluate(load_params())
