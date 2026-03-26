"""Generate comparison grids."""
import json, logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, torch
from PIL import Image
from dataset import CASIASegmentationDataset, collect_image_paths
from models import build_model
from utils import get_device, load_params, seed_everything
logging.basicConfig(level=logging.INFO)

def visualize(params):
    seed_everything(params["seed"]); device=get_device()
    mc,vc,pc,dc = params["model"],params["visualization"],params["preprocessing"],params["data"]
    model = build_model(mc["encoder"],None,mc["in_channels"],mc["num_classes"]).to(device)
    ckpt = torch.load("../models/best_model.pt",map_location=device,weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict",ckpt),strict=False); model.eval()
    with open("artifacts/ela_statistics.json") as f: s=json.load(f)
    em,es = torch.tensor(s["mean"],dtype=torch.float32), torch.tensor(s["std"],dtype=torch.float32)
    tp = collect_image_paths(f"{dc['dataset_root']}/Tp"); n=min(vc["num_samples"],len(tp)); sp=tp[:n]
    ds=CASIASegmentationDataset(sp,[None]*n,[1]*n,em,es,pc["image_size"],pc["ela_qualities"])
    od=Path(vc["output_dir"]); od.mkdir(parents=True,exist_ok=True)
    cols=vc.get("grid_cols",4); rows=(n+cols-1)//cols
    fig,axes=plt.subplots(rows,cols*3,figsize=(cols*9,rows*3))
    if rows==1: axes=axes.reshape(1,-1)
    for i in range(n):
        r,cb=i//cols,(i%cols)*3; t,_,_=ds[i]
        orig=Image.open(sp[i]).convert("RGB").resize((pc["image_size"],pc["image_size"]))
        with torch.no_grad(): pm=torch.sigmoid(model(t.unsqueeze(0).to(device)).float()).squeeze().cpu().numpy()
        axes[r,cb].imshow(np.array(orig)); axes[r,cb].set_title("Original",fontsize=8); axes[r,cb].axis("off")
        axes[r,cb+1].imshow(pm,cmap="hot",vmin=0,vmax=1); axes[r,cb+1].set_title("Mask",fontsize=8); axes[r,cb+1].axis("off")
        ov=np.array(orig).astype(np.float32)/255.0; mr=np.zeros_like(ov); mr[:,:,0]=pm
        axes[r,cb+2].imshow(np.clip(ov*0.6+mr*0.4,0,1)); axes[r,cb+2].set_title("Overlay",fontsize=8); axes[r,cb+2].axis("off")
    plt.suptitle("TIDAL Results",fontsize=14); plt.tight_layout()
    plt.savefig(od/"comparison_grid.png",dpi=150,bbox_inches="tight"); plt.close()

if __name__=="__main__": visualize(load_params())
