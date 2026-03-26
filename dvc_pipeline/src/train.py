"""Training CLI for TIDAL DVC pipeline."""
import argparse, json, logging, time
import numpy as np, torch, torch.optim as optim
import segmentation_models_pytorch as smp
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import CASIASegmentationDataset, collect_image_paths
from models import build_model, get_model_info
from utils import compute_pixel_f1, get_device, load_params, save_metrics, seed_everything
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(params):
    seed_everything(params["seed"]); device=get_device()
    mc,tc,pc,dc = params["model"],params["training"],params["preprocessing"],params["data"]
    model = build_model(mc["encoder"],mc["encoder_weights"],mc["in_channels"],mc["num_classes"],mc["freeze_strategy"]).to(device)
    info = get_model_info(model)
    logger.info("Model: %s trainable / %s total", f"{info['trainable_params']:,}", f"{info['total_params']:,}")
    from sklearn.model_selection import train_test_split
    au=collect_image_paths(f"{dc['dataset_root']}/Au"); tp=collect_image_paths(f"{dc['dataset_root']}/Tp")
    all_p,all_l = au+tp, [0]*len(au)+[1]*len(tp)
    from pathlib import Path
    stats_path = Path("artifacts/ela_statistics.json")
    if stats_path.exists():
        with open(stats_path) as f: s=json.load(f)
        em,es = torch.tensor(s["mean"],dtype=torch.float32), torch.tensor(s["std"],dtype=torch.float32)
    else: em,es = torch.zeros(9),torch.ones(9)
    tr_i,tmp = train_test_split(range(len(all_p)),test_size=0.30,stratify=all_l,random_state=params["seed"])
    v_i,_ = train_test_split(tmp,test_size=0.50,stratify=[all_l[i] for i in tmp],random_state=params["seed"])
    def mk(idx,sh):
        ds=CASIASegmentationDataset([all_p[i] for i in idx],[None]*len(idx),[all_l[i] for i in idx],em,es,pc["image_size"],pc["ela_qualities"])
        return DataLoader(ds,batch_size=tc["batch_size"],shuffle=sh,num_workers=tc["num_workers"],pin_memory=True,drop_last=sh)
    tl,vl = mk(tr_i,True),mk(v_i,False)
    bce=smp.losses.SoftBCEWithLogitsLoss(); dice=smp.losses.DiceLoss(mode="binary",from_logits=True)
    crit = lambda p,t: bce(p,t)+dice(p,t)
    opt=optim.Adam([p for p in model.parameters() if p.requires_grad],lr=tc["learning_rate"],weight_decay=tc["weight_decay"])
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=tc["scheduler_factor"],patience=tc["scheduler_patience"])
    scaler=GradScaler(enabled=tc["use_amp"])
    best_f1,patience_c = 0.0,0
    for ep in range(1,tc["epochs"]+1):
        model.train(); tloss=0.0
        for imgs,masks,_ in tqdm(tl,desc=f"Epoch {ep}",leave=False):
            imgs,masks=imgs.to(device),masks.to(device); opt.zero_grad(set_to_none=True)
            with autocast(enabled=tc["use_amp"]): loss=crit(model(imgs),masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tloss+=loss.item()
        tloss/=len(tl); model.eval(); vloss=0.0; ap,am=[],[]
        with torch.no_grad():
            for imgs,masks,_ in vl:
                imgs,masks=imgs.to(device),masks.to(device)
                with autocast(enabled=tc["use_amp"]): loss=crit(model(imgs),masks)
                vloss+=loss.item(); ap.append(torch.sigmoid(model(imgs).float()).cpu().numpy()); am.append(masks.cpu().numpy())
        vloss/=len(vl); mets=compute_pixel_f1(np.concatenate(ap),np.concatenate(am)); vf1=mets["pixel_f1"]
        sched.step(vloss); logger.info("Ep %d: tl=%.4f vl=%.4f vf1=%.4f",ep,tloss,vloss,vf1)
        if vf1>best_f1:
            best_f1=vf1; patience_c=0; torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"best_f1":best_f1},"../models/best_model.pt")
        else:
            patience_c+=1
            if patience_c>=tc["patience"]: logger.info("Early stop at %d",ep); break
    save_metrics({"best_pixel_f1":best_f1,"total_epochs":ep}, "metrics/train_metrics.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--params",default="params.yaml")
    train(load_params(ap.parse_args().params))
