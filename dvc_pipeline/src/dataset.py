"""CASIA 2.0 Segmentation Dataset."""
import io, os
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageChops, ImageEnhance
from torch.utils.data import Dataset

class CASIASegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, ela_mean, ela_std, img_size=384, qualities=None):
        self.image_paths=image_paths; self.mask_paths=mask_paths; self.labels=labels
        self.ela_mean=ela_mean; self.ela_std=ela_std; self.img_size=img_size
        self.qualities=qualities or [75,85,95]
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        try: mqela = self._compute_multi_q_ela(self.image_paths[idx])
        except: mqela = np.zeros((self.img_size,self.img_size,9), dtype=np.uint8)
        tensor = torch.from_numpy(mqela.astype(np.float32)/255.0).permute(2,0,1)
        for c in range(9): tensor[c]=(tensor[c]-self.ela_mean[c])/self.ela_std[c]
        mp = self.mask_paths[idx]
        if mp and os.path.exists(mp):
            m = Image.open(mp).convert("L").resize((self.img_size,self.img_size),Image.NEAREST)
            ma = (np.array(m).astype(np.float32)/255.0 > 0.5).astype(np.float32)
        else:
            ma = np.ones((self.img_size,self.img_size),dtype=np.float32) if self.labels[idx]==1 else np.zeros((self.img_size,self.img_size),dtype=np.float32)
        return tensor, torch.from_numpy(ma).unsqueeze(0), self.labels[idx]
    def _compute_multi_q_ela(self, path):
        img = Image.open(path).convert("RGB")
        return np.concatenate([self._compute_ela_rgb(img,q) for q in self.qualities], axis=-1)
    def _compute_ela_rgb(self, image, quality):
        buf = io.BytesIO(); image.save(buf,"JPEG",quality=quality); buf.seek(0)
        ela = ImageChops.difference(image, Image.open(buf))
        mx = max(v[1] for v in ela.getextrema()) or 1
        ela = ImageEnhance.Brightness(ela).enhance(255.0/mx)
        return np.array(ela.resize((self.img_size,self.img_size),Image.BILINEAR))

def collect_image_paths(directory, extensions=None):
    if extensions is None: extensions={".jpg",".jpeg",".png",".tif",".bmp"}
    return sorted([os.path.join(directory,f) for f in os.listdir(directory) if Path(f).suffix.lower() in extensions])
