# 19. Hugging Face Deployment — Spaces, Inference API & Demo Guide

## 19.1 What Is HF Deployment?

Hugging Face offers multiple ways to **deploy your trained model** as a live, interactive application — no server management, no Docker, no cloud billing:

| Deployment Option | What It Is | Infrastructure |
|-------------------|-----------|----------------|
| **HF Spaces (Gradio)** | Interactive web app with image upload | HF-hosted, free CPU or paid GPU |
| **HF Spaces (Streamlit)** | Dashboard-style app | HF-hosted, free CPU or paid GPU |
| **Inference API** | REST endpoint for model predictions | HF-hosted, auto-scaled |
| **Inference Endpoints** | Dedicated GPU endpoint (production) | HF-hosted, pay-per-hour |

---

## 19.2 Why Deploy Your Model?

### For This Assignment

| Benefit | Impact |
|---------|--------|
| **Live demo link** | Evaluator uploads any image → sees prediction instantly |
| **Beyond a notebook** | Shows you can take a model from training to production |
| **Portfolio piece** | A working demo is more impressive than a static notebook |
| **Zero cost** | HF Spaces free tier is sufficient for a demo |

### Evaluator Experience
Instead of:
> "Here's a Colab notebook. Run it for 3.5 hours to see results."

You send:
> "Here's a live demo. Upload any image and see the tampering mask in 2 seconds."

---

## 19.3 Option 1: Gradio Space (Recommended)

### What Is Gradio?
Gradio is a Python library that creates web UIs for ML models in ~20 lines of code. HF Spaces hosts the app for free.

### Step 1: Create a Space

Go to `https://huggingface.co/spaces` → Create Space:
- **Name**: `tampering-detector`
- **SDK**: Gradio
- **Hardware**: CPU Basic (free) — sufficient for inference on single images
- **Visibility**: Public

### Step 2: Create the App

```python
# app.py — the complete Gradio application

import gradio as gr
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import segmentation_models_pytorch as smp

# ========== Model Definition ==========
# (Copy TamperingDetector, SRMFilterLayer, ChannelReducer classes here)

class SRMFilterLayer(torch.nn.Module):
    # ... (same as training code)
    pass

class ChannelReducer(torch.nn.Module):
    # ... (same as training code)
    pass

class TamperingDetector(torch.nn.Module):
    # ... (same as training code)
    pass

# ========== Load Model ==========
def load_model():
    model = TamperingDetector(encoder_name='efficientnet-b1', encoder_weights=None)
    
    # Download weights from HF Hub
    weights_path = hf_hub_download(
        repo_id="your-username/tampering-detector-unet-effb1",
        filename="model.pt"
    )
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.eval()
    return model

model = load_model()

# ========== Inference ==========
THRESHOLD = 0.45  # Oracle threshold from validation

def predict_tampering(input_image):
    """
    Takes a PIL image, returns:
    1. Heatmap overlay (red = tampered regions)
    2. Binary mask
    3. Confidence text
    """
    if input_image is None:
        return None, None, "Please upload an image"
    
    # Preprocess
    image = input_image.resize((512, 512))
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_np - mean) / std
    
    # To tensor
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze().numpy()
    
    # Create outputs
    # 1. Heatmap overlay
    heatmap_overlay = image_np.copy()
    mask_rgb = np.zeros_like(image_np)
    mask_rgb[:, :, 0] = probs  # Red channel = tampering probability
    heatmap_overlay = np.clip(heatmap_overlay * 0.6 + mask_rgb * 0.4, 0, 1)
    heatmap_overlay = (heatmap_overlay * 255).astype(np.uint8)
    
    # 2. Binary mask
    binary_mask = (probs >= THRESHOLD).astype(np.uint8) * 255
    
    # 3. Confidence
    max_prob = probs.max()
    tampered_pct = (probs >= THRESHOLD).mean() * 100
    
    if max_prob < THRESHOLD:
        verdict = f"✅ AUTHENTIC (confidence: {(1-max_prob)*100:.1f}%)"
    else:
        verdict = f"⚠️ TAMPERED — {tampered_pct:.1f}% of image affected (confidence: {max_prob*100:.1f}%)"
    
    return Image.fromarray(heatmap_overlay), Image.fromarray(binary_mask), verdict

# ========== Gradio Interface ==========
demo = gr.Interface(
    fn=predict_tampering,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Tampering Heatmap"),
        gr.Image(type="pil", label="Binary Mask"),
        gr.Textbox(label="Verdict"),
    ],
    title="🔍 Image Tampering Detector",
    description=(
        "Upload an image to detect if it has been tampered with (splicing or copy-move). "
        "The model predicts a pixel-level tampering mask showing exactly which regions were manipulated.\n\n"
        "**Architecture**: U-Net + EfficientNet-B1 + SRM forensic preprocessing\n"
        "**Trained on**: CASIA v2.0 dataset"
    ),
    examples=[
        # Add a few example images to the Space repo
        ["examples/authentic_01.jpg"],
        ["examples/tampered_01.jpg"],
        ["examples/tampered_02.jpg"],
    ],
    cache_examples=True,
)

demo.launch()
```

### Step 3: Create `requirements.txt`

```
torch>=2.0
segmentation-models-pytorch
huggingface-hub
numpy
Pillow
```

### Step 4: Push to HF Spaces

```bash
# Clone the Space repo
git clone https://huggingface.co/spaces/your-username/tampering-detector
cd tampering-detector

# Add files
# - app.py (main application)
# - requirements.txt (dependencies)
# - examples/ (sample images)

git add .
git commit -m "Initial deployment"
git push
```

Or from Python:
```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id="your-username/tampering-detector",
    repo_type="space"
)

api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id="your-username/tampering-detector",
    repo_type="space"
)
```

### What the Live Demo Looks Like

```
┌──────────────────────────────────────────────────────────┐
│  🔍 Image Tampering Detector                              │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Upload Image  │  │  Heatmap     │  │ Binary Mask  │   │
│  │  [drag/drop]  │→ │  [red=forge] │  │ [B&W mask]   │   │
│  │              │  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                          │
│  Verdict: ⚠️ TAMPERED — 12.3% of image affected          │
│                                                          │
│  Examples: [auth_01] [tampered_01] [tampered_02]         │
└──────────────────────────────────────────────────────────┘
```

---

## 19.4 Option 2: HF Inference API (Auto-Deployed)

If your model follows HF conventions, the Inference API is auto-activated:

```python
# If your model repo has a proper config.json and model weights,
# HF provides a REST API automatically:

import requests

API_URL = "https://api-inference.huggingface.co/models/your-username/tampering-detector-unet-effb1"
headers = {"Authorization": "Bearer hf_YOUR_TOKEN"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

result = query("test_image.jpg")
```

> **Note**: The Inference API works best for models using standard HF architectures (transformers, diffusers). For custom architectures like ours (SMP + SRM), a Gradio Space is more reliable.

---

## 19.5 Deployment Architecture

```
┌──────────────────────────────────────────────────────┐
│                     HF Spaces                         │
│                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ User's    │    │  Gradio App  │    │ HF Hub    │  │
│  │ Browser   │───→│  (app.py)    │←───│ Model Repo│  │
│  │           │←───│              │    │ (weights) │  │
│  └──────────┘    └──────────────┘    └───────────┘  │
│                                                      │
│  Free CPU: ~30s inference per image                   │
│  Paid GPU (T4): ~2s inference per image               │
└──────────────────────────────────────────────────────┘
```

---

## 19.6 Performance Considerations

| Hardware | Cost | Inference Time | Suitable For |
|----------|------|---------------|-------------|
| **CPU Basic** | Free | ~20-30 seconds | Demo/showcase only |
| **CPU Upgrade** | $0.03/hr | ~10-15 seconds | Light usage |
| **T4 Small** | $0.60/hr | ~1-2 seconds | Interactive demo |
| **A10G Small** | $1.05/hr | <1 second | Production demo |

For a free assignment demo, **CPU Basic is fine**. Add a loading indicator and "Please wait ~20s" note.

---

## 19.7 Should You Deploy for This Project?

| Factor | Assessment |
|--------|-----------|
| **Required by assignment?** | No |
| **Time to set up** | ~30-45 minutes |
| **Cost** | Free (CPU tier) |
| **Evaluator impression** | Very strong — "this candidate ships products" |
| **Portfolio value** | High — live demo > static notebook |
| **Technical risk** | Low — Gradio is straightforward; model weights from HF Hub |

**Verdict: Yes, if time permits.** A live Gradio demo is worth the 30-minute investment. Include the Space link in your notebook conclusion:

```markdown
## Live Demo
Try the model: https://huggingface.co/spaces/your-username/tampering-detector
```

---

## 19.8 Deployment Checklist

- [ ] Model weights uploaded to HF Hub (Doc 14)
- [ ] `app.py` created with model loading + inference + Gradio UI
- [ ] `requirements.txt` with pinned major versions
- [ ] 2-3 example images included in `examples/` folder
- [ ] Space builds successfully (check Logs tab on HF Spaces)
- [ ] Tested with both authentic and tampered images
- [ ] Added Space link to notebook conclusion
- [ ] Added description explaining the model and architecture
