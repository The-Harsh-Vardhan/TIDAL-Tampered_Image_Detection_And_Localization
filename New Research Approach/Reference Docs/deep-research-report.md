# Paper Overview  
The identifier **ETASR_9593** corresponds to the paper *“Enhanced Image Tampering Detection using Error Level Analysis and a CNN”* by R. Gorle and A. Guttavelli (Eng. Technol. Appl. Sci. Res., Feb. 2025)【1†L18-L23】.  This work combines classical *Error Level Analysis (ELA)* with a CNN to spot image forgeries.  We examined the ETASR journal page – notably its “Downloads” section shows **“Download data is not yet available”**【16†L88-L90】 – suggesting the authors did **not** attach any code or supplementary files.  No official GitHub/GitLab repo was mentioned by the authors, and we found no code under their names. 

## Related Open-Source Projects  
Although no official code for ETASR_9593 was found, we identified several GitHub projects that implement similar ELA + CNN tamper-detection ideas.  These repositories may serve as useful references or starting points:

- **FakeImageDetector** by Agus Gunawan *et al.* (GitHub: `agusgun/FakeImageDetector`): A Jupyter-based project titled *“Image Tampering Detection using ELA and CNN”*【35†L243-L251】.  It implements ELA preprocessing and a CNN classifier, reporting ~91.8% accuracy.  The repo includes an IPython notebook (`fake-image-detection.ipynb`) that trains on CASIA 2.0 images.  (See the README title and description in [35] or [41]【35†L243-L251】.)  The code is in Python (Keras/TensorFlow), and usage is via the notebook (no published pip package).  

- **Image-Forgery-Detection** by Divyansh Gautam (`Divyansh-git10/image-forgery-detection`): Implements a *custom CNN* on ELA-transformed images【33†L463-L472】.  The readme touts that their task-specific CNN (≈24M parameters) outperforms transfer models.  Usage is documented in the repo: clone with `git clone https://github.com/Divyansh-git10/image-forgery-detection`, install requirements, and run the included `ML_PROJECT_IMPLEMENTATION.ipynb` (see lines [33†L434-L443]).  This project trains on CASIA 2.0, showing ~91.2% val accuracy for their CNN.  (See [33†L463-L472] and [33†L434-L441] for description and run instructions.)

- **Image-Tampering-Detection-using-ELA-and-Metadata-Analysis** by Madake *et al.* (`jayant1211/Image-Tampering-Detection-using-ELA-and-Metadata-Analysis`): A multi-modal system combining ELA with a weather-based metadata check【47†L312-L320】.  The GitHub pages shows detailed setup steps: clone the repo, run `pip install -r requirements.txt`, and use `streamlit run app.py` for inference【47†L295-L303】.  It uses DenseNet121 on ELA images plus a “Weather CNN” for context.  This is more complex than plain ELA+CNN, but it is a relevant open implementation of ELA-based tampering. 

- **Forgegy-Image-Detection-Using-Error-level-Analysis-and-Deep-Learning** by user *skj092* (`skj092/Forgegy-Image-Detection-Using-Error-level-Analysis-and-Deep-Learning`): Another Python project using ELA + CNN. Its README explains using ELA to highlight compression artifacts and mentions achieved results (CNN: 88.8% vs ResNet: 96.2% accuracy)【37†L261-L269】.  The repo contains Python scripts (`models.py`, `predict.py`) and notebooks for training and inference.  This shows a typical ELA→CNN workflow; clone via `git clone https://github.com/skj092/Forgegy-Image-Detection-Using-Error-level-Analysis-and-Deep-Learning` (the README demonstrates how to run predictions with `predict.py`)【37†L289-L297】.

> **Note:** We found *no official code repository* specifically for the ETASR_9593 paper itself. The authors did not publish their source code on GitHub or similar as far as our search could determine.  The projects above are community implementations of ELA+CNN methods and can provide insight or a starting point, but they are not the authors’ own code. 

## Summary of Repositories Found  
- **agusgun/FakeImageDetector (GitHub)** – *“Fake Image Detector: Image Tampering Detection using ELA and CNN”*【35†L243-L251】. Jupyter notebook implementation by Agus Gunawan et al., uses ELA preprocessing + CNN.  
- **Divyansh-git10/image-forgery-detection (GitHub)** – Custom CNN on ELA images【33†L463-L472】. Clone and run the provided notebook as documented【33†L434-L441】.  
- **jayant1211/Image-Tampering-Detection-using-ELA-and-Metadata-Analysis (GitHub)** – Multi-modal tamper detection with ELA + weather metadata【47†L295-L303】. Provides Streamlit demo and instructions for local use.  
- **skj092/Forgegy-Image-Detection-Using-Error-level-Analysis-and-Deep-Learning (GitHub)** – ELA + deep learning, with example results【37†L261-L269】. Contains scripts and notebooks for training and testing.  

Each of the above repos includes code (often Python notebooks) demonstrating ELA-based tamper detection.  If you’re looking to experiment with or reproduce ETASR_9593’s ideas, they offer concrete implementations to study.  

**References:** The information above is drawn from the official paper (title and DOI)【1†L18-L23】, the ETASR journal site【16†L88-L90】, and the cited GitHub repositories for related projects【35†L243-L251】【33†L463-L472】【47†L295-L303】【37†L261-L269】. These sources detail the repo contents, purpose, and usage. 

