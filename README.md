# A-FINE
Official PyTorch code for our Paper "Toward Generalized Image Quality Assessment:
Relaxing the Perfect Reference Quality Assumption" in CVPR 2025.

### [Paper and Supplementary (Arxiv Version)](https://arxiv.org/)

> **Toward Generalized Image Quality Assessment:
Relaxing the Perfect Reference Quality Assumption** <br>
> [Du CHEN\*](https://github.com/ChrisDud0257), [Tianhe WU\*](https://github.com/TianheWu), [Kede MA](https://kedema.org/) and [Lei ZHANG](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>



## :blush: Introduction



## 🚀 Quick Start

### 1.Installation
 - python == 3.10
 - PyTorch == 2.0
 - Anaconda
 - CUDA == 11.8

Then install the relevant environments :
```
git clone https://github.com/ChrisDud0257/AFINE
cd QuickInference
conda create --name afine python=3.10
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Download the [pretrained CLIP model](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) and our [A-FINE model](https://drive.google.com/drive/folders/1SgcMmv-9yejHYTT8F8hGN_5Vv8hfGMmR?usp=sharing).


### 3. Inference
For quick test towards any pair of images (distortion, reference), you can run the following command:
```
cd QuickInference
python afine.py --pretrain_CLIP_path [path to the pretrained CLIP ViT-B-32.pt] --afine_path [path to our afine.pth] \
--dis_img_path [path to the distortion image] --ref_img_path [path to the reference image]
```

We also provide one pair of testing examples here, the [reference image](figures/online20_Original.png) and [distortion image](figures/online20_RealESRNetx4.png).

Please note that, you cannnot change the path of reference image and distortion image, since A-FINE(dis, ref) != A-FINE(ref, dis).
As for A-FINE, the lower, the better.
