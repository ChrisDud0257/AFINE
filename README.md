# A-FINE
Official PyTorch code for our Paper "Toward Generalized Image Quality Assessment:
Relaxing the Perfect Reference Quality Assumption" in CVPR 2025.

### [Paper and Supplementary (Arxiv Version)](https://arxiv.org/)

> **Toward Generalized Image Quality Assessment:
Relaxing the Perfect Reference Quality Assumption** <br>
> [Du CHEN\*](https://github.com/ChrisDud0257), [Tianhe WU\*](https://github.com/TianheWu), [Kede MA](https://kedema.org/) and [Lei ZHANG](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>



## :blush: Introduction

### 1.Example
![teaser](figures/teaser.png)
<div style="text-align: center;"> <span style="color: red;">With the reference image in the middle, which image, A or B, has better perceived visual quality?</span></div>

Image A and B are generated by state-of-the-art generative-based models. While in (b), the generated Image B 
has much better visual quality than the reference image. All of the existing FR-IQA models fail to give an accurate judgement, 
since they only measure the similarity between two images, and assume that the reference image has the best quality. On the contrary,
the proposed A-FINE generalizes and outperforms standard FR-IQA models under both perfect and
imperfect reference conditions. Please zoom in for better visibility.



### 2.Abstract
Full-reference image quality assessment (FR-IQA) generally assumes that reference images are 
of perfect quality. However, this assumption is flawed due to the sensor and optical limitations 
of modern imaging systems. Moreover, recent generative enhancement methods are capable of producing 
images of higher quality than their original. All of these challenge the effectiveness and applicability 
of current FR-IQA models. To relax the assumption of perfect reference image  quality, we build a 
large-scale IQA database, namely DiffIQA, containing approximately $180,000$ images generated by a 
diffusion-based image enhancer with adjustable hyper-parameters. 
Each image is annotated by human subjects as either worse, similar, or better quality compared to its reference. 
Building on this, we present a generalized FR-IQA model, namely **A**daptive **FI**delity-**N**aturalness **E**valuator (A-FINE), 
to accurately assess and adaptively combine the fidelity and naturalness of a test image. 
A-FINE aligns well with standard FR-IQA when the reference image is much more natural than the test image. 
We demonstrate by extensive experiments that A-FINE surpasses standard FR-IQA models on well-established IQA 
datasets and our newly created DiffIQA. To further validate A-FINE, we additionally construct a super-resolution 
IQA benchmark (SRIQA-Bench), encompassing test images derived from ten state-of-the-art SR methods with reliable human 
quality annotations. Tests on SRIQA-Bench re-affirm the advantages of A-FINE.

### 3.Formula
$$
D(dis, ref) = F(dis, ref) + \lambda * N(dis)
$$

$$
\lambda = exp(k(N(ref) - N(dis)))
$$

where $F$ denotes the fidelity term, $N$ denotes the naturalness term, and $k > 0$ is a learnable hyperparameter.
$F$ and $N$ are predicted by deep neural network. $D(dis, ref)$ denotes the A-FINE score. $F \in (-2, 2)$, $N \in (-2, 2)$. $D(dis, ref) \in (-\infty, \infty)$.

**As for $F$, $N$, $D$, the lower, the better.** 

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

### 2. Download the pretrained CLIP model and our A-FINE model.

|      Model       |                                                               Download                                                                |
|:----------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| CLIP ViT-B-32.pt | [OPENAI](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |
|    afine.pth     |                 [Google Drive](https://drive.google.com/drive/folders/1SgcMmv-9yejHYTT8F8hGN_5Vv8hfGMmR?usp=sharing)                  |

### 3. Inference
For quick test towards any pair of images (distortion, reference), you can run the following command:
```
cd QuickInference
python afine.py --pretrain_CLIP_path [path to the pretrained CLIP ViT-B-32.pt] --afine_path [path to our afine.pth] \
--dis_img_path [path to the distortion image] --ref_img_path [path to the reference image]
```

### 4.Explanations about the final A-FINE score

In very few cases, the reference image is of poor quality, while the distortion image has much 
better quality, then $D(dis, ref)$ will be a considerable negative value. To prevent from numeric overflow, we utilize a non-linear mapping
function to scale it to $D(dis, ref)_{s} \in (0, 100)$.



The lower $D(dis, ref)_{s}$ value, the better quality.

As for the final output, in [afine.py](QuickInference/afine.py), the **afine_all** indicates $D(dis, ref)$, while **afine_all_scale**
indicates $D(dis, ref)_{s}$. You could choose either one of them. If you use A-FINE in your publication, please specific which version
(scaled or not) you are using.

We also provide one pair of testing examples here, the [reference image](figures/online20_Original.png) and [distortion image](figures/online20_RealESRNetx4.png).

Please note that, you cannnot change the path of reference image and distortion image, since A-FINE is an asymmetric FR-IQA model:

$$
D(dis, ref) \neq D(ref, dis)
$$




