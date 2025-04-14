# Dataset preparation

## Training dataset

We utilize the widely-used DIV2K, Flickr2K, OST, FFHQ, LSDIR and DIV8K to train our enhancer. Please download them from:

|Dataset|Link|
|:---:|:---:|
|DIV2K(1-800 images for training)|[Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/)|
|Flickr2K|[Download](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)|
|OST|[Download](https://github.com/xinntao/SFTGAN)|
|FFHQ|[Download](https://github.com/NVlabs/ffhq-dataset)|
|LSDIR|[Download](https://ofsoundof.github.io/lsdir-data/)|
|DIV8K|[Download](https://competitions.codalab.org/competitions/22217#participate)|


Please note that, since FFHQ is a large-scale face image dataset, we only utilize the firt $10000$ images from FFHQ.


## Inference datasets preparation and how to generate DiffIQA

All of the original full-size images (which means the original images we collect from exising dataset/internet/cameras), the cropped $512 \times 512$ image patches (which serves as the standard original refence image for inferring DiffIQA dataset), and the generated DiffIQA images could be download from here,

|Dataset|Link|
|:---:|:---:|
|Full-size images|[Google Drive](https://drive.google.com/drive/folders/1vZehlUPDyDfo6Mq1K8pAMe3pcjqdDRht?usp=sharing)|
|Cropped $512 \times 512$ image patches|[Google Drive](https://drive.google.com/drive/folders/1vZehlUPDyDfo6Mq1K8pAMe3pcjqdDRht?usp=sharing)|
|Generated DiffIQA images (Training/Validation/Testing parts)|[Google Drive](https://drive.google.com/drive/folders/1vZehlUPDyDfo6Mq1K8pAMe3pcjqdDRht?usp=sharing)|

**Please note that, the cropped $512 \times 512$ image patches denote the same original reference image in DiffIQA.**


Firstly, we collect 1200 images from DIV2K and Flickr2K, 1000 images from Internet under the license of Creative Commons, and we capture 640 images from real-world situations through our mobile phones or DSLR cameras.
Secondly, we filter out the full-size images with large areas of flatten regions, please find the code [here](../scripts/select_img_with_var.py). Use the following command,
```bash
cd DiffEnhancer/scripts/preparation_for_diffiqa
python select_img_with_var.py --img_path [path to our provided original full-size images] --save_path [path to your save path]
```
Thirdly, please crop the filtered image from the 2nd step into $512 \times 512$ size, please find the code [here](../scripts/crop_img.py). Use the following command,
```bash
cd DiffEnhancer/scripts/preparation_for_diffiqa
python crop_img.py --img_path [path to the filtered images from step 2nd] --save_path [path to your save path]
```
Finally, please generate DiffIQA through feeding the cropped images from the 3rd step into our well-trained Diffusion Enhancer. Please follow this [instruction](../READEME.md) to infer DiffIQA.




