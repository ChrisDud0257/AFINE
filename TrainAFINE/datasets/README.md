# Dataset preparation

We split the total DiffIQA datasets and the traditional IQA datasets, i.e. [KADID-10K](https://database.mmsp-kn.de/kadid-10k-database.html), [PIPAL](https://github.com/HaomingCai/PIPAL-dataset), [TID2013](https://www.ponomarenko.info/tid2013.htm) into $7:1:2$ ratio, for training, validation, testing respectively. We provide the splitted DiffIQA dataset here. Please note that, for the images of traditional IQA datasets, we only provide the download link here. Please download the images from the official websites all by yourself, we only provide our splitted parts which record the image names as well as the MOS scores in ".txt" file, so you could reproduce our A-FINE model.


|  Dataset  |                             Image Download                             |              Label Download              |
| :---------: | :-----------------------------------------------------------------------: | :-----------------------------------------: |
|  DiffIQA  |                            [Google Drive](https://drive.google.com/drive/folders/1vZehlUPDyDfo6Mq1K8pAMe3pcjqdDRht?usp=drive_link)                            |             [Google Drive](https://drive.google.com/drive/folders/1vZehlUPDyDfo6Mq1K8pAMe3pcjqdDRht?usp=drive_link)             |
| KADID-10K | [Official website](https://database.mmsp-kn.de/kadid-10k-database.html) | [Our processed version on Google Drive](https://drive.google.com/drive/folders/1Agcgg-mA8aAGLSOrqkrycr5mYcv6IQav?usp=drive_link) |
|   PIPAL   |     [Official website](https://github.com/HaomingCai/PIPAL-dataset)     | [Our processed version on Google Drive](https://drive.google.com/drive/folders/1yGz3q8pySPFAPoixSjjXRGulbUurFK8b?usp=sharing) |
|  TID2013  |      [Official website](https://www.ponomarenko.info/tid2013.htm)      | [Our processed version on Google Drive](https://drive.google.com/drive/folders/1_iX8IvCNHQMR4Qvac35dS4HZr8XgySlI?usp=sharing) |

Note that, as for the files in the download link, we already split them into $7:1:2$ ratio, for training, validation, testing respectively. Please download all of them and carefully check the splitted parts.

Since the original KADID-10K, PIPAL and TID2013 do not split their whole datasets into clear training, validation and testing parts, then you could also randomly splitted them with different ratios according to your own need. 

**Please note that, the original image names in TID2013 are not uniformly formatted, sometimes they use lowercase and sometimes uppcase letters. To unify the naming convention, we change the original image names to lowercase letters. So after you download them from the original website, you need to rename them all to lowercase letters.**
We also provide the code for renaming:
```bash
cd TrainAFINE/scripts/process_process_traditional_iqa
python rename_tid2013.py --ori_path [path to the TID2013 image path] --save_path [path to your save path]
```


On the contrary, we have splitted DiffIQA into clear training, validation and testing parts, please don't split again if you use DiffIQA in your publication. Still, We provide our splitting code here for reference.

```bash
cd TrainAFINE/scripts/process_diffiqa
python split_train_val_test_DiffIQA --original_path [path to the whole DiffIQA dataset] --save_path [path to your save path]
```

## 1. Training datasets

Please note that, to simplify the training/validation/testing progress, we still need to process the labels of all datasets into our standard formats. We will give detailed descrptions as in the following.

### 1.1 DiffIQA

#### 1.1.1 Original labels

For the original human subjective labels, please download the whole DiffIQA dataset, after extracting them, please fine the original labels in ```trainlabel/label```. For the effective label for each $(generated, reference)$ pair, we ask $3$ different people to give subjects, we take the majority as the final label.

#### 1.1.2 Triplet

##### 1.1.2.1 The construction principle of our triplet
We implemente the learning-to-rank strategy to train A-FINE, so we need a triplet $(x_1, x_2, y)$ as our input, where $x1, x2$ denote two different generated images, $y$ denotes the reference image. Please note that, in DiffIQA, the generated image $x$ only has clear comparison result with $y$, and there is no direct subjective comparison between $(x_1, x_2)$, to make $(x_1, x_2)$ have clear and accurate comparison, we classify the training triplet $(x_1, x_2, y)$ into seven different categories:


| Type |        Description         |            Quality Comparison            |
| :-----: |:--------------------------:| :---------------------------------: |
| P,S,Y | P: $x_1$, S: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 > y$, $x_2 = y$ |
| P,N,Y | P: $x_1$, N: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 > y$, $x_2 < y$ |
| P,Y,Y | P: $x_1$, S: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 > y$, $x_2 = y$ |
| S,N,Y | S: $x_1$, N: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 = y$, $x_2 < y$ |
| N,Y,Y | N: $x_1$, Y: $x_2$, Y: $y$ | $x_1 < x_2$, $x_1 < y$, $x_2 = y$ |
| S,S,Y | S: $x_1$, S: $x_2$, Y: $y$ | $x_1 = x_2$, $x_1 = y$, $x_2 = y$ |
| S,Y,Y | S: $x_1$, Y: $x_2$, Y: $y$ | $x_1 = x_2$, $x_1 = y$, $x_2 = y$ |

For example, for any triplet $(x_1, x_2, y)$ triplet:
If the type is $(P,S,Y)$, $P$ means the selected $x_1$ image is a generated image, and it has better quality than the reference image; $S$ means the selected $x_2$ image is another generated image, and it has similari quality with the reference image; $Y$ means the selected $y$ image is the reference image.

If the type is $(N,Y,Y)$, $N$ means the selected $x_1$ image is a generated image, and it has worse quality than the reference image; the first $Y$ means the selected $x_2$ image is the reference image; the second $Y$ means the selected $y$ image is the reference image.

If the type is $(S,S,Y)$, the first $S$ means the selected $x_1$ image is a generated image, and it has similar quality with the reference image; the second $S$ means the selected $x_2$ image is another generated image, and it has similar quality with the reference image; the second $Y$ means the selected $y$ image is the reference image.

For other kinds of triplet, please extrapolate based on the table above.




###### 1.1.2.2 The specific recording format of our triplet
We have already explained how we construct the training triplet for DiffIQA, now we need to record them as well as the relative quality into ```.txt```, so that we could read in all of the necessary information during training. We embed the the triplets information into ```trainlabel/Triplet/Triplet.txt```, please check them carefully. The information is recorded as follows,
```bash
...
000006x0y32_01.png,000006x0y32_03.png,000006x0y32.png,0,0,1
...
```
The recording type could be indicated as $(x_1, x_2, y, score_{x1x2}, score_{x1y}, score_{x2y})$.  Here, we clarify the meaning of each item in the line of the txt. $x_1$ must be a generated image, $x_2$ could either be a generated one or the reference, $y$ must be the reference image. $score_{x1x2}$ denotes the relative score of $x_1$ and $x_2$; $score_{x1y}$ denotes the relative score of $x_1$ and $y$;  $score_{x2y}$ denotes the relative score of $x_2$ and $y$:


| Type |        Score        |           Quality Comparison            |
| :-----: | :-------------------------: | :---------------------------------: |
| $score_{x1x2}$ | 1 | $x_1 > x_2$ |
| $score_{x1x2}$ | 0.5 | $x_1 = x_2$ |
| $score_{x1x2}$ | 0 | $x_1 < x_2$ |
| $score_{x1y}$ | 1 | $x_1 > y$ |
| $score_{x1y}$ | 0.5 | $x_1 = y$ |
| $score_{x1y}$ | 0 | $x_1 < y$ |
| $score_{x2y}$ | 1 | $x_2 > y$ |
| $score_{x2y}$ | 0.5 | $x_2 = y$ |
| $score_{x2y}$ | 0 | $x_2 < y$ |

Through this format, we could train A-FINE under learning-to-rank strategy.

We also provide the triplet generation code here,
```bash
cd TrainAFINE/scripts/process_diffiqa
python generate_triplet.py --all_label_path [path to the original extracted labels] --save_path [path to the save path]
```

We have also recorded each type of the triplets into ```trainlabel/TripletEachType/```, please also check them carefully. The information is recorded as follows,
```bash
000017x27y13_02.png,000017x27y13_03.png,000017x27y13.png,0
...
```
The recording type could be indicated as $(x_1, x_2, y, score_{x1x2})$. $x_1$ must be a generated image, $x_2$ could either be a generated one or the reference, $y$ must be the reference image. $score_{x1x2}$ denotes the relative score of $x_1$ and $x_2$.
We also provide the generation code of each type here,
```bash
cd TrainAFINE/scripts/process_diffiqa
python generate_triplet_eachtype.py --all_label_path [path to the original extracted labels] --save_path [path to the save path]
```

**We strongly recommend you to directly use our well-processed triplet to train A-FINE. Otherwise, you might need to re-write the image reading code by yourself.**

#### 1.1.3 Images

Please directly download the images of DiffIQA from the link provided in the table at the top of this page. After downloading, please extract them. **Please don't change any formats of the images of DiffIQA, if you want to use our code to train A-FINE. Otherwise, you might need to re-write the image reading code by yourself.**

### 1.2 Traditional IQA datasets

#### 1.2.1 Original MOS scores

We provide the MOS scores for the training, validation, testing parts for KADID10K, PIPAL and TID2013. Please download them from the link provided at the top of this page. After downloading, please extract them from the ```.zip``` files. For the training parts, please find the MOS scores in ```KADID10K/Train/MOS/```, ```PIPAL/Train/MOS/```, ```TID2013/Train/MOS/```. Please note that, we don't change any MOS scores towards the existing IQA dataset. We only split them into clear training, validation and testing parts, and just provide the image grouping information.

As for the recording information in MOS scores, for PIPAL and TID2013, they are recored as follows,
```bash
A0002_00_00.bmp,A0002.bmp,1466.4372
```
The recording type could be indicated as $(x, y, MOS_{x})$. $x$ is the distortion image, $y$ is the reference image. $ MOS_{x}$ denotes the MOS score of $x$.



For KADID10K, they are recored as follows,
```bash
I01_01_02.png,I01.png,4.33,0.869
```
The recording type could be indicated as $(x, y, MOS_{x}, Var_{x})$. $x$ is the distortion image, $y$ is the reference image. $ MOS_{x}$ denotes the MOS score of $x$, $Var_{x}$ means the variation value of this image.

For more dataset information, please infer to their offical websit: [KADID-10K](https://database.mmsp-kn.de/kadid-10k-database.html), [PIPAL](https://github.com/HaomingCai/PIPAL-dataset), [TID2013](https://www.ponomarenko.info/tid2013.htm).

#### 1.2.2 Triplet

##### 1.2.2.1 The construction principle of our triplet
We implemente the learning-to-rank strategy to train A-FINE, so we need a triplet $(x_1, x_2, y)$ as our input, where $x1, x2$ denote two different generated images, $y$ denotes the reference image. Please note that, different from DiffIQA, the existing traditional IQA datasets have clear comparison results of two different distortion images $x_{1}$ and $x_{2}$, since any distortion image $x$ has its MOS score, then we could just obtain the relative quality comparison with their MOS scores. **For KADID10K, PIPAL and TID2013, the higher MOS score, the better quality**. Since all of the distortion images have worse quality than the reference image, then we classify the training triplet $(x_1, x_2, y)$ into two different categories:


| Type |        Description        |            Quality Comparison            |
| :-----: | :-------------------------: | :---------------------------------: |
| N,N,Y | N:$x_1$, N: $x_2$, Y: $y$ | $x_1 > x_2$ or $x_1 < x_2$, $x_1 < y$, $x_2 < y$ |
| N,Y,Y | N:$x_1$, Y: $x_2$, Y: $y$ | $x_1 < x_2$, $x_1 < y$, $x_2 = y$ |


For example, for any triplet $(x_1, x_2, y)$ triplet:
If the type is $(N,N,Y)$, the first $N$ means the selected $x_1$ image is a distortion image, and it has worse quality than the reference image; the second $N$ means the selected $x_2$ image is another distortion image, and it has worse quality with the reference image; $Y$ means the selected $y$ image is the reference image.

If the type is $(N,Y,Y)$, $N$ means the selected $x_1$ image is a distortion image, and it has worse quality than the reference image; the first $Y$ means the selected $x_2$ image is the reference image; the second $Y$ means the selected $y$ image is the reference image.


###### 1.2.2.2 The specific recording format of our triplet
We have already explained how we construct the training triplet for traditional IQA datasets, now we need to record them as well as the relative quality into ```.txt```, so that we could read in all of the necessary information during training. We embed the the triplets information into ```KADID10K/Train/Triplet/Triplet.txt```, ```PIPAL/Train/Triplet/Triplet.txt```, ```TID2013/Train/Triplet/Triplet.txt```, please check them carefully. The information is recorded as follows,
```bash
...
I01_05_01.png,I01_23_05.png,I01.png,1,0,0
...
```
The recording type could be indicated as $(x_1, x_2, y, score_{x1x2}, score_{x1y}, score_{x2y})$.  Here, we clarify the meaning of each item in the line of the txt. $x_1$ must be a distortion image, $x_2$ could either be a distortion one or the reference, $y$ must be the reference image. $score_{x1x2}$ denotes the relative score of $x_1$ and $x_2$; $score_{x1y}$ denotes the relative score of $x_1$ and $y$;  $score_{x2y}$ denotes the relative score of $x_2$ and $y$:


| Type |        Score        |           Quality Comparison            |
| :-----: | :-------------------------: | :---------------------------------: |
| $score_{x1x2}$ | 1 | $x_1 > x_2$ |
| $score_{x1x2}$ | 0.5 | $x_1 = x_2$ |
| $score_{x1x2}$ | 0 | $x_1 < x_2$ |
| $score_{x1y}$ | 1 | $x_1 > y$ |
| $score_{x1y}$ | 0.5 | $x_1 = y$ |
| $score_{x1y}$ | 0 | $x_1 < y$ |
| $score_{x2y}$ | 1 | $x_2 > y$ |
| $score_{x2y}$ | 0.5 | $x_2 = y$ |
| $score_{x2y}$ | 0 | $x_2 < y$ |


Through this format, we could train A-FINE under learning-to-rank strategy.

Please note that, if we want to list all triplets for traditiona IQA datasets, then there will be a huge quantity of combination groups. To simplify the triplet numbers, for each image $x_i$ in group ${x_1, x_2, ..., x_n, y}$, we randomly select $log_{2}^{n+1}$ combinations without repetition.

We also provide the triplet generation code here,
```bash
cd TrainAFINE/scripts/process_process_traditional_iqa
python generate_triplet.py --all_path [path to the KADID10K/PIPAL/TID2013 path, this path should contain our processed ```Train/Validation/Test``` folder]
```

#### 1.2.3 Images

Please download the images of traditional IQA datasets from their offical website, **and we strongly recommend you to extract all of the images (including the distortion and reference) into ```KADID10K/images```, ```PIPAL/images```, ```TID2013/images```, if you want to directly use our code to train A-FINE**.

## 2. Validation datasets

### 2.1 DiffIQA

#### 2.1.1 Original labels and triplets

Please note that, the format of labels and construction principle towards the triplets of validation parts are the same as the training parts.

**One thing you need to pay attention is that we only classfify the validation triplet into five different categories**:

| Type |        Description        |            Quality Comparison            |
| :-----: | :-------------------------: | :---------------------------------: |
| P,S,Y | P:$x_1$, S: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 > y$, $x_2 = y$ |
| P,N,Y | P:$x_1$, N: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 > y$, $x_2 < y$ |
| P,Y,Y |  P:$x_1$, S: $x_2$, Y: $y$  | $x_1 > x_2$, $x_1 > y$, $x_2 = y$ |
| S,N,Y | S:$x_1$, N: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 = y$, $x_2 < y$ |
| N,Y,Y |  N:$x_1$, Y: $x_2$, Y: $y$  | $x_1 < x_2$, $x_1 < y$, $x_2 = y$ |

We don't test $S,Y,Y$ or $S,S,Y$, becase the threshold for prediting the error among $S,S$ $S,Y$, $Y,Y$ is extremely difficult to define.
Please note that, in Table.2 in our main paper, in validation/testing parts, ```Ref < Test``` means the triplet $P,S,Y$, $P,N,Y$ and $P,Y,Y$, while ```Ref > Test``` means the triplet $S,N,Y$ and $N,Y,Y$.

#### 2.1.2 Images
Please directly download the images of DiffIQA from the link provided in the table at the top of this page. After downloading, please extract them. **Please don't change any formats of the images of DiffIQA, if you want to use our code to validate A-FINE. Otherwise, you might need to re-write the image reading code by yourself.**


### 2.2 Traditional IQA datasets

#### 2.2.1 Original labels and triplets
Please note that, the format of labels and construction principle towards the triplets of validation parts are the same as the training parts. 

#### 2.2.2 Images

Please download the images of traditional IQA datasets from their offical website, **and we strongly recommend you to extract all of the images (including the distortion and reference) into ```KADID10K/images```, ```PIPAL/images```, ```TID2013/images```, if you want to directly use our code to train A-FINE**.




## 3. Testing datasets

### 3.1 DiffIQA

#### 3.1.1 Original labels and triplets

Please note that, the format of labels and construction principle towards the triplets of testing parts are the same as the training parts.

**One thing you need to pay attention is that we only classfify the testing triplet into five different categories**:

| Type |        Description        |            Quality Comparison            |
| :-----: | :-------------------------: | :---------------------------------: |
| P,S,Y | P:$x_1$, S: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 > y$, $x_2 = y$ |
| P,N,Y | P:$x_1$, N: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 > y$, $x_2 < y$ |
| P,Y,Y |  P:$x_1$, S: $x_2$, Y: $y$  | $x_1 > x_2$, $x_1 > y$, $x_2 = y$ |
| S,N,Y | S:$x_1$, N: $x_2$, Y: $y$ | $x_1 > x_2$, $x_1 = y$, $x_2 < y$ |
| N,Y,Y |  N:$x_1$, Y: $x_2$, Y: $y$  | $x_1 < x_2$, $x_1 < y$, $x_2 = y$ |

We don't test $S,Y,Y$ or $S,S,Y$, becase the threshold for prediting the error among $S,S$ $S,Y$, $Y,Y$ is extremely difficult to define.
Please note that, in Table.2 in our main paper, in validation/testing parts, ```Ref < Test``` means the triplet $P,S,Y$, $P,N,Y$ and $P,Y,Y$, while ```Ref > Test``` means the triplet $S,N,Y$ and $N,Y,Y$.

#### 3.1.2 Images
Please directly download the images of DiffIQA from the link provided in the table at the top of this page. After downloading, please extract them. **Please don't change any formats of the images of DiffIQA, if you want to use our code to validate A-FINE. Otherwise, you might need to re-write the image reading code by yourself.**


### 3.2 Traditional IQA datasets

#### 3.2.1 Original labels and triplets
Please note that, the format of labels and construction principle towards the triplets of testing parts are the same as the training parts. 

#### 3.2.2 Images

Please download the images of traditional IQA datasets from their offical website, **and we strongly recommend you to extract all of the images (including the distortion and reference) into ```KADID10K/images```, ```PIPAL/images```, ```TID2013/images```, if you want to directly use our code to train A-FINE**.

### 3.3 SRIQA-Bench
### 1.Download SRIQA-Bench

| Dataset |                                                 Link                                                 |
|:-------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|
|SRIQA-Bench| [Google Drive](https://drive.google.com/drive/folders/1oRNyjHvjpop2OhurebdFuvisbpys2DRP?usp=sharing) |


### 2.Generation progress of SRIQA-Bench

We first compiled $100$ original images covering a wide range of natural scenes and subjected them to common
[Real-ESRGAN degradations](https://github.com/xinntao/Real-ESRGAN) and [BSRGAN degradations](https://github.com/cszn/BSRGAN) to generate
low-resolution (LR) images. We then adopted two regression-based SR methods and eight generative-based (GAN-based/Diffusion-based) models which are 
trained under blind degradations to produce SR results for each LR input:
 

|                                                                                    Regression-based                                                                                     |                                                                                      GAN-based                                                                                      |                                                                          Diffusion-based                                                                          |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                     [SwinIR](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf)                     | [Real-ESRGAN](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf) |                                             [StableSR](https://link.springer.com/article/10.1007/s11263-024-02168-7)                                              |
| [RealESRNet/RRDB](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf) |       [BSRGAN](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.pdf)       | [SUPIR](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Scaling_Up_to_Excellence_Practicing_Model_Scaling_for_Photo-Realistic_Image_CVPR_2024_paper.pdf) |
|                                                                                                                                                                                         |           [HGGT](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Human_Guided_Ground-Truth_Generation_for_Realistic_Image_Super-Resolution_CVPR_2023_paper.html)           |       [SeeSR](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_SeeSR_Towards_Semantics-Aware_Real-World_Image_Super-Resolution_CVPR_2024_paper.html)        |
|                                                                                                                                                                                         |                                                                                                                                                                                     |       [SinSR](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_SinSR_Diffusion-Based_Image_Super-Resolution_in_a_Single_Step_CVPR_2024_paper.html)        |
|                                                                                                                                                                                         |                                                                                                                                                                                     |                    [OSEDiff](https://proceedings.neurips.cc/paper_files/paper/2024/file/a8223b0ad64007423ffb308b0dd92298-Paper-Conference.pdf)                    |

For more details about the generation progress, please refer to the contents of Section 5.2 from the [main paper](https://arxiv.org/pdf/2503.11221) and D parts from the [supplementary](https://arxiv.org/pdf/2503.11221).

### 3.Annotation progress of SRIQA-Bench

For the images with same contents, we ask ten people to rate their visual quality through a complete paired comparison,
resulting in $C_{11}^{2}=55$ pairs of comparisons, resulting in $55 * 100=5500$ pairs of comparisons. Please note that, we also take the original reference image into consideration, thus the image with same contents contains the original reference image (ground truth) 
and ten SR images. Each pair of comparison is annotated by $10$ different people, we final obtain $5500*10=55000$ labels. During the subjective experiment, we ask $40$ people to participate in our task. The software randomly selects two different images from 11 each time without repetition and display them
on the screen. Finally, we obtain $55000$ comparison results. For the total $1100$ images, we compute the MOS score for each image, including the 
original reference image.

|           Type           |Number|
|:------------------------:|:---:|
| Original reference image |100|
|        SR images         |1000|
|       Total images       |1100|
|       __ volunteers take part in the experiment    | 40 |
|   Each pair of comparison is annoated by __ different people| 10 |
|Number of the pair of comparisons|5500|
|    Number of the labels made by human    |55000|
|        MOS scores        |1100|

For more details about the annotation progress, please refer to the contents of Section 5.2 from the [main paper](https://arxiv.org/pdf/2503.11221) and D parts from the [supplementary](https://arxiv.org/pdf/2503.11221).

### 4.Data structure of SRIQA-Bench
The data structure of SRIQA-Bench is as follows:
```
SRIQA-Bench
├── LRImages
│   ├── x1_Original.png
│   ├── x2_Original.png
│   ├── x3_Original.png
│   ├── ...
│   └── x100_Original.png
├── images
│   ├── SwinIR
│   │   ├── x1_SwinIRx4.png
│   │   ├── x2_SwinIRx4.png
│   │   ├── x3_SwinIRx4.png
│   │   ├──...
│   │   └── x100_SwinIRx4.png
│   ├── RealESRNet
│   │   ├── x1_RealESRNetx4.png
│   │   ├── x2_RealESRNetx4.png
│   │   ├── x3_RealESRNetx4.png
│   │   ├──...
│   │   └── x100_RealESRNetx4.png
│   ├──...
│   ├── Original
│   │   ├── x1_Original.png
│   │   ├── x2_Original.png
│   │   ├── x3_Original.png
│   │   ├──...
│   │   └── x100_Original.png
│── MOS
│   ├── x1.txt
│   ├── x2.txt
│   ├── x3.txt
│   ├──...
│   └── x100.txt
│── MOS_Reg
│   ├── x1.txt
│   ├── x2.txt
│   ├── x3.txt
│   ├──...
│   └── x100.txt
│── MOS_Gen
│   ├── x1.txt
│   ├── x2.txt
│   ├── x3.txt
│   ├──...
│   └── x100.txt
```

where ```x1``` to ```x100``` represent the prefix name of different images. The suffix name ```_SwinIRx4``` means that the image is generated by SwinIR under scaling factor $\times 4$. Similarily, other images adhere to identical naming protocols as demonstrated. Specially, ```_Original``` indicates the original
reference image (ground truth) in ```SRIQA-Bench\images\Original``` folder, or the low-resolution image in ```SRIQA-Bench\LRImages``` folder. All of the original image are also saved in ```SRIQA-Bench\images\Original``` folder.

For each comparison group towards the $11$ images with the same contents, we record their MOS scores 
in ```x1.txt``` to ```x100.txt```. Note that, ```MOS/``` folder records the MOS scores of both generative and regression models, ```MOS_Reg/``` folder only records the MOS scores of regression models, ```MOS_Gen/``` folder only rerecords the MOS scores of generative models. We also provide the code towards computation of MOS scores,
```bash
cd TrainAFINE/scripts/process_sriqa
python compute_MOS.py --label_path [path to the label path] --image_path [path to the image path] --save_path [path to the save path]
```
where the label path contains the original comparison results made by human.

Here, to compute MOS scores, we also provide the labels of original comparison results in ```SRIQA-Bench/labels/```, ```A/,B/,C/,D/,E/,F/,G/,H/,I/,J/``` denotes ten different people. The dataset structure could be seen as follows,

```bash
SRIQA-Bench
|── labels
|   ├── A
|   |   ├── x1
|   |   |    ├── x1_SwinIRx4_Original.json
|   |   |    ├── x1_BSRGANx4_Original.json
|   |   |    ├── ...
|   |   ├── x2
|   |   |    ├── x2_SwinIRx4_Original.json
|   |   |    ├── x2_BSRGANx4_Original.json
|   |   |    ├── ...
|   ├── B
|   |   ├── x1
|   |   |    ├── x1_SwinIRx4_Original.json
|   |   |    ├── x1_BSRGANx4_Original.json
|   |   |    ├── ...
|   |   ├── x2
|   |   |    ├── x2_SwinIRx4_Original.json
|   |   |    ├── x2_BSRGANx4_Original.json
|   |   |    ├── ...
|   ├── J
|   |   ├── x1
|   |   |    ├── x1_SwinIRx4_Original.json
|   |   |    ├── x1_BSRGANx4_Original.json
|   |   |    ├── ...
|   |   ├── x2
|   |   |    ├── x2_SwinIRx4_Original.json
|   |   |    ├── x2_BSRGANx4_Original.json
|   |   |    ├── ...
```


Note that, your computaion results of MOS might be a slightly differen from us, since the MOS scores are derived using a convex optimization progress, but this is not a problem since the difference in results is very small. However, we still strongly recommand you to direcrly use the MOS scores provided by us.

**The MOS score is varied in $(0, 100)$, the higher, the better.**



### 5.Other Declarations

**Copyright, License and Agreement for the SRIQA-Bench dataset Usage**

1. Please note that this dataset is made available for non-commercial academic research purposes ONLY.
2. You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
3. You agree not to further copy, publish or distribute any portion of the SRIQA-Bench dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.
4. The image contents are released upon request for research purposes ONLY.
5. Any violation of this protocol will be at his own risk. If any of the images include your information and you would like to remove them, please kindly inform us, and we will remove them from our dataset immediately.

**Towards the MOS scores for different methods**

Please note that, since every pretrained SR models are trained under different degradation conditions, some with weak degradation factors (such as HGGT), some with strong degradation factors (such as RealESRGAN),
and we just generate LR images just with our own settings, then the final MOS scores just indicate the quality of the generated SR images under this specific situations.

The primary purpose of establishing SRIQA-Bench is to evaluate the performance of different Full-Reference Image Quality Assessment (FR-IQA) methods, rather than to compare the superiority or inferiority of different SR models. 
The comparative results in SRIQA-Bench may not adequately reflect the performance differences between different SR models as follows:

(1). Limited Test Coverage: We only use 100 low-resolution (LR) images, which cannot fully cover diverse testing scenarios.

(2). Self-Defined Degradation Parameters: The blind degradation parameters (e.g., blur, noise, downscaling) are independently set by us, potentially introducing bias.

Given this, we have no intention of comparing SR models' performance. We hope the Mean Opinion Score (MOS) results from SRIQA-Bench will not be misinterpreted as an evaluation of SR models themselves.
