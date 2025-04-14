# Train details of A-FINE
Our training progress of A-FINE could be classified into three stages, we will detailed describe them as follows.

## 1. Installation
 - python == 3.10
 - PyTorch == 2.0
 - Anaconda
 - CUDA == 11.8

Then install the relevant environments :
```bash
git clone https://github.com/ChrisDud0257/AFINE
cd TrainAFINE
conda create --name trainafine python=3.10
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop
```

For more installation issues, please refer to the excellent [BasicSR](https://github.com/XPixelGroup/BasicSR) project.


## 2. Pretrained models

|           Stage           |lDowload|
|:------------------------:|:---:|
|CLIP ViT-b-32.pt|[OPENAI](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)|
| Stage 1 |[Google Drive](https://drive.google.com/drive/folders/1TCiecDmQGlw0U0tdOirH97WHXnWFXEqc?usp=drive_link)|
|        Stage 2        |[Google Drive](https://drive.google.com/drive/folders/1rDcwE5bB69D4nxD2oCQzNQPELxQXVx1v?usp=sharing)|
|      Stage 3       |[Google Drive](https://drive.google.com/drive/folders/1wAmonXCckwYCTi3nrRoOi5fYenULk1B1?usp=sharing)|


## 3. Training A-FINE
### 3.1 Stage 1
In this stage, we will train the naturalness term and finetune the CLIP model for A-FINE. The training datasets include DiffIQA, KADID10K, PIPAL and TID2013. The training command is as follows, which is also written in ```demo.sh```

```bash
cd TrainAFINE/
### For single GPU training
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage1_nlogn.yml --auto_resume

### For DDP training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage1_nlogn.yml --launcher pytorch --auto_resume
```

in ```options/train/AFINE/train_AFINE_stage1_nlogn.yml```, you need to modify the training dataset settings ,validation dataset settings as well as the pretrained CLIP ```ViT-b-32.pt``` model:

Training dataset settings:
```bash
(line 14)   all_img_path: ['/home/notebook/data/group/chendu/dataset/DiffIQA/Train/images',
                        '/home/notebook/data/group/chendu/dataset/IQA/KADID10K/images',
                        '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/images',
                        '/home/notebook/data/group/chendu/dataset/IQA/TID2013/images']
            ThreeAFC_list_path: ['/home/notebook/data/group/chendu/dataset/DiffIQA/Train/trainlabel/Triplet/Triplet.txt',
                                '/home/notebook/data/group/chendu/dataset/IQA/KADID10K/Train/Triplet/Triplet.txt',
                                '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/Train/Triplet/Triplet.txt',
                                '/home/notebook/data/group/chendu/dataset/IQA/TID2013/Train/Triplet/Triplet.txt']
```
You are supposed to indicate the image paths and triplet information of DiffIQA, KADID10K, PIPAL and TID2013. For more details, please refer to this [instruction](datasets/README.md) to prepare the training datasets.

Validation dataset settings:
```bash
(line 42)   all_img_path: ['/home/notebook/data/group/chendu/dataset/DiffIQA/Validation/images',
                    '/home/notebook/data/group/chendu/dataset/IQA/KADID10K/images',
                    '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/images',
                    '/home/notebook/data/group/chendu/dataset/IQA/TID2013/images']
            ThreeAFC_list_path: ['/home/notebook/data/group/chendu/dataset/DiffIQA/Validation/TripletEachType/PNY.txt',
                            '/home/notebook/data/group/chendu/dataset/IQA/KADID10K/Validation/Triplet/Triplet.txt',
                            '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/Validation/Triplet/Triplet.txt',
                            '/home/notebook/data/group/chendu/dataset/IQA/TID2013/Validation/Triplet/Triplet.txt']

```
You are supposed to indicate the image paths and triplet information of DiffIQA, KADID10K, PIPAL and TID2013. For more details, please refer to this [instruction](datasets/README.md) to prepare the validation datasets.


Pretrained CLIP ViT-b-32.pt:
```bash
(line 81)   path_CLIP:
                    pretrain_CLIP_path: /home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt
                    mode: 'Original'
                    finetune_CLIP_path: ~
```
You are supposed to download the pretrained CLIP ```ViT-b-32.pt``` model.

### 3.2 Stage 2
In this stage, we will train the fidelity term for A-FINE. As for CLIP backbone, we directly load the well-trained clip_model.pth from Stage 1 and just fix its parameters. The training datasets include KADID10K, PIPAL and TID2013. The training command is as follows, which is also written in ```demo.sh```

```bash
cd TrainAFINE/
### For single GPU training
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage2_nlogn.yml --auto_resume

### For DDP training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage2_nlogn.yml --launcher pytorch --auto_resume
```

in ```options/train/AFINE/train_AFINE_stage2_nlogn.yml```, you need to modify the training dataset settings ,validation dataset settings, the pretrained CLIP ```ViT-b-32.pt``` model and the finetuned ```clip_model.pth``` which is trained in Stage 1:

Training dataset settings:
```bash
(line 12)   all_img_path: ['/home/notebook/data/group/chendu/dataset/IQA/KADID10K/images',
                   '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/images',
                   '/home/notebook/data/group/chendu/dataset/IQA/TID2013/images']
            ThreeAFC_list_path: ['/home/notebook/data/group/chendu/dataset/IQA/KADID10K/Train/Triplet/Triplet.txt',
                         '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/Train/Triplet/Triplet.txt',
                         '/home/notebook/data/group/chendu/dataset/IQA/TID2013/Train/Triplet/Triplet.txt']
```
You are supposed to indicate the image paths and triplet information of KADID10K, PIPAL and TID2013. For more details, please refer to this [instruction](datasets/README.md) to prepare the training datasets.

Validation dataset settings:
```bash
(line 39)   all_img_path: ['/home/notebook/data/group/chendu/dataset/IQA/KADID10K/images',
                   '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/images',
                   '/home/notebook/data/group/chendu/dataset/IQA/TID2013/images']
            ThreeAFC_list_path: ['/home/notebook/data/group/chendu/dataset/IQA/KADID10K/Validation/Triplet/Triplet.txt',
                         '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/Validation/Triplet/Triplet.txt',
                         '/home/notebook/data/group/chendu/dataset/IQA/TID2013/Validation/Triplet/Triplet.txt']

```
You are supposed to indicate the image paths and triplet information of DiffIQA, KADID10K, PIPAL and TID2013. For more details, please refer to this [instruction](datasets/README.md) to prepare the validation datasets.


Pretrained CLIP ```ViT-b-32.pt``` and the finetuned ```clip_model.pth``` model in Stage 1:
```bash
(line 81)     path_CLIP:
                    pretrain_CLIP_path: /home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt
                    mode: 'Original'
                    finetune_CLIP_path: experiments/pretrained_models/AFINE_stage1_nlogn/clip_model.pth
```
You are supposed to download the pretrained CLIP ViT-b-32.pt model, and you should also indicate the finetuned CLIP path.


### 3.3 Stage 3
In this stage, we will train the non-linear mapping and adaptive term for A-FINE. The training datasets include DiffIQA,  KADID10K, PIPAL and TID2013. 
As for CLIP backbone, we directly load the finetuned clip_model.pth from Stage 1 and just fix its parameters. 
As for the naturalness term, we directly load the well-trained net_qhead.pth from Stage 1 and just fix its parameters. 
As for the fidelity term, we directly load the well-trained net_dhead.pth from Stage 2 and just fix its parameters. 

#### 3.3.1 Give reasonable initial values for the non-linear mapping function
Due to the excessively wide range of naturalness term, we need to give a reasonable inital value for $\gamma_{3}, \gamma_{4}$ (in Equation.8 in our main paper), so as to accelerate the convergence of the training progress, through a speific fitting strategy. We fit the initial value on the training parts of PIPAL dataset. Here are two steps you need to do,

(1). Normalize the original MOS scores of PIPAL to $(-2, 2)$
```bash
cd TrainAFINE/scripts/fit_parameter/normalize_mos
python normalize_PIPAL_mos.py --input_path [path to the original MOS scores provided by us] --save_path [path to your save path]
```
This step will save the normalized MOS scores to your save path.

(2). Fit the initial value of $\gamma_{3}, \gamma_{4}$
```bash
cd TrainAFINE/scripts/fit_parameter/fit_beta
python fit_beta_AFINE_Stage1.py [please change the argument parameters according to your own path]
```
This step will output the initial value.

Note that, for the initial value of $\eta_{3}, \eta_{4}$ in fidelity terms (in Equation.7 in our main paper), since the range of fidelity term is $(0, 1)$, we don't need to fit them on other datasets. Instead, we just set $\eta_{3} = 0.5, \eta_{4} = 0.15$.

#### 3.3.2 Train Stage 3
The training command is as follows, which is also written in ```demo.sh```

```bash
cd TrainAFINE/
### For single GPU training
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage3_nlogn.yml --auto_resume

### For DDP training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage3_nlogn.yml --launcher pytorch --auto_resume
```

in ```options/train/AFINE/train_AFINE_stage3_nlogn.yml```, you need to modify the training dataset settings ,validation dataset settings, the pretrained CLIP ```ViT-b-32.pt``` model, the finetuned ```clip_model.pth``` model which is trained in Stage 1, the pretrained naturalness term in Stage1, the pretrained fidelity term in Stage 2:

Training dataset settings:
```bash
(line 12)   all_img_path: ['/home/notebook/data/group/chendu/dataset/DiffIQA/Train/images',
                        '/home/notebook/data/group/chendu/dataset/IQA/KADID10K/images',
                        '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/images',
                        '/home/notebook/data/group/chendu/dataset/IQA/TID2013/images']
            ThreeAFC_list_path: ['/home/notebook/data/group/chendu/dataset/DiffIQA/Train/trainlabel/Triplet/Triplet.txt',
                                '/home/notebook/data/group/chendu/dataset/IQA/KADID10K/Train/Triplet/Triplet.txt',
                                '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/Train/Triplet/Triplet.txt',
                                '/home/notebook/data/group/chendu/dataset/IQA/TID2013/Train/Triplet/Triplet.txt']
```
You are supposed to indicate the image paths and triplet information of KADID10K, PIPAL and TID2013. For more details, please refer to this [instruction](datasets/README.md) to prepare the training datasets.

Validation dataset settings:
```bash
(line 41)   all_img_path: ['/home/notebook/data/group/chendu/dataset/DiffIQA/Validation/images',
                    '/home/notebook/data/group/chendu/dataset/IQA/KADID10K/images',
                    '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/images',
                    '/home/notebook/data/group/chendu/dataset/IQA/TID2013/images']
            ThreeAFC_list_path: ['/home/notebook/data/group/chendu/dataset/DiffIQA/Validation/TripletEachType/PNY.txt',
                            '/home/notebook/data/group/chendu/dataset/IQA/KADID10K/Validation/Triplet/Triplet.txt',
                            '/home/notebook/data/group/chendu/dataset/IQA/PIPAL/Validation/Triplet/Triplet.txt',
                            '/home/notebook/data/group/chendu/dataset/IQA/TID2013/Validation/Triplet/Triplet.txt']

```
You are supposed to indicate the image paths and triplet information of DiffIQA, KADID10K, PIPAL and TID2013. For more details, please refer to this [instruction](datasets/README.md) to prepare the validation datasets.


Pretrained ```CLIP ViT-b-32.pt``` and the finetuned ```clip_model.pth``` model in Stage 1:
```bash
(line 128)     path_CLIP:
                    pretrain_CLIP_path: /home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt
                    mode: 'Original'
                    finetune_CLIP_path: experiments/pretrained_models/AFINE_stage1_nlogn/clip_model.pth
```
You are supposed to download the pretrained ```CLIP ViT-b-32.pt``` model, and you should also indicate the path of finetuned ```clip_model.pth```.

Pretrained naturalness term in Stage 1:
```bash
(line 102)     path_qhead:
                    pretrain_network_qhead: experiments/pretrained_models/AFINE_stage1_nlogn/net_qhead.pth
                    strict_load_qhead: True
                    param_key_qhead: params
```
You are supposed to indicate the path of pretrained ```net_qhead.pth`` model.

Pretrained fidelity term in Stage 2:
```bash
(line 107)     path_dhead:
                    pretrain_network_dhead: experiments/pretrained_models/AFINE_stage2_nlogn/net_dhead.pth
                    strict_load_dhead: True
                    param_key_dhead: params
```
You are supposed to indicate the path of pretrained ```net_dhead.pth`` model.


AS fo the initial value of $\eta_{3}, \eta_{4}, \gamma_{3}, \gamma_{4}$, we have already give it here, you could either use the values provided by us, or just change the initial values of $\gamma_{3}, \gamma_{4}$ to the values fit by yourself.

```bash
           network_scale_nr:
                type: AFINENLM_NR_Fit
                yita1: 2
                yita2: -2
(line 82)       yita3: 4.9592
(line 83)       yita4: 21.5968

           network_scale_fr:
                type: AFINENLM_FR_Fit_with_limit
                yita1: 2
                yita2: -2
(line 89)       yita3: 0.5
(line 90)       yita4: 0.15
                yita3_upper: 0.95
                yita3_lower: 0.05
                yita4_upper: 0.70
                yita4_lower: 0.01

```

# Testing details of A-FINE
We provide the testing code for the three different training stages.

## 1. Pretrained models

|           Stage           |lDowload|
|:------------------------:|:---:|
|CLIP ViT-b-32.pt|[OPENAI](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)|
| Stage 1 |[Google Drive](https://drive.google.com/drive/folders/1TCiecDmQGlw0U0tdOirH97WHXnWFXEqc?usp=drive_link)|
|        Stage 2        |[Google Drive](https://drive.google.com/drive/folders/1rDcwE5bB69D4nxD2oCQzNQPELxQXVx1v?usp=sharing)|
|      Stage 3       |[Google Drive](https://drive.google.com/drive/folders/1wAmonXCckwYCTi3nrRoOi5fYenULk1B1?usp=sharing)|


## 2. Testing A-FINE
### 2.1 Stage 1
In this stage, we will test the naturalness term on different testing datasets include SRIQA-Bench, DiffIQA, KADID10K, PIPAL and TID2013. The testing command is as follows, which is also written in ```demo.sh```

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/AFINE/test_AFINE_stage1_nlogn.yml
```

in ```options/test/AFINE/test_AFINE_stage1_nlogn.yml```, you need to modify the testing dataset settings, the pretrained CLIP ```ViT-b-32.pt``` model and the finetuned ```clip_model.pth```, as well as the naturalness term:

Testing dataset settings:
```bash
(line 9)    test_1:
                name: SRIQA-Bench-All
                type: SRIQADataset
                all_img_path: /home/notebook/data/group/chendu/dataset/SRIQA-Bench/images
                all_score_path: /home/notebook/data/group/chendu/dataset/SRIQA-Bench/MOS
                io_backend:
                type: disk
                use_hflip: false
                use_rot: false
                mean: [0.48145466, 0.4578275, 0.40821073]
                std: [0.26862954, 0.26130258, 0.27577711]
...
```
You are supposed to indicate the image paths and triplet information of DiffIQA, KADID10K, PIPAL and TID2013. For SRIQA-Bench, you need to indicate the image paths and the MOS score path. For more details, please refer to this [instruction](datasets/README.md) to prepare the training datasets.


Pretrained CLIP ViT-b-32.pt and the finetuned clip_model.pth:
```bash
(line 164)  path_CLIP:
                pretrain_CLIP_path: /home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt
                mode: 'finetune' # if is set to 'finetune', the should provide the finetuned CLIP model path bellow
                finetune_CLIP_path: experiments/pretrained_models/AFINE_stage1_nlogn/clip_model.pth
```
You are supposed to download the pretrained CLIP ```ViT-b-32.pt``` model and the finetuned ```clip_model.pth```.



The well-trained naturalness term:
```bash
(line 158)  path_qhead:
                pretrain_network_qhead: experiments/pretrained_models/AFINE_stage1_nlogn/net_qhead.pth
                strict_load_qhead: True
                param_key_qhead: params
```
You are supposed to download the well-trained ```net_qhead.pth```.


### 2.2 Stage 2
In this stage, we will test the fidelity term on different testing datasets include SRIQA-Bench, DiffIQA, KADID10K, PIPAL and TID2013. The testing command is as follows, which is also written in ```demo.sh```

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/AFINE/test_AFINE_stage2_nlogn.yml
```

in ```options/test/AFINE/test_AFINE_stage2_nlogn.yml```, you need to modify the testing dataset settings, the pretrained CLIP ```ViT-b-32.pt``` model and the finetuned ```clip_model.pth```, as well as the fidelity term:

Testing dataset settings:
```bash
(line 7)    test_1:
                name: SRIQA-Bench-All
                type: SRIQADataset
                all_img_path: /home/notebook/data/group/chendu/dataset/SRIQA-Bench/images
                all_score_path: /home/notebook/data/group/chendu/dataset/SRIQA-Bench/MOS
                io_backend:
                type: disk
                use_hflip: false
                use_rot: false
                mean: [0.48145466, 0.4578275, 0.40821073]
                std: [0.26862954, 0.26130258, 0.27577711]
...
```
You are supposed to indicate the image paths and triplet information of DiffIQA, KADID10K, PIPAL and TID2013. For SRIQA-Bench, you need to indicate the image paths and the MOS score path. For more details, please refer to this [instruction](datasets/README.md) to prepare the training datasets.


Pretrained ```CLIP ViT-b-32.pt``` and the finetuned ```clip_model.pth```:
```bash
(line 159)  path_CLIP:
                pretrain_CLIP_path: /home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt
                mode: 'finetune' # if is set to 'finetune', the should provide the finetuned CLIP model path bellow
                finetune_CLIP_path: experiments/pretrained_models/AFINE_stage1_nlogn/clip_model.pth
```
You are supposed to download the pretrained CLIP ```ViT-b-32.pt model``` and the finetuned ```clip_model.pth```.



The well-trained fidelity term:
```bash
(line 153)  path_dhead:
                pretrain_network_dhead: experiments/pretrained_models/AFINE_stage2_nlogn/net_dhead.pth
                strict_load_dhead: True
                param_key_dhead: params
```
You are supposed to download the well-trained ```net_dhead.pth```.



### 2.3 Stage 3
In this stage, we will test the whole A-FINE model on different testing datasets include SRIQA-Bench, DiffIQA, KADID10K, PIPAL and TID2013. The testing command is as follows, which is also written in ```demo.sh```

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/AFINE/test_AFINE_stage3_nlogn.yml
```

in ```options/test/AFINE/test_AFINE_stage3_nlogn.yml```, you need to modify the testing dataset settings, the pretrained CLIP ```ViT-b-32.pt``` model and the finetuned ```clip_model.pth```, the naturalness term, the fidelity term, the non-linear mapping and adaptive term:

Testing dataset settings:
```bash
(line 7)    test_1:
                name: SRIQA-Bench-All
                type: SRIQADataset
                all_img_path: /home/notebook/data/group/chendu/dataset/SRIQA-Bench/images
                all_score_path: /home/notebook/data/group/chendu/dataset/SRIQA-Bench/MOS
                io_backend:
                type: disk
                use_hflip: false
                use_rot: false
                mean: [0.48145466, 0.4578275, 0.40821073]
                std: [0.26862954, 0.26130258, 0.27577711]
...
```
You are supposed to indicate the image paths and triplet information of DiffIQA, KADID10K, PIPAL and TID2013. For SRIQA-Bench, you need to indicate the image paths and the MOS score path. For more details, please refer to this [instruction](datasets/README.md) to prepare the training datasets.


Pretrained ```CLIP ViT-b-32.pt``` and the finetuned ```clip_model.pth```:
```bash
(line 210)  path_CLIP:
                pretrain_CLIP_path: /home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt
                mode: 'finetune' # if is set to 'finetune', the should provide the finetuned CLIP model path bellow
                finetune_CLIP_path: experiments/pretrained_models/AFINE_stage1_nlogn/clip_model.pth
```
You are supposed to download the pretrained CLIP ```ViT-b-32.pt``` model and the finetuned ```clip_model.pth```.

The well-trained naturalness term:
```bash
(line 184)  path_qhead:
                pretrain_network_qhead: experiments/pretrained_models/AFINE_stage1_nlogn/net_qhead.pth
                strict_load_qhead: True
                param_key_qhead: params
```
You are supposed to download the well-trained ```net_qhead.pth```.

The well-trained fidelity term:
```bash
(line 189)  path_dhead:
                pretrain_network_dhead: experiments/pretrained_models/AFINE_stage2_nlogn/net_dhead.pth
                strict_load_dhead: True
                param_key_dhead: params
```
You are supposed to download the well-trained ```net_dhead.pth```.

The non-linear mapping for naturalness term and fidelity term:
```bash
(line 199)  path_scale_nr:
                pretrain_network_scale_nr: experiments/pretrained_models/AFINE_stage3_nlogn/net_scale_nr.pth
                strict_load_scale_nr: True
                param_key_scale_nr: params

(line 204)  path_scale_fr:
                pretrain_network_scale_fr: experiments/pretrained_models/AFINE_stage3_nlogn/net_scale_fr.pth
                strict_load_scale_fr: True
                param_key_scale_fr: params
```
You are supposed to download the well-trained ```net_scale_nr.pth``` and ```net_scale_fr.pth```.


The adaptive term:
```bash
(line 194)  path_finalscore:
                pretrain_network_finalscore: experiments/pretrained_models/AFINE_stage3_nlogn/net_finalscore.pth
                strict_load_finalscore: True
                param_key_finalscore: params
```
You are supposed to download the well-trained ```net_finalscore.pth```.