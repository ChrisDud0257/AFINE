# Training Enhancer

We simplify the well-known [PASD](https://github.com/yangxy/PASD) project as our pipeline.

## 1.Installation

- Python == 3.10
- CUDA == 11.8
- pytorch_lightning
- Anaconda

```bash
git clone https://github.com/ChrisDud0257/AFINE
cd DiffEnhancer
conda create --name diffenhancer python=3.10
pip install -r requirements.txt
```

All of the models are trained under the excellent [PASD](https://github.com/yangxy/PASD) framework. For any installation issues,
please refer to [PASD](https://github.com/yangxy/PASD).

## 2.Preliminary

Firstly, please download Stable-diffusion-v1.5 from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), and put them in ```checkpoints/stable-diffusion-v1-5/```.

|       Model       |                           Link                           |
| :------------------: | :---------------------------------------------------------: |
| Stable-diffusion-v1.5 | [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
|DiffEnhancer|[Google Drive](https://drive.google.com/drive/folders/1eW3H1eiFPnqFgstr6CQmWiE3X3435Vyp?usp=sharing)|

Secondly, please prepare the training dataset by following this [instruction](datasets/README.md).

## 3.Training commands

Please run the following codes which are also written in [demo.sh](demo.sh),

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 20655 \
	train_pasd_enhancer.py \
 	--pretrained_model_name_or_path="checkpoints/stable-diffusion-v1-5" \
	--output_dir="runs/pasd_enhancer" \
	--resolution=512 \
	--learning_rate=5e-5 \
	--gradient_accumulation_steps=4 \
	--train_batch_size=4 \
	--num_train_epochs=10 \
	--tracker_project_name="pasd_enhancer" \
	--enable_xformers_memory_efficient_attention \
	--checkpointing_steps=10000 \
	--control_type="realisr" \
	--mixed_precision="no" \
	--dataloader_num_workers=1 \
	--degradation_param_path="/home/notebook/code/personal/S9053766/chendu/FinalUpload/AFINE/DiffEnhancer/dataloader/degradation_param/params_weak_degra.yml" \
	--gt_dir "/home/notebook/data/group/chendu/dataset/DIV8K/trainHR" \
	         "/home/notebook/data/group/chendu/dataset/FFHQ/trainHR1024x1024" \
			 "/home/notebook/data/group/chendu/dataset/DIV2K/trainHR" \
			 "/home/notebook/data/group/chendu/dataset/Flickr2K/trainHR" \
			 "/home/notebook/data/group/chendu/dataset/OST/trainHR" \
			 "/home/notebook/data/group/chendu/dataset/LSDIR/trainHR"

```

Please note that, you should indicate ```--degradation_param_path``` to ```dataloader/degradation_param/params_weak_degra.yml```.
And you should indicate ```--gt_dir``` to your own GT paths.
We train our Enhancer on 8 NVIDIA V100 GPUs.

## 4. Commands for inferring DiffIQA

Firstly, please download our well-trained Diffusion-based Enhancer,


|       Model       |                           Link                           |
| :------------------: | :---------------------------------------------------------: |
| DiffEnhancer | [Google Drive](https://drive.google.com/drive/folders/1eW3H1eiFPnqFgstr6CQmWiE3X3435Vyp?usp=sharing) |

Secondly, please prepare the cropped $512 \times 512$ image patches (which will serve as the input in inference stage) by following this [instruction](datasets/README.md).
Thirdly, please feed the cropped $512 \times 512$ image patches into Enhancer, use the following command which are also written in [demo.sh](demo.sh),
```bash
CUDA_VISIBLE_DEVICES=0 python infer_pasd_enhancer.py \
--pretrained_model_path /home/notebook/code/personal/S9053766/chendu/FinalUpload/AFINE/DiffEnhancer/checkpoints/stable-diffusion-v1-5 \
--pasd_model_path /home/notebook/code/personal/S9053766/chendu/FinalUpload/AFINE/DiffEnhancer/runs/pretrained_models/pasd_enhancer/checkpoint-40000 \
--GT_path /home/notebook/data/group/chendu/dataset/CropImages/Original \
--output_dir /home/notebook/data/group/chendu/dataset/OutputDiffIQA --random_shuffle --upscale_prob 0.25 \
--param_path /home/notebook/code/personal/S9053766/chendu/FinalUpload/AFINE/DiffEnhancer/dataloader/degradation_param/params_weak_degra.yml
```
Please note that, you should indicate ```--param_path``` to ```dataloader/degradation_param/params_weak_degra.yml```.
Please indicate ```-pasd_model_path``` to our provided well-trained DiffEnhancer.
Please indicate ```--GT_path``` to the cropped $512 \times 512$ image patches.
Please indicate ```--output_dir``` to save the generated DiffIQA dataset.
For each original reference $512 \times 512$ image patch, we randomly select different settings to generate 6 different outputs. To accelerate the inference stage, We parallerly infer the total DiffIQA dataset on 20 NVIDIA V100 GPUs.
**Please note that, the seed is totally unfixed during the inference stage, so your inference results might have some differences with us.**