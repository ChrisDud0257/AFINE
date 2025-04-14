# pip install diffusers==0.21.4 -i https://mirrors.aliyun.com/pypi/simple

# TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 20655 \
# 	train_pasd_enhancer.py \
#  	--pretrained_model_name_or_path="checkpoints/stable-diffusion-v1-5" \
# 	--output_dir="runs/pasd_enhancer" \
# 	--resolution=512 \
# 	--learning_rate=5e-5 \
# 	--gradient_accumulation_steps=4 \
# 	--train_batch_size=4 \
# 	--num_train_epochs=10 \
# 	--tracker_project_name="pasd_enhancer" \
# 	--enable_xformers_memory_efficient_attention \
# 	--checkpointing_steps=10000 \
# 	--control_type="realisr" \
# 	--mixed_precision="no" \
# 	--dataloader_num_workers=1 \
# 	--degradation_param_path="/home/notebook/code/personal/S9053766/chendu/FinalUpload/AFINE/DiffEnhancer/dataloader/degradation_param/params_weak_degra.yml" \
# 	--gt_dir "/home/notebook/data/group/chendu/dataset/DIV8K/trainHR" \
# 	         "/home/notebook/data/group/chendu/dataset/FFHQ/trainHR1024x1024" \
# 			 "/home/notebook/data/group/chendu/dataset/DIV2K/trainHR" \
# 			 "/home/notebook/data/group/chendu/dataset/Flickr2K/trainHR" \
# 			 "/home/notebook/data/group/chendu/dataset/OST/trainHR" \
# 			 "/home/notebook/data/group/chendu/dataset/LSDIR/trainHR"


# CUDA_VISIBLE_DEVICES=0 python infer_pasd_enhancer.py \
# --pretrained_model_path /home/notebook/code/personal/S9053766/chendu/FinalUpload/AFINE/DiffEnhancer/checkpoints/stable-diffusion-v1-5 \
# --pasd_model_path /home/notebook/code/personal/S9053766/chendu/FinalUpload/AFINE/DiffEnhancer/runs/pretrained_models/pasd_enhancer/checkpoint-40000 \
# --GT_path /home/notebook/data/group/chendu/dataset/SDIQA-dataset/SDIQA-29868/images/Original \
# --output_dir /home/notebook/data/group/chendu/dataset/DiffIQA --random_shuffle --upscale_prob 0.25 \
# --param_path /home/notebook/code/personal/S9053766/chendu/FinalUpload/AFINE/DiffEnhancer/dataloader/degradation_param/params_weak_degra.yml