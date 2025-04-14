# sh make_setup.sh


#AFINE_stage1_nlogn
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage1_nlogn.yml --auto_resume

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage1_nlogn.yml --launcher pytorch --auto_resume

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/AFINE/test_AFINE_stage1_nlogn.yml



#AFINE_stage2_nlogn
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage2_nlogn.yml --auto_resume

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage2_nlogn.yml --launcher pytorch --auto_resume

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/AFINE/test_AFINE_stage2_nlogn.yml




#AFINE_stage3_nlogn
# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage3_nlogn.yml --auto_resume

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 ./basicsr/train.py -opt ./options/train/AFINE/train_AFINE_stage3_nlogn.yml --launcher pytorch --auto_resume

# CUDA_VISIBLE_DEVICES=0 \
# python ./basicsr/test.py -opt ./options/test/AFINE/test_AFINE_stage3_nlogn.yml
