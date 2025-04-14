import torch
import torch.nn as nn
from CLIP_ReturnFea import clip
import argparse

def main(args):
    # pretrain_clip_model = torch.load(args.pretrain_CLIP_path, map_location = 'cpu')

    # with open(args.pretrain_CLIP_path, 'rb') as opened_file:
    # try:
    #     # loading JIT archive
    #     model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
    #     state_dict = None
    # except RuntimeError:
    #     # loading saved state dict
    #     if jit:
    #         warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
    #         jit = False
    #     state_dict = torch.load(opened_file, map_location="cpu")

    finetuned_clip_model = torch.load(args.finetune_CLIP_path, map_location = 'cpu')
    natural_model = torch.load(args.qhead_path, map_location = 'cpu')['params']
    fidelity_model = torch.load(args.dhead_path, map_location = 'cpu')['params']
    fidelity_scale_model = torch.load(args.scale_fr_path, map_location = 'cpu')['params']
    natural_scale_model = torch.load(args.scale_nr_path, map_location = 'cpu')['params']
    adapter_model = torch.load(args.lambdaK_path, map_location = 'cpu')['params']

    save_dict = {}
    # pretrain_clip = {'pretrain_clip':pretrain_clip_model}
    finetuned_clip = {'finetuned_clip':finetuned_clip_model}
    natural = {'natural':natural_model}
    fidelity = {'fidelity':fidelity_model}
    fidelity_scale = {'fidelity_scale':fidelity_scale_model}
    natural_scale = {'natural_scale':natural_scale_model}
    adapter_scale = {'adapter':adapter_model}
    # save_dict.update(pretrain_clip)
    save_dict.update(finetuned_clip)
    save_dict.update(natural)
    save_dict.update(fidelity)
    save_dict.update(fidelity_scale)
    save_dict.update(natural_scale)
    save_dict.update(adapter_scale)
    torch.save(save_dict, args.save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrain_CLIP_path',
        type=str,
        default= '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt')
    parser.add_argument(
        '--finetune_CLIP_path',
        type=str,
        default= '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/AFINE_stage1_nlogn/models/clip_model.pth')
    parser.add_argument('--qhead_path', type = str, default = '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/AFINE_stage1_nlogn/models/net_qhead.pth')
    parser.add_argument('--dhead_path', type = str, default = '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/AFINE_stage2_nlogn/models/net_dhead.pth')
    parser.add_argument('--scale_fr_path', type = str, default = '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/AFINE_stage3_nlogn/models/net_scale_fr.pth')
    parser.add_argument('--scale_nr_path', type = str, default = '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/AFINE_stage3_nlogn/models/net_scale_nr.pth')
    parser.add_argument('--lambdaK_path', type = str, default = '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/AFINE_stage3_nlogn/models/net_finalscore.pth')
    parser.add_argument('--save_path', type = str, default = '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/scripts/merge_pth/afine.pth')
    args = parser.parse_args()
    main(args)
