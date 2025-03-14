import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import math

from utils.network_archs import AFINEQhead, AFINEDhead, AFINENLM_NR_Fit, AFINENLM_FR_Fit_with_limit, AFINELearnLambda
from CLIP_ReturnFea import clip
from utils.img_utils import imfromfile, img2tensor

def scale_finalscore(score, yita1 = 100, yita2 = 0, yita3 = -1.9710, yita4 = -2.3734):

    exp_pow = -1 * (score - yita3) / (math.fabs(yita4) + 1e-10)
    if exp_pow >=10:
        scale_score = (yita1 - yita2) * torch.exp(-1 * exp_pow) / (1 + torch.exp(-1 * exp_pow)) + yita2
    else:
        scale_score = (yita1 - yita2) / (1 + torch.exp(exp_pow)) + yita2

    # scale_score = (yita1 - yita2) / (1 + math.exp(-1 * (score - yita3) / (np.abs(yita4)))) + yita2
    return scale_score

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    # load pretrained CLIP
    clip_model, _ = clip.load(args.pretrain_CLIP_path, device="cpu", jit = False)
    # load our finetuned CLIP
    finetuned_clip_checkpoint = torch.load(args.afine_path, map_location = 'cpu')['finetuned_clip']
    clip_model.load_state_dict(finetuned_clip_checkpoint)
    clip_model = clip_model.to(device)
    # load naturalness term
    net_qhead = AFINEQhead()
    net_qhead.load_state_dict(torch.load(args.afine_path, map_location = 'cpu')['natural'], strict=True)
    net_qhead = net_qhead.to(device)
    # load fidelity term
    net_dhead = AFINEDhead()
    net_dhead.load_state_dict(torch.load(args.afine_path, map_location = 'cpu')['fidelity'], strict=True)
    net_dhead = net_dhead.to(device)
    # load non-linear mapping for fidelity term
    net_scale_fr = AFINENLM_FR_Fit_with_limit(yita1=2,yita2=-2,yita3=0.5,yita4=0.15,
                                              yita3_upper=0.95,yita3_lower=0.05,yita4_upper=0.70,yita4_lower=0.01)
    net_scale_fr.load_state_dict(torch.load(args.afine_path, map_location = 'cpu')['fidelity_scale'], strict=True)
    net_scale_fr = net_scale_fr.to(device)
    # load non-linear mapping for naturalness term
    net_scale_nr = AFINENLM_NR_Fit(yita1 = 2, yita2 = -2, yita3 = 4.9592, yita4 = 21.5968)
    net_scale_nr.load_state_dict(torch.load(args.afine_path, map_location = 'cpu')['natural_scale'], strict=True)
    net_scale_nr = net_scale_nr.to(device)
    # load adptive term
    adapter = AFINELearnLambda(k=5)
    adapter.load_state_dict(torch.load(args.afine_path, map_location = 'cpu')['adapter'], strict=True)
    adapter = adapter.to(device)

    clip_model.eval()
    net_qhead.eval()
    net_dhead.eval()
    net_scale_fr.eval()
    net_scale_nr.eval()
    adapter.eval()

    # preprocess for distortion image and reference image
    dis = imfromfile(path=args.dis_img_path, float32=True)
    dis = img2tensor([dis], bgr2rgb=True, float32=True)[0]
    normalize(dis, mean, std, inplace=True)
    dis = dis.unsqueeze(0)
    dis = dis.to(device)

    ref = imfromfile(path=args.ref_img_path, float32=True)
    ref = img2tensor([ref], bgr2rgb=True, float32=True)[0]
    normalize(ref, mean, std, inplace=True)
    ref = ref.unsqueeze(0)
    ref = ref.to(device)

    # The height and width of all the images must be divisible by 32, since we utilize the pretrained CLIP ViT-B-32 model
    _,c,h,w = dis.shape
    if h % 32 != 0:
        pad_h = 32 - h % 32
    else:
        pad_h = 0

    if w % 32 != 0:
        pad_w = 32 - w % 32
    else:
        pad_w = 0

    if pad_h > 0 or pad_w > 0:
        dis = F.interpolate(dis, size = (h + pad_h, w + pad_w), mode = 'bicubic', align_corners = False)
        ref = F.interpolate(ref, size = (h + pad_h, w + pad_w), mode = 'bicubic', align_corners = False)

    # Compute A-FINE scores
    # Please note that, for all terms, including the final A-FINE score, the A-FINE fidelity/naturalness term, lower values indicate better quality
    # To prevent from numerical overflow, we use 'afine_all_scale' value to indicate the final scaled Full-reference score for (dis, ref)
    with torch.no_grad():
        cls_dis, feat_dis = clip_model.encode_image(dis)
        cls_ref, feat_ref = clip_model.encode_image(ref)
        natural_dis = net_qhead(dis, feat_dis)
        natural_ref = net_qhead(ref, feat_ref)
        natural_dis_scale = net_scale_nr(natural_dis)
        natural_ref_scale = net_scale_nr(natural_ref)

        fidelity_disref = net_dhead(dis, ref, feat_dis, feat_ref)
        fidelity_disref_scale = net_scale_fr(fidelity_disref)

        afine_all = adapter(natural_dis_scale, natural_ref_scale, fidelity_disref_scale)

        afine_all_scale = scale_finalscore(score = afine_all)

        print(f"A-FINE scaled score is {afine_all_scale.item():.4f},\n" 
              f"A-FINE score is {afine_all.item():.4f},\n" 
              f"A-FINE fidelity term is {fidelity_disref_scale.item():.4f},\n"
              f"A-FINE naturalness term for distortion image is {natural_dis_scale.item():.4f},\n"
              f"A-FINE naturalness term for reference image is {natural_ref_scale.item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### Please download the pretrained CLIP ViT-B-32.pt from their official repo: https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
    parser.add_argument(
        '--pretrain_CLIP_path',
        type=str,
        default= '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt')
    ### Please download our afine.pth model
    parser.add_argument(
        '--afine_path',
        type=str,
        default= '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/scripts/merge_pth/afine.pth')
    ###!!!Note that, you should not exchange distortion image path with reference image path, since A-FINE(dis, ref)!=A-FINE(ref, dis)
    parser.add_argument('--dis_img_path', type = str, default = '/home/notebook/data/group/chendu/dataset/SR-Testing-Dataset/The4thRound/SR-Testing-Dataset-200/images/RealESRNetx4/online20_RealESRNetx4.png', 
                        help = 'input the distortion image path')
    parser.add_argument('--ref_img_path', type = str, default = '/home/notebook/data/group/chendu/dataset/SR-Testing-Dataset/The4thRound/SR-Testing-Dataset-200/images/Original/online20_Original.png',
                        help = 'input the reference image path')
    ###!!!Note that, you should not exchange distortion image path with reference image path, since A-FINE(dis, ref)!=A-FINE(ref, dis)
    args = parser.parse_args()
    main(args)
