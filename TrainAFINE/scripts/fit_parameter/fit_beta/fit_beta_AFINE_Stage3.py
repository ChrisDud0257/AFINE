import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from basicsr.archs.diststype_arch import AFINEQhead, AFINEDhead
from basicsr.archs.scale_arch import AFINENLM_NR_Fit, AFINENLM_FR_Fit, AFINENLM_FR_Fit_with_limit
from basicsr.archs.learned_hyperparam_arch import AFINELearnLambda
from CLIP_ReturnFea import clip
from basicsr.utils import imfromfile, img2tensor
from scipy.optimize import curve_fit

def fit_curve(x, y):
    betas_init_4params = [np.mean(x), np.std(x) / 4.]

    def logistic_4params(x, beta3, beta4, beta1 = 100, beta2 = 0):
        yhat = []
        for value in x:
            exp_pow = -1 * (value - beta3) / (np.abs(beta4) + 1e-10)
            if exp_pow >=10:
                y = (beta1 - beta2) * np.exp(-1 * exp_pow) / (1 + np.exp(-1 * exp_pow)) + beta2
            else:
                y = (beta1 - beta2) / (1 + np.exp(exp_pow)) + beta2
            yhat.append(y)

        # yhat = (beta1 - beta2) / (1 + np.exp(- (x - beta3) / np.abs(beta4))) + beta2
        # print(f"yhat is {yhat}")
        return yhat

    logistic = logistic_4params
    betas_init = betas_init_4params
    betas, _ = curve_fit(logistic, x, y, p0=betas_init, maxfev=10000)
    beta3, beta4 = betas
    print(f'beta3: {beta3}, beta4: {beta4}')

    yhat = logistic(x, *betas)
    return yhat, beta3, beta4


def main(args):
    os.makedirs(args.save_path, exist_ok = True)

    save_txt_path = os.path.join(args.save_path, args.save_txt_name)
    save_txt = open(save_txt_path, mode = 'w', encoding = 'utf-8')

    img_paths = args.input_img_path
    dmos_path = args.input_txt_path

    ori_dmos_list = []
    fr_score_list = []

    afine_path = '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/scripts/merge_pth/afine.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    # load pretrained CLIP
    clip_model, _ = clip.load('/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt', device="cpu", jit = False)
    finetuned_clip_checkpoint = torch.load(afine_path, map_location = 'cpu')['finetuned_clip']
    clip_model.load_state_dict(finetuned_clip_checkpoint)
    clip_model = clip_model.to(device)

    net_qhead = AFINEQhead()
    net_qhead.load_state_dict(torch.load(afine_path, map_location = 'cpu')['natural'], strict=True)
    net_qhead = net_qhead.to(device)

    net_dhead = AFINEDhead()
    net_dhead.load_state_dict(torch.load(afine_path, map_location = 'cpu')['fidelity'], strict=True)
    net_dhead = net_dhead.to(device)

    net_scale_fr = AFINENLM_FR_Fit_with_limit(yita1=2,yita2=-2,yita3=0.5,yita4=0.15,
                                              yita3_upper=0.95,yita3_lower=0.05,yita4_upper=0.70,yita4_lower=0.01)
    net_scale_fr.load_state_dict(torch.load(afine_path, map_location = 'cpu')['fidelity_scale'], strict=True)
    net_scale_fr = net_scale_fr.to(device)

    net_scale_nr = AFINENLM_NR_Fit(yita1 = 2, yita2 = -2, yita3 = 4.9592, yita4 = 21.5968)
    net_scale_nr.load_state_dict(torch.load(afine_path, map_location = 'cpu')['natural_scale'], strict=True)
    net_scale_nr = net_scale_nr.to(device)

    adapter = AFINELearnLambda(k=5)
    adapter.load_state_dict(torch.load(afine_path, map_location = 'cpu')['adapter'], strict=True)
    adapter = adapter.to(device)


    clip_model.eval()
    net_qhead.eval()
    net_dhead.eval()
    net_scale_fr.eval()
    net_scale_nr.eval()
    adapter.eval()

    img_name_list = []

    with open(dmos_path, mode = 'r', encoding = 'utf-8') as f:
        for line in f:
            info = line.strip().split(",")
            dis_name = str(info[0])
            ref_name = str(info[1])
            score_dis = float(info[2])

            ori_dmos_list.append(score_dis)

            dis_path = os.path.join(img_paths, dis_name)
            ref_path = os.path.join(img_paths, ref_name)

            dis = imfromfile(path=dis_path, float32=True)
            dis = img2tensor([dis], bgr2rgb=True, float32=True)[0]
            normalize(dis, mean, std, inplace=True)
            dis = dis.unsqueeze(0)
            dis = dis.to(device)

            ref = imfromfile(path=ref_path, float32=True)
            ref = img2tensor([ref], bgr2rgb=True, float32=True)[0]
            normalize(ref, mean, std, inplace=True)
            ref = ref.unsqueeze(0)
            ref = ref.to(device)

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

            fr_score_list.append(afine_all.item())

            print(f"FR score for {dis_name} is {afine_all.item():.4f}, GT score is {score_dis:.4f}")
            if afine_all.item() <= -50:
                img_name_list.append([dis_name])

    _, beta3, beta4 = fit_curve(np.array(fr_score_list), np.array(ori_dmos_list))
    print(f"Min FR score is {min(fr_score_list)}, Max FR score is {max(fr_score_list)}")
    save_txt.write(f"beta3:{beta3:.4f}, beta4:{beta4:.4f}\n")
    save_txt.close()
    print(img_name_list, len(img_name_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_path', type = str, default='/home/notebook/data/group/chendu/dataset/SR-Testing-Dataset/The4thRound/SR-Testing-Dataset-200/PatchImgs')
    parser.add_argument('--input_txt_path', type = str, default='/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/scripts/fit_parameter/normalize_results/normalize_SRIQA_mos.txt')
    parser.add_argument('--save_path', type=str, default='/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/scripts/fit_parameter/fit_beta_results', help='output folder')
    parser.add_argument('--save_txt_name', type=str, default='SRIQA_AFINE_Stage3_nlogn_beta.txt', help='output folder')
    args = parser.parse_args()
    main(args)
