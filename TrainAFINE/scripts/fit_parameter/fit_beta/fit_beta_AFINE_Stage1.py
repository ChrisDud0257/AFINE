import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from basicsr.archs.diststype_arch import AFINEQhead
from CLIP_ReturnFea import clip
from basicsr.utils import imfromfile, img2tensor
from scipy.optimize import curve_fit

def fit_curve(x, y):
    betas_init_4params = [np.mean(x), np.std(x) / 4.]

    def logistic_4params(x, beta3, beta4):
        yhat = 4.0 / (1 + np.exp(- (x - beta3) / np.abs(beta4))) - 2.0
        return yhat

    logistic = logistic_4params
    betas_init = betas_init_4params
    betas, _ = curve_fit(logistic, x, y, p0=betas_init, maxfev=10000)
    beta3, beta4 = betas
    print(f'beta3: {beta3}, beta4: {beta4}')

    yhat = logistic(x, *betas)
    return yhat, beta3, beta4

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    # mean = torch.tensor(mean).view(1,-1,1,1).to(device)
    # std = torch.tensor(std).view(1,-1,1,1).to(device)
    # load pretrained CLIP
    clip_model, _ = clip.load(args.pretrain_CLIP_path, device="cpu", jit = False)
    finetuned_clip_checkpoint = torch.load(args.finetune_CLIP_path, map_location = 'cpu')
    clip_model.load_state_dict(finetuned_clip_checkpoint)
    clip_model = clip_model.to(device)

    net_qhead = AFINEQhead()
    net_qhead.load_state_dict(torch.load(args.qhead_path)['params'], strict=True)
    net_qhead = net_qhead.to(device)

    clip_model.eval()
    net_qhead.eval()

    os.makedirs(args.save_path, exist_ok = True)

    save_txt_path = os.path.join(args.save_path, args.save_txt_name)
    save_txt = open(save_txt_path, mode = 'w', encoding = 'utf-8')

    img_paths = args.input_img_path
    dmos_path = args.input_txt_path

    ori_dmos_list = []
    nr_score_list = []


    with open(dmos_path, mode = 'r', encoding = 'utf-8') as f:
        for line in f:
            info = line.strip().split(",")
            dis_name = str(info[0])
            ref_name = str(info[1])
            score_dis = float(info[2])

            ori_dmos_list.append(score_dis)

            dis_path = os.path.join(img_paths, dis_name)
            dis = imfromfile(path=dis_path, float32=True)
            dis = img2tensor([dis], bgr2rgb=True, float32=True)[0]
            normalize(dis, mean, std, inplace=True)
            dis = dis.unsqueeze(0)
            dis = dis.to(device)

            _,c,h,w = dis.shape
            if h % 32 != 0:
                pad_h = 32 - h % 32
            else:
                pad_h = 0

            if w % 32 != 0:
                pad_w = 32 - w % 32
            else:
                pad_w = 0

            dis = F.interpolate(dis, size = (h + pad_h, w + pad_w), mode = 'bicubic', align_corners = False)


            with torch.no_grad():
                cls_dis, feat_dis = clip_model.encode_image(dis)
                nr_score = net_qhead(dis, feat_dis)

            nr_score_list.append(nr_score.item())

            print(f"NR score for {dis_name} is {nr_score.item():.4f}, GT score is {score_dis:.4f}")

    _, beta3, beta4 = fit_curve(np.array(nr_score_list), np.array(ori_dmos_list))
    save_txt.write(f"beta3:{beta3:.4f}, beta4:{beta4:.4f}\n")
    save_txt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrain_CLIP_path',
        type=str,
        default= '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/pretrained_models/CLIP/ViT-B-32.pt')
    parser.add_argument(
        '--finetune_CLIP_path',
        type=str,
        default= '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/AFINE_stage1_nlogn/clip_model.pth')
    parser.add_argument('--qhead_path', type = str, default = '/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/experiments/AFINE_stage1_nlogn/net_qhead.pth')
    parser.add_argument('--input_img_path', type = str, default='/home/notebook/data/group/chendu/dataset/IQA/PIPAL/images')
    parser.add_argument('--input_txt_path', type = str, default='/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/scripts/fit_parameter/normalize_PIPAL_Train_mos.txt')
    parser.add_argument('--save_path', type=str, default='/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/scripts/fit_parameter', help='output folder')
    parser.add_argument('--save_txt_name', type=str, default='PIPAL_Train_AFINE_Stage1_nlogn_beta.txt', help='output folder')
    args = parser.parse_args()
    main(args)
