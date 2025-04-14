from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import random

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir, imfromfile
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment
import os


def fetch_img_path_3AFC_3GTscore_nobalance(ThreeAFC_list_path, all_img_path, seed = 2, count_percentage = 0.5):

    all_img_path_set = set(all_img_path)
    all_3GTscore_list = []

    for list_path in ThreeAFC_list_path:
        with open(list_path, mode = 'r', encoding = 'utf-8') as f:
            for line in f:
                info = line.strip().split(",")
                img1_name = str(info[0])
                img2_name = str(info[1])
                ref_name = str(info[2])
                gt_score_12 = float(info[3])


                if 'SDIQA' in list_path:
                    dataset_path = [element for element in all_img_path_set if 'SDIQA' in element][0]

                    if "_" in img1_name:
                        img1_index = os.path.splitext(img1_name)[0].split("_")[-1]
                    else:
                        img1_index = "Original"

                    if "_" in img2_name:
                        img2_index = os.path.splitext(img2_name)[0].split("_")[-1]
                    else:
                        img2_index = "Original"

                    img1_path = os.path.join(dataset_path, img1_index, img1_name)
                    img2_path = os.path.join(dataset_path, img2_index, img2_name)
                    ref_path = os.path.join(dataset_path, "Original", ref_name)
                    imgpath_3GTscore = [img1_path, img2_path, ref_path, gt_score_12]
                    all_3GTscore_list.append(imgpath_3GTscore)

                if 'CSIQ' in list_path:
                    dataset_path = [element for element in all_img_path_set if 'CSIQ' in element][0]
                    img1_path = os.path.join(dataset_path, img1_name)
                    if img2_name == ref_name:
                        img2_path = os.path.join(dataset_path, img2_name)
                    else:
                        img2_path = os.path.join(dataset_path, img2_name)
                    ref_path = os.path.join(dataset_path, ref_name)
                    imgpath_3GTscore = [img1_path, img2_path, ref_path, gt_score_12]
                    all_3GTscore_list.append(imgpath_3GTscore)

                if 'KADID10K' in list_path:
                    dataset_path = [element for element in all_img_path_set if 'KADID10K' in element][0]
                    img1_path = os.path.join(dataset_path, img1_name)
                    if img2_name == ref_name:
                        img2_path = os.path.join(dataset_path, img2_name)
                    else:
                        img2_path = os.path.join(dataset_path, img2_name)
                    ref_path = os.path.join(dataset_path, ref_name)
                    imgpath_3GTscore = [img1_path, img2_path, ref_path, gt_score_12]
                    all_3GTscore_list.append(imgpath_3GTscore)

                if 'PIPAL' in list_path:
                    dataset_path = [element for element in all_img_path_set if 'PIPAL' in element][0]
                    img1_path = os.path.join(dataset_path, img1_name)
                    if img2_name == ref_name:
                        img2_path = os.path.join(dataset_path, img2_name)
                    else:
                        img2_path = os.path.join(dataset_path, img2_name)
                    ref_path = os.path.join(dataset_path, ref_name)
                    imgpath_3GTscore = [img1_path, img2_path, ref_path, gt_score_12]
                    all_3GTscore_list.append(imgpath_3GTscore)


                if 'TID2013' in list_path:
                    dataset_path = [element for element in all_img_path_set if 'TID2013' in element][0]
                    img1_path = os.path.join(dataset_path, img1_name)
                    if img2_name == ref_name:
                        img2_path = os.path.join(dataset_path, img2_name)
                    else:
                        img2_path = os.path.join(dataset_path, img2_name)
                    ref_path = os.path.join(dataset_path, ref_name)
                    imgpath_3GTscore = [img1_path, img2_path, ref_path, gt_score_12]
                    all_3GTscore_list.append(imgpath_3GTscore)

    random.seed(seed)
    for i in range(4):
        random.shuffle(all_3GTscore_list)

    length_choice = int(len(all_3GTscore_list)*count_percentage)

    choice_3GTscore_list = all_3GTscore_list[:length_choice]

    return choice_3GTscore_list



@DATASET_REGISTRY.register()
class DiffIQATraIQAValDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(DiffIQATraIQAValDataset, self).__init__()
        self.opt = opt
        self.all_img_path = opt['all_img_path']
        self.ThreeAFC_list_path = opt['ThreeAFC_list_path']
        self.count_percentage = opt.get('count_percentage', 0.5)


        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.all_3AFC_pair_list = fetch_img_path_3AFC_3GTscore_nobalance(self.ThreeAFC_list_path, self.all_img_path, count_percentage = self.count_percentage)


    def __getitem__(self, index):
        threeAFC_pair = self.all_3AFC_pair_list[index]
        img1_path = threeAFC_pair[0]
        img2_path = threeAFC_pair[1]
        ref_path = threeAFC_pair[2]
        gt_score = threeAFC_pair[3]

        # print(f"img1_path is {img1_path}")

        img1 = imfromfile(path=img1_path, float32=True)
        img2 = imfromfile(path=img2_path, float32=True)
        ref = imfromfile(path=ref_path, float32=True)

        img1, img2, ref = augment([img1, img2, ref], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img1, img2, ref = img2tensor([img1, img2, ref], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img1, self.mean, self.std, inplace=True)
            normalize(img2, self.mean, self.std, inplace=True)
            normalize(ref, self.mean, self.std, inplace=True)


        return {'img1': img1, 'img2': img2, 'ref': ref, 'gt_score': gt_score}

    def __len__(self):
        return len(self.all_3AFC_pair_list)
