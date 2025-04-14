from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir, imfromfile
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment
import os
import itertools

def fetch_2difference_effective_combination(*l):
    all_effective_combination = []
    all_combinations = list(itertools.combinations(l, 2))
    for combination in all_combinations:
        # print(combination[0][1], combination[1][1])
        if combination[0][1] > combination[1][1]:
            gt_score = 1
        elif combination[0][1] < combination[1][1]:
            gt_score = 0
        elif combination[0][1] == combination[1][1]:
            gt_score = 0.5
        all_effective_combination.append([combination[0][0], combination[1][0], gt_score])
    # print(all_effective_combination)
    return all_effective_combination

@DATASET_REGISTRY.register()
class SRIQADataset(data.Dataset):
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
        super(SRIQADataset, self).__init__()
        self.opt = opt
        self.all_img_path = opt['all_img_path']
        self.all_score_path = opt['all_score_path']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.all_2AFC_pair_list = []

        img_label_list = os.listdir(self.all_score_path)

        self.all_2AFC_pair_list = []
        for label_txt in img_label_list:
            img_score_list = []
            with open(os.path.join(self.all_score_path, label_txt), mode = 'r', encoding = 'utf-8') as f:
                for line in f:
                    if ',' in line:
                        info = line.strip().split(',')
                    else:
                        info = line.strip().split(' ')
                    # print(info)
                    img_name = str(info[0])
                    score = float(info[1])
                    pair = [img_name, score]
                    img_score_list.append(pair)
            # print(img_score_list)
            twoAFC_pair = fetch_2difference_effective_combination(*img_score_list)
            self.all_2AFC_pair_list.extend(twoAFC_pair)


    def __getitem__(self, index):
        twoAFC_pair = self.all_2AFC_pair_list[index]
        img1_name = twoAFC_pair[0]
        img2_name = twoAFC_pair[1]
        ref_name = f"{os.path.splitext(img1_name)[0].split('_')[0]}_Original.png"
        gt_score = float(twoAFC_pair[2])

        img1_algo = os.path.splitext(img1_name)[0].split("_")[1]
        img2_algo = os.path.splitext(img2_name)[0].split("_")[1]

        img_name = os.path.splitext(img1_name)[0].split("_")[0]
        assert img_name == os.path.splitext(img2_name)[0].split("_")[0], f"{img1_name} and {img2_name} has different original image name"

        img1_path = os.path.join(self.all_img_path, img1_algo, img1_name)
        img2_path = os.path.join(self.all_img_path, img2_algo, img2_name)
        ref_path = os.path.join(self.all_img_path, "Original", ref_name)
        # print(f"img1_path is {img1_path}, img2_path is {img2_path}, ref_path is {ref_path}, gt_score is {gt_score}")

        img1 = imfromfile(path=img1_path, float32=True)
        img2 = imfromfile(path=img2_path, float32=True)
        ref = imfromfile(path=ref_path, float32=True)

        # if self.opt['phase'] == 'train':
        #     img1, img2, ref = augment([img1, img2, ref], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img1, img2, ref = img2tensor([img1, img2, ref], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img1, self.mean, self.std, inplace=True)
            normalize(img2, self.mean, self.std, inplace=True)
            normalize(ref, self.mean, self.std, inplace=True)

        # print(f"img1 - ref is {(img1 - ref).sum()}")


        return {'img1': img1, 'img2': img2, 'ref': ref, 'gt_score': gt_score}

    def __len__(self):
        return len(self.all_2AFC_pair_list)
