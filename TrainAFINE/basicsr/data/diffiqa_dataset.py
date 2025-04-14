from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir, imfromfile
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment
import os

@DATASET_REGISTRY.register()
class DiffIQADataset(data.Dataset):
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
        super(DiffIQADataset, self).__init__()
        self.opt = opt
        self.all_img_path = opt['all_img_path']
        self.ThreeAFC_list_path = opt['ThreeAFC_list_path']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.all_3AFC_pair_list = []
        with open(self.ThreeAFC_list_path, mode = 'r', encoding = 'utf-8') as f:
            for line in f:
                info = line.strip().split(",")
                img1_name = str(info[0])
                img2_name = str(info[1])
                ref_name = str(info[2])
                gt_score = float(info[3])
                threeAFC_pair = [img1_name, img2_name, ref_name, gt_score]
                self.all_3AFC_pair_list.append(threeAFC_pair)

    def __getitem__(self, index):
        threeAFC_pair = self.all_3AFC_pair_list[index]
        img1_name = threeAFC_pair[0]
        img2_name = threeAFC_pair[1]
        ref_name = threeAFC_pair[2]
        gt_score = threeAFC_pair[3]

        if "_" in img1_name:
            img1_index = os.path.splitext(img1_name)[0].split("_")[-1]
        else:
            img1_index = "Original"

        if "_" in img2_name:
            img2_index = os.path.splitext(img2_name)[0].split("_")[-1]
        else:
            img2_index = "Original"

        img1_path = os.path.join(self.all_img_path, img1_index, img1_name)
        img2_path = os.path.join(self.all_img_path, img2_index, img2_name)
        ref_path = os.path.join(self.all_img_path, "Original", ref_name)

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
