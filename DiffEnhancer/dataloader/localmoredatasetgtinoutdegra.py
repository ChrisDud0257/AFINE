import os
import glob
import torch
import random
import numpy as np
from PIL import Image, ImageFile
from functools import partial
import sys
sys.path.append('./')

from torch import nn
from torchvision import transforms
from torch.utils import data as data

from .myrealesrgan import RealESRGAN_degradation
from myutils.img_util import convert_image_to_fn

ImageFile.LOAD_TRUNCATED_IMAGES = True

def randomcrop(img, crop_size = 512):
    w,h,_ = img.shape
    # print(f"w,h is {w}-{h}")
    top_w = random.randint(0, w - crop_size)
    top_h = random.randint(0, h - crop_size)

    return img[top_w: top_w + crop_size, top_h: top_h + crop_size, :]


class LocalMoreImageDatasetGTInOutDegra(data.Dataset):
    def __init__(self,
                 gt_dir="datasets/pngtxt",
                 image_size=512,
                 tokenizer=None,
                 accelerator=None,
                 control_type="realisr",
                 null_text_ratio=0.0,
                 center_crop=False,
                 random_flip=True,
                 resize_bak=True,
                 convert_image_to="RGB",
                 opt_path = 'dataloader/degradation_param/params_weak_degra.yml'
                 ):
        super(LocalMoreImageDatasetGTInOutDegra, self).__init__()
        self.tokenizer = tokenizer
        self.control_type = control_type
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio
        self.image_size = image_size

        self.degradation = RealESRGAN_degradation(opt_path=opt_path, device='cpu')

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to, image_size)
        self.crop_preproc = transforms.Compose([
            transforms.Lambda(maybe_convert_fn),
            #transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])

        self.img_preproc = transforms.Compose([
            # transforms.Lambda(maybe_convert_fn),
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])



        img_paths = []
        for folder in gt_dir:
            for i in os.listdir(folder):
                img_path = os.path.join(folder, i)
                img_paths.append(img_path)
        self.img_paths = img_paths
        random.shuffle(self.img_paths)

    def tokenize_caption(self, caption):
        if random.random() < self.null_text_ratio:
            caption = ""

        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()

        # load image
        img_path = self.img_paths[index]
        # print(f"img_path is {img_path}")
        image = Image.open(img_path).convert('RGB')

        if 'ffhq' in img_path.lower():
            # print(f"Yes the image path is ffhq")
            random_resize = random.randint(self.image_size, self.image_size + 200)
            image = image.resize((random_resize, random_resize))

        w, h = image.size

        if min(w, h) < self.image_size:
            # print(f"Before resize, w and h is{image.size}")
            if w < h:
                ratio = float(h)/w
                another_size = int(self.image_size * ratio)
                image = image.resize((self.image_size, another_size))
            else:
                ratio = float(w)/h
                another_size = int(self.image_size * ratio)
                image = image.resize((another_size, self.image_size))
            # print(f"Ratio is {ratio}. After resize, w and h is{image.size}")


        image = np.asarray(image)

        # print(f"image size is {image.shape}")

        image = randomcrop(img=image, crop_size=self.image_size)

        image = Image.fromarray(image)


        example["pixel_values"] = self.img_preproc(image)
        if self.control_type is not None:
            if self.control_type == 'realisr':
                GT_image_t, LR_image_t, GT_image_t_only_downsample = self.degradation.degrade_process(np.asarray(image)/255., resize_bak=self.resize_bak)
                if random.random() < 0.5:
                    example["conditioning_pixel_values"] = GT_image_t_only_downsample.squeeze(0)
                    # print(f"No degradation")
                else:
                    # print(f"Yes degradation")
                    example["conditioning_pixel_values"] = LR_image_t.squeeze(0)
                example["pixel_values"] = GT_image_t.squeeze(0) * 2.0 - 1.0
            elif self.control_type == 'grayscale':
                image = np.asarray(image.convert('L').convert('RGB').astype(np.float32)) / 255.
                example["conditioning_pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
            else:
                raise NotImplementedError

        caption = ""
        if self.tokenizer is not None:
            example["input_ids"] = self.tokenize_caption(caption).squeeze(0)

        return example

    def __len__(self):
        return len(self.img_paths)