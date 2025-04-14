import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import math


def compute_var(img, img_name, args):
    cont_var_thresh = args.cont_var_thresh
    freq_var_thresh = args.freq_var_thresh
    if img.shape[2] == 3:
        img = Image.fromarray(img.astype(np.uint8))
        im_gray = img.convert("L")
        im_gray = np.array(im_gray)
        # im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        im_gray = img
    [_, var] = cv2.meanStdDev(im_gray.astype(np.float32))
    freq_var = cv2.Laplacian(im_gray, cv2.CV_8U).var()
    print(f"{img_name}: cont_var is {var[0][0]}, freq_var is {freq_var}.")
    if var[0][0] >= cont_var_thresh and freq_var >= freq_var_thresh:
        return True
    else:
        return False

def main(args):
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    all_img_paths = []
    for dataset_path in args.img_path:
        img_list = os.listdir(dataset_path)
        img_paths = [os.path.join(dataset_path, i) for i in img_list]
        all_img_paths = all_img_paths + img_paths

    for img_path in all_img_paths:
        print(f"img_path is {img_path}")
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32)

        b_img_np = compute_var(img_np, img_name, args)

        if b_img_np:
            os.system(f"cp -r {img_path} {save_path}")

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type = list, default=["/home/notebook/data/group/chendu/dataset/DIV2K/trainHR",
                                                            "/home/notebook/data/group/chendu/dataset/Flickr2K/trainHR"])
    parser.add_argument("--save_path", type = str, default="/home/notebook/data/group/chendu/dataset/FilterImages/FullImages")
    parser.add_argument("--cont_var_thresh", type = int, default=60)
    parser.add_argument("--freq_var_thresh", type = int, default=60)
    args = parser.parse_args()
    main(args)