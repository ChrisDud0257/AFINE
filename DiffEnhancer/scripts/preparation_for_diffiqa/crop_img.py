import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import math


def compute_var(img, args):
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
    print(f"cont_var is {var[0][0]}, freq_var is {freq_var}.")
    if var[0][0] >= cont_var_thresh and freq_var >= freq_var_thresh:
        return True
    else:
        return False


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    for dataset_path in args.img_path:
        img_paths = [os.path.join(dataset_path, i) for i in os.listdir(dataset_path)]
        for img_path in img_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"img_name is {img_name}")
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img).astype(np.float32)

            h, w, _ = img_np.shape
            if h >= 2400:
                k = 64
            elif h >= 1200:
                k = 32
            else:
                k = 16
            init_h = random.randint(0, k)
            h_space = np.arange(init_h, h, args.step_size)
            # h_space_new = []
            # for i in h_space:
            #     i = max(i + random.randint(-16, 16),0)
            #     if i + args.crop_size <= h:
            #         h_space_new.append(i)

            if w >= 2400:
                k = 64
            elif w >= 1200:
                k = 32
            else:
                k = 16
            init_w = random.randint(0, k)
            w_space = np.arange(init_w, w, args.step_size)

            for x in h_space:
                for y in w_space:
                    db_x = random.randint(-16, 16)
                    db_y = random.randint(-16, 16)
                    x_ = max(x + db_x, 0)
                    y_ = max(y + db_y, 0)

                    if x_ + args.crop_size < h and y_ + args.crop_size < w:
                        img_np_crop = img_np[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
                        img_np_crop = np.ascontiguousarray(img_np_crop)

                        b_img_np_crop = compute_var(img_np_crop, args)
                        if b_img_np_crop:
                            img_np_crop = img_np_crop.astype(np.uint8)
                            img_crop = Image.fromarray(img_np_crop)
                            print(f"x_ is {x_}, y_ is {y_}, original image shape is {h}-{w}")
                            img_crop.save(os.path.join(args.save_path, f"{img_name}x{x_}y{y_}.png"), "PNG", quality = 95)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type = list, default=["/home/notebook/data/group/chendu/dataset/FilterImages/FullImages"])
    parser.add_argument("--save_path", type = str, default="/home/notebook/data/group/chendu/dataset/FilterImages/CropImages")
    parser.add_argument("--crop_size", type = int, default=512)
    parser.add_argument("--step_size", type = int, default=384)
    parser.add_argument("--cont_var_thresh", type = int, default=50)
    parser.add_argument("--freq_var_thresh", type = int, default=50)
    args = parser.parse_args()
    main(args)
