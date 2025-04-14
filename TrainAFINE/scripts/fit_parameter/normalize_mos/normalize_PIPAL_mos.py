import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torch.nn.functional as F
import csv

def normalize_list_to_range(mos_list, new_min=-2, new_max=2):
    mos_array = np.array(mos_list, dtype=float)
    old_min = mos_array.min()
    old_max = mos_array.max()
    normalized_array = (mos_array - old_min) / (old_max - old_min)
    scaled_array = normalized_array * (new_max - new_min) + new_min
    scaled_array = -1.0 * scaled_array
    return scaled_array.tolist()

def main(args):
    os.makedirs(os.path.dirname(args.save_path), exist_ok = True)

    all_data_list = []
    ori_dmos_list = []
    for list_name in os.listdir(args.input_path):
        list_path = os.path.join(args.input_path, list_name)
        with open(list_path, mode = 'r', encoding = 'utf-8') as f:
            for line in f:
                info = line.strip().split(",")
                img1_name = str(info[0])
                ref_name = str(info[1])
                mos = float(info[2])
                ori_dmos_list.append(mos)
                row_data_list = [img1_name, ref_name]
                all_data_list.append(row_data_list)

    normalized_mos_list = normalize_list_to_range(ori_dmos_list)

    save_txt = open(args.save_path, mode = 'w', encoding = 'utf-8')
    for idx, mos in enumerate(normalized_mos_list):
        new_list = all_data_list[idx]
        new_list.append(mos)
        print(new_list)
        save_txt.write(f"{new_list[0]},{new_list[1]},{new_list[2]}\n")

    save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type = str, default='/home/notebook/data/group/chendu/dataset/IQA/PIPAL/Train/MOS')
    parser.add_argument('--save_path', type=str, default='/home/notebook/code/personal/S9053766/chendu/myprojects/SDIQA/scripts/fit_parameter/normalize_PIPAL_Train_mos.txt', help='output folder')
    args = parser.parse_args()
    main(args)