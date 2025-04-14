import os
import cv2
from PIL import Image
import numpy as np
import argparse
from collections import Counter
import json
import itertools
import math
import random

def fetch_gt_label_for_2difference(score1, score2):
    if score1 > score2:
        final_gt = 1
    elif score1 < score2:
        final_gt = 0
    elif score1 == score2:
        final_gt = 0.5
    return final_gt

def fetch_2difference_effective_combination(*l):
    n = len(l)
    log_n = round(math.log2(n))
    all_effective_combinations = []
    for i in range(n):
        current_element = l[i]
        other_elements = l[:i]+l[i+1:]
        selected_elements = random.sample(other_elements, log_n)
        for element in selected_elements:
            all_effective_combinations.append([current_element, element])
    return all_effective_combinations

def main(args):
    train_val_test_list = os.listdir(args.all_path)

    for stage in train_val_test_list:
        stage_path = os.path.join(args.all_path, stage)
        dmos_path = os.path.join(stage_path, "MOS")
        save_path = os.path.join(stage_path, "Triplet")
        os.makedirs(save_path, exist_ok= True)

        save_txt_path = os.path.join(save_path, "Triplet.txt")
        save_txt = open(save_txt_path, 'w')

        all_label_list = os.listdir(dmos_path)
        for label in all_label_list:
            label_path = os.path.join(dmos_path, label)
            img_score_list = []
            with open(label_path, mode = 'r', encoding='utf-8') as f:
                for line in f:
                    info = line.strip().split(",")
                    dist_img = info[0]
                    score = float(info[2])
                    img_score = [dist_img, score]
                    img_score_list.append(img_score)
            ref_img = f'{os.path.splitext(label)[0]}.png'
            ref_score = [ref_img, 9999]
            img_score_list.append(ref_score)
            all_combinations = fetch_2difference_effective_combination(*img_score_list)
            for combination in all_combinations:
                dist_img1 = combination[0][0]
                dist_img2 = combination[1][0]

                dist_img1_score = float(combination[0][1])
                dist_img2_score = float(combination[1][1])
                ref_img_score = float(9999)

                gt_score_12 = fetch_gt_label_for_2difference(dist_img1_score, dist_img2_score)
                gt_score_1ref = fetch_gt_label_for_2difference(dist_img1_score, ref_img_score)
                gt_score_2ref = fetch_gt_label_for_2difference(dist_img2_score, ref_img_score)


                save_txt.write(f"{dist_img1},{dist_img2},{ref_img},{gt_score_12},{gt_score_1ref},{gt_score_2ref}\n")

        save_txt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_path', type = str, default='/home/notebook/data/sharedgroup/RG_YLab/aigc_share_group_data/chendu/dataset/IQA/KADID10K')
    args = parser.parse_args()
    main(args)