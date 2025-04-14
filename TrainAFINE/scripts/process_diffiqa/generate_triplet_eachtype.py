import os
import cv2
from PIL import Image
import numpy as np
import argparse
from collections import Counter
import json
import itertools


def find_max_repetition(*l):
    count = Counter(l)
    most_counterNum = count.most_common(1)
    most_element = most_counterNum[0][0]
    most_num = most_counterNum[0][1]
    return most_element, most_num

def fetch_number_for_label(label):
    if label == 'Positive':
        num = 1
    elif label == 'Similar':
        num = 0
    elif label == 'Negative':
        num = -1
    return num

def fetch_gt_label_for_2difference(label1, label2):
    label1_num = fetch_number_for_label(label1)
    label2_num = fetch_number_for_label(label2)

    if label1_num > label2_num:
        final_gt = 1
    elif label1_num < label2_num:
        final_gt = 0
    elif label1_num == label2_num == 0:
        final_gt = 0.5
    return final_gt

def fetch_gt_label_for_1element(label):
    label_num = fetch_number_for_label(label)

    if label_num  == 1:
        final_gt = 1
    elif label_num == -1:
        final_gt = 0
    elif label_num == 0:
        final_gt = 0.5
    return final_gt

def fetch_2difference_effective_combination(*l):
    all_effective_combination = []
    all_combinations = list(itertools.combinations(l, 2))
    for combination in all_combinations:
        if combination[0][1] != combination[1][1] or combination[0][1] == combination[1][1] == 'Similar':
            all_effective_combination.append([combination[0], combination[1]])
    return all_effective_combination


def fetch_1element_effective_combination(*l):
    all_combinations = list(l)
    return all_combinations

def data_statistics_2difference(*l):
    PSY_count = 0
    PNY_count = 0
    SNY_count = 0
    SSY_count = 0
    for combination in l:
        label1 = combination[0][1]
        label2 = combination[1][1]

        label_pair = [label1, label2]

        if "Positive" in label_pair and "Similar" in label_pair:
            PSY_count = PSY_count + 1
        elif "Positive" in label_pair and "Negative" in label_pair:
            PNY_count = PNY_count + 1
        elif "Similar" in label_pair and "Negative" in label_pair:
            SNY_count = SNY_count + 1
        elif label1 == label2 == "Similar":
            SSY_count = SSY_count + 1
    return PSY_count, PNY_count, SNY_count, SSY_count

def data_statistics_1element(*l):
    PYY_count = 0
    SYY_count = 0
    NYY_count = 0
    for combination in l:
        label = combination[1]
        if label == "Positive":
            PYY_count = PYY_count + 1
        elif label == "Similar":
            SYY_count = SYY_count + 1
        elif label == "Negative":
            NYY_count = NYY_count + 1
    return PYY_count, SYY_count, NYY_count


def transfer(ori):
    if ori == 'P':
        new = 'Positive'
    elif ori == 'S':
        new = 'Similar'
    elif ori == 'N':
        new = 'Negative'
    elif ori == 'Y':
        new = 'Original'
    return new


def split_type(tp):
    tp = list(tp)
    label1 = tp[0]
    label2 = tp[1]
    label3 = tp[2]

    new1 = transfer(label1)
    new2 = transfer(label2)
    new3 = transfer(label3)

    return new1, new2, new3

def main(args):
    os.makedirs(args.save_path, exist_ok = True)
    for tp in args.type_list:
        A_path = os.path.join(args.all_label_path, "A")
        B_path = os.path.join(args.all_label_path, "B")
        C_path = os.path.join(args.all_label_path, "C")

        label_name_list = [os.path.splitext(i)[0].split("_")[0] for i in os.listdir(os.path.join(A_path, "01"))]

        save_txt_path = os.path.join(args.save_path, f"{tp}.txt")
        save_txt = open(save_txt_path, 'w')

        all_count = 0

        dst_labels = list(split_type(tp))

        for label_name in label_name_list:
            idx_list = os.listdir(A_path)
            group_img_label_list = []
            for idx in idx_list:
                A_idx_path = os.path.join(A_path, idx)
                B_idx_path = os.path.join(B_path, idx)
                C_idx_path = os.path.join(C_path, idx)

                A_label_path = os.path.join(A_idx_path, f"{label_name}_{idx}.json")
                B_label_path = os.path.join(B_idx_path, f"{label_name}_{idx}.json")
                C_label_path = os.path.join(C_idx_path, f"{label_name}_{idx}.json")

                with open(A_label_path, mode='r', encoding='utf-8') as fA:
                    A_label_info = json.load(fA)
                A_label = A_label_info['Picture_AI']['Label']

                with open(B_label_path, mode='r', encoding='utf-8') as fB:
                    B_label_info = json.load(fB)
                B_label = B_label_info['Picture_AI']['Label']

                with open(C_label_path, mode='r', encoding='utf-8') as fC:
                    C_label_info = json.load(fC)
                C_label = C_label_info['Picture_AI']['Label']

                label_three = [A_label, B_label, C_label]
                img_name = f"{label_name}_{idx}"
                if len(set(label_three)) == len(label_three):
                    pass
                else:
                    element, num = find_max_repetition(*label_three)
                    img_label_list = [img_name, element]
                    group_img_label_list.append(img_label_list)

            if len(dst_labels) == len(set(dst_labels)) or dst_labels.count('Similar') == 2:
                group_2difference_effective_combination = fetch_2difference_effective_combination(*group_img_label_list)

                if len(group_2difference_effective_combination) > 0:
                    for combination in group_2difference_effective_combination:
                        label_comb = [combination[0][1], combination[1][1], 'Original']
                        if set(label_comb) == set(dst_labels):
                            final_gt_score = fetch_gt_label_for_2difference(combination[0][1], combination[1][1])
                            img1 = f"{combination[0][0]}.png"
                            img2 = f"{combination[1][0]}.png"
                            reference_img = f"{label_name}.png"
                            save_txt.write(f"{img1},{img2},{reference_img},{final_gt_score}\n")
                            all_count = all_count + 1
            else:
                assert dst_labels.count('Original') == 2, f"dst_labels is {dst_labels}"
                group_1element_effective_combination = fetch_1element_effective_combination(*group_img_label_list)
                if len(group_1element_effective_combination) > 0:
                    for combination in group_1element_effective_combination:
                        label_comb = [combination[1], 'Original', 'Original']
                        if set(label_comb) == set(dst_labels):
                            final_gt_score = fetch_gt_label_for_1element(combination[1])
                            img1 = f"{combination[0]}.png"
                            img2 = f"{label_name}.png"
                            reference_img = f"{label_name}.png"
                            save_txt.write(f"{img1},{img2},{reference_img},{final_gt_score}\n")
                            all_count = all_count + 1
        save_txt.close()
        print(f"{tp} count is {all_count}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_label_path', type = str, default='/home/notebook/data/group/chendu/DiffIQA/Train/trainlabel/labels')
    parser.add_argument('--save_path', type = str, default = '/home/notebook/data/group/chendu/DiffIQA/Train/trainlabel/TripletEachType')
    parser.add_argument('--type_list', type = list, default = ['PSY', 'PNY', 'SNY', 'SSY', 'PYY', 'SYY', 'NYY'])
    args = parser.parse_args()
    main(args)