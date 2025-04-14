import os
import cv2
from PIL import Image
import numpy as np
import argparse
import random

def copy_label(src, dst, name_list):
    person_list = os.listdir(src)
    src_person_index_list = []
    dst_person_index_list= []
    for person in person_list:
        src_person = os.path.join(src, person)
        dst_person = os.path.join(dst, person)
        index_list = os.listdir(src_person)
        for index in index_list:
            src_person_index = os.path.join(src_person, index)
            src_person_index_list.append(src_person_index)

            dst_person_index = os.path.join(dst_person, index)
            os.makedirs(dst_person_index, exist_ok = True)
            dst_person_index_list.append(dst_person_index)

    for name in name_list:
        for i, src_person_index in enumerate(src_person_index_list):
            index = os.path.basename(src_person_index).split("_")[-1]
            src_person_index_path = os.path.join(src_person_index, f"{name}_{index}.json")
            dst_person_index_path = dst_person_index_list[i]
            os.system(f'cp -r {src_person_index_path} {dst_person_index_path}')





def copy_img(src, dst, name_list):
    src_original = os.path.join(src, "Original")
    index_list = os.listdir(src)
    index_list.remove("Original")
    src_index_list = []
    for index in index_list:
        src_index = os.path.join(src, index)
        src_index_list.append(src_index)

    dst_original = os.path.join(dst, "Original")
    os.makedirs(dst_original, exist_ok = True)
    dst_index_list = []
    for index in index_list:
        dst_index = os.path.join(dst, index)
        os.makedirs(dst_index, exist_ok = True)
        dst_index_list.append(dst_index)

    for name in name_list:
        src_original_path = os.path.join(src_original, f"{name}.png")
        dst_original_path = dst_original
        os.system(f'cp -r {src_original_path} {dst_original_path}')

        for i, src_index in enumerate(src_index_list):
            index = os.path.basename(src_index).split("_")[-1]
            src_index_path = os.path.join(src_index, f"{name}_{index}.png")
            dst_index_path = dst_index_list[i]
            os.system(f'cp -r {src_index_path} {dst_index_path}')


def main(args):
    save_path = args.save_path
    save_path_image_train = os.path.join(save_path, "Train", "images")
    save_path_label_train = os.path.join(save_path, "Train", "labels")
    os.makedirs(save_path_image_train, exist_ok = True)
    os.makedirs(save_path_label_train, exist_ok = True)

    save_path_image_test = os.path.join(save_path, "Test", "images")
    save_path_label_test = os.path.join(save_path, "Test", "labels")
    os.makedirs(save_path_image_test, exist_ok = True)
    os.makedirs(save_path_label_test, exist_ok = True)

    save_path_image_val = os.path.join(save_path, "Validation", "images")
    save_path_label_val = os.path.join(save_path, "Validation", "labels")
    os.makedirs(save_path_image_val, exist_ok = True)
    os.makedirs(save_path_label_val, exist_ok = True)


    image_path = os.path.join(args.original_path, "images")
    label_path = os.path.join(args.original_path, "labels")

    image_path_ori = os.path.join(image_path, "Original")
    ori_img_name_list = [os.path.splitext(i)[0] for i in os.listdir(image_path_ori)]
    # print(ori_img_name_list)
    for _ in range(3):
        random.shuffle(ori_img_name_list)

    train_length = round(len(ori_img_name_list) * 0.7)
    test_length = round(len(ori_img_name_list) * 0.2)
    val_length = len(ori_img_name_list) - train_length - test_length

    train_img_name_list = ori_img_name_list[:train_length]
    test_img_name_list = ori_img_name_list[-test_length:]
    val_img_name_list = ori_img_name_list[train_length:-test_length]

    assert len(set(train_img_name_list)) + len(set(test_img_name_list)) == len(set(train_img_name_list + test_img_name_list))
    assert len(set(train_img_name_list)) + len(set(val_img_name_list)) == len(set(train_img_name_list + val_img_name_list))
    assert len(set(test_img_name_list)) + len(set(val_img_name_list)) == len(set(test_img_name_list + val_img_name_list))

    copy_img(src = image_path, dst = save_path_image_train, name_list = train_img_name_list)
    copy_label(src = label_path, dst = save_path_label_train, name_list = train_img_name_list)

    copy_img(src = image_path, dst = save_path_image_test, name_list = test_img_name_list)
    copy_label(src = label_path, dst = save_path_label_test, name_list = test_img_name_list)

    copy_img(src = image_path, dst = save_path_image_val, name_list = val_img_name_list)
    copy_label(src = label_path, dst = save_path_label_val, name_list = val_img_name_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path', type=str, default='/home/notebook/data/group/chendu/dataset/DiffIQA', help='Path to gt (Ground-Truth)')

    parser.add_argument('--save_path', type=str, default='/home/notebook/data/group/chendu/dataset/DiffIQA-split')
    args = parser.parse_args()
    main(args)