import os
import argparse
import json
import numpy as np



def main(args):
    save_path = args.save_path
    os.makedirs(save_path, exist_ok = True)
    ori_img_name_list = os.listdir(args.ori_path)


    for ori_img_name in ori_img_name_list:
        ori_path = os.path.join(args.ori_path, f"{ori_img_name}")
        print(ori_img_name)
        new_path = os.path.join(save_path, f"{ori_img_name.lower()}")
        os.rename(src = ori_path, dst = new_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_path", type=str,
                        default="/home/notebook/data/group/chendu/dataset/IQA/TID2013/Original/images")
    parser.add_argument("--save_path", type = str, default="/home/notebook/data/group/chendu/dataset/IQA/TID2013/Rename/images")
    args = parser.parse_args()
    main(args)