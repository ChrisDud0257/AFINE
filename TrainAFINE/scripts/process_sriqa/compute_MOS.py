import os
import argparse
import json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def fetch_score(label):
    if label == 'Positive':
        score = 1
    elif label == 'Negative':
        score = -1
    elif label == 'Similar':
        score = 0
    return score

def fetch_gtscore_in_2imgs(label1, label2):
    label1_score = fetch_score(label1)
    label2_score = fetch_score(label2)

    if label1_score > label2_score:
        gtscore12 = 1
        gtscore21 = 0
    elif label1_score < label2_score:
        gtscore12 = 0
        gtscore21 = 1
    elif label1_score == label2_score:
        gtscore12 = 1
        gtscore21 = 1

    return gtscore12, gtscore21

def optimize_score(C, seed=0, original_seed=20):
    np.random.seed(seed)

    C = np.array(C)
    num_images = C.shape[0]

    def objective(S):
        sum_log_diff = np.sum(C * np.log(np.maximum(norm.cdf(S[:, None] - S), 1e-6)))
        sum_squares = np.sum(S ** 2) / 2
        return -(sum_log_diff - sum_squares)

    initial_scores = np.random.rand(num_images)
    constraints = {'type': 'eq', 'fun': lambda S: np.sum(S)}
    result = minimize(objective, initial_scores, constraints=constraints)
    optimized_scores = result.x
    min_score, max_score = np.min(optimized_scores), np.max(optimized_scores)
    scaled_scores = 100 * (optimized_scores - min_score) / (max_score - min_score)
    np.random.seed(original_seed)

    return scaled_scores

def main(args):
    label_A_path = os.path.join(args.label_path, "A")
    image_name_list = os.listdir(label_A_path)

    algo_count = len(os.listdir(args.image_path))

    person_list = os.listdir(args.label_path)
    os.makedirs(args.save_path, exist_ok = True)
    for image_name in image_name_list:
        save_path_image_name_txt = os.path.join(args.save_path, f"{os.path.splitext(image_name)[0]}.txt")
        txt = open(save_path_image_name_txt, mode='w')


        adjacency_matrix = np.zeros((algo_count, algo_count), dtype=float)

        image_algo_name_set_list = []
        for person in person_list:
            person_image_name_path = os.path.join(args.label_path, person, image_name)
            person_label_list = os.listdir(person_image_name_path)
            image_algo_name_list = []

            for person_label in person_label_list:
                person_label_path = os.path.join(person_image_name_path, person_label)
                with open(person_label_path, mode='r', encoding='utf-8') as fA:
                    label_info = json.load(fA)
                img_1_name = os.path.splitext(label_info['Picture_01']['Name'])[0]
                img_2_name = os.path.splitext(label_info['Picture_02']['Name'])[0]
                image_algo_name_list.append(img_1_name)
                image_algo_name_list.append(img_2_name)

            image_algo_name_set = set(image_algo_name_list)
            image_algo_name_set_list.append(image_algo_name_set)

        for i in range(0, len(image_algo_name_set_list) - 1):
            assert image_algo_name_set_list[i] == image_algo_name_set_list[i + 1], f"{i} is not the same as {i+1}"

        image_algo_name_set = image_algo_name_set_list[0]
        for idx, person in enumerate(person_list):
            person_image_name_path = os.path.join(args.label_path, person, image_name)
            person_label_list = os.listdir(person_image_name_path)

            # print(image_algo_name_set)
            for person_label in person_label_list:
                person_label_path = os.path.join(person_image_name_path, person_label)
                with open(person_label_path, mode='r', encoding='utf-8') as fA:
                    label_info = json.load(fA)
                img_1_name = os.path.splitext(label_info['Picture_01']['Name'])[0]
                img_2_name = os.path.splitext(label_info['Picture_02']['Name'])[0]

                img_1_label = label_info['Picture_01']['Label']
                img_2_label = label_info['Picture_02']['Label']

                gtscore12, gtscore21 = fetch_gtscore_in_2imgs(img_1_label, img_2_label)

                img_1_idx_in_adjacency_matrix = list(image_algo_name_set).index(img_1_name)
                img_2_idx_in_adjacency_matrix = list(image_algo_name_set).index(img_2_name)

                adjacency_matrix[img_1_idx_in_adjacency_matrix][img_2_idx_in_adjacency_matrix] += gtscore12
                adjacency_matrix[img_2_idx_in_adjacency_matrix][img_1_idx_in_adjacency_matrix] += gtscore21
        # print(f"{image_name}, matrix is {adjacency_matrix}, sum is {adjacency_matrix.sum()}")

        image_algo_score = optimize_score(C=adjacency_matrix, seed=0)
        # print(f"{image_algo_score}")

        combined = sorted(zip(image_algo_score, list(image_algo_name_set)), reverse=True)

        sorted_image_algo_score, sorted_image_algo_name_set = zip(*combined)
        sorted_image_algo_score_list = list(sorted_image_algo_score)
        sorted_image_algo_name_list = list(sorted_image_algo_name_set)

        print(f"{sorted_image_algo_score_list}, {sorted_image_algo_name_list}")

        for i, score in enumerate(sorted_image_algo_score_list):
            txt.write(f"{sorted_image_algo_name_list[i]}.png,{score}\n")
        txt.close()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=str,
                        default=r"/home/notebook/data/group/chendu/dataset/SRIQA-Bench/labels")
    parser.add_argument("--image_path", type=str,
                        default=r"/home/notebook/data/group/chendu/dataset/SRIQA-Bench/images")
    parser.add_argument("--save_path", type = str, default=r"/home/notebook/data/group/chendu/dataset/SRIQA-Bench/MOS")
    args = parser.parse_args()
    main(args)