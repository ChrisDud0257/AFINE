import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_nr_accuracy(u1, u2, gt_score, delta = 0.1, **kwargs):
    count = 0
    if u1 < u2 and abs(u1 - u2) > delta and gt_score == 1:
        count = 1
    elif u1 > u2 and abs(u1 - u2) > delta and gt_score == 0:
        count = 1
    elif abs(u1 - u2) <= delta and gt_score == 0.5:
        count = 1
    return count


@METRIC_REGISTRY.register()
def calculate_nr_accuracy_higher_better(u1, u2, gt_score, delta = 0.1, **kwargs):
    count = 0
    if u1 > u2 and abs(u1 - u2) > delta and gt_score == 1:
        count = 1
    elif u1 < u2 and abs(u1 - u2) > delta and gt_score == 0:
        count = 1
    elif abs(u1 - u2) <= delta and gt_score == 0.5:
        count = 1
    return count


@METRIC_REGISTRY.register()
def calculate_all_accuracy(u1ref_all, u2ref_all, gt_score, delta = 0.1, **kwargs):
    count = 0
    if u1ref_all < u2ref_all and abs(u1ref_all - u2ref_all) > delta and gt_score == 1:
        count = 1
    elif u1ref_all > u2ref_all and abs(u1ref_all - u2ref_all) > delta and gt_score == 0:
        count = 1
    elif abs(u1ref_all - u2ref_all) <= delta and gt_score == 0.5:
        count = 1
    return count


@METRIC_REGISTRY.register()
def calculate_all_accuracy_higher_better(u1ref_all, u2ref_all, gt_score, delta = 0.1, **kwargs):
    count = 0
    if u1ref_all > u2ref_all and abs(u1ref_all - u2ref_all) > delta and gt_score == 1:
        count = 1
    elif u1ref_all < u2ref_all and abs(u1ref_all - u2ref_all) > delta and gt_score == 0:
        count = 1
    elif abs(u1ref_all - u2ref_all) <= delta and gt_score == 0.5:
        count = 1
    return count

