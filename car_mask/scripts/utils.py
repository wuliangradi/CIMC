#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    The way of AI
    Contact:wuliangwuwu@126.com
"""

import os
import random

import numpy as np
import torch


def image_to_tensor(image, mean=0, std=1.):
    """
    :param image:
    :param mean:
    :param std:
    :return:
    """
    image = image.astype(np.float32)
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(image)
    return img_tensor


def label_to_tensor(label, threshold=0.5):
    """
    :param label:
    :param threshold:
    :return:
    """
    label = (label > threshold).astype(np.float32)
    lab_tensor = torch.from_numpy(label).type(torch.FloatTensor)
    return lab_tensor


def run_length_encode(mask):
    """
    :param mask:
    :return:
    """
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def run_length_decode(rel, hidth, width, fill_value=255):
    mask = np.zeros((hidth * width), np.uint8)
    rel = np.array([int(s) for s in rel.split(' ')]).reshape(-1, 2)
    for r in rel:
        start = r[0]
        end = start + r[1]
        mask[start:end] = fill_value
    mask = mask.reshape(hidth, width)
    return mask


def split_train(length, num_valid, valid_file, train_file):
    """
    :param valid_file:
    :param train_file:
    :param length:
    :param num_valid:
    :return:
    """
    num_list = range(length)
    random.shuffle(num_list)
    fp_valid = open(valid_file, "w")
    fp_train = open(train_file, "w")

    for i in range(length):
        if i <= num_valid:
            fp_valid.writelines(str(num_list[i]) + '\n')
        else:
            fp_train.writelines(str(num_list[i]) + '\n')
    fp_valid.close()
    fp_train.close()

    return num_list[:num_valid], num_list[num_valid:]


def get_train_valid(logger, proportion_valid=0.2, num_all=None):
    if num_all is None:
        raise Exception('sample nums of all dataset is necessary.')
    valid_num = int(num_all * proportion_valid)

    logger.info("All dataset size is {}".format(num_all))
    logger.info("Train dataset size is {}".format(num_all - valid_num))

    logger.info("Valid dataset size is {}".format(valid_num))

    valid_file = "valid_list_{}".format(str(valid_num))
    train_file = "train_list_{}".format(str(num_all - valid_num))

    if not os.path.isfile(valid_file):
        split_train(num_all, valid_num, valid_file, train_file)

    valid_list = [int(x.strip().split()[0]) for x in open(valid_file).readlines()]
    train_list = [int(x.strip().split()[0]) for x in open(train_file).readlines()]
    return valid_list, train_list
