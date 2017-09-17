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
from PIL import Image
from PIL import ImageEnhance


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


def random_flip(img_, mask_, u=0.5):
    if np.random.random() < u:
        img_ = img_.transpose(Image.FLIP_LEFT_RIGHT)
        mask_ = mask_.transpose(Image.FLIP_LEFT_RIGHT)
    return img_, mask_


def random_rotate(img_, mask_, u=0.15):
    if np.random.random() < u:
        img_ = img_.rotate(45, Image.BILINEAR)
        mask_ = mask_.rotate(45, Image.BILINEAR)
    return img_, mask_


from numpy import linalg


def _create_coeff(
        xyA1, xyA2, xyA3, xyA4,
        xyB1, xyB2, xyB3, xyB4):
    A = np.array([
        [xyA1[0], xyA1[1], 1, 0, 0, 0, -xyB1[0] * xyA1[0], -xyB1[0] * xyA1[1]],
        [0, 0, 0, xyA1[0], xyA1[1], 1, -xyB1[1] * xyA1[0], -xyB1[1] * xyA1[1]],
        [xyA2[0], xyA2[1], 1, 0, 0, 0, -xyB2[0] * xyA2[0], -xyB2[0] * xyA2[1]],
        [0, 0, 0, xyA2[0], xyA2[1], 1, -xyB2[1] * xyA2[0], -xyB2[1] * xyA2[1]],
        [xyA3[0], xyA3[1], 1, 0, 0, 0, -xyB3[0] * xyA3[0], -xyB3[0] * xyA3[1]],
        [0, 0, 0, xyA3[0], xyA3[1], 1, -xyB3[1] * xyA3[0], -xyB3[1] * xyA3[1]],
        [xyA4[0], xyA4[1], 1, 0, 0, 0, -xyB4[0] * xyA4[0], -xyB4[0] * xyA4[1]],
        [0, 0, 0, xyA4[0], xyA4[1], 1, -xyB4[1] * xyA4[0], -xyB4[1] * xyA4[1]],
    ], dtype=np.float32)
    B = np.array([
        xyB1[0],
        xyB1[1],
        xyB2[0],
        xyB2[1],
        xyB3[0],
        xyB3[1],
        xyB4[0],
        xyB4[1],
    ], dtype=np.float32)
    return linalg.solve(A, B)


def peroective(img, mask, u=0.2):
    if np.random.random() < u:
        ran_num = [int(20 * random.normalvariate(0, 2)) for i in xrange(16)]
        coeff = _create_coeff(
            (0 + ran_num[0], 0 + ran_num[1]),
            (img.width + ran_num[2], 0 + ran_num[3]),
            (img.width + ran_num[4], img.height + ran_num[5]),
            (0 + ran_num[6], img.height + ran_num[7]),
            (ran_num[8], 0 + ran_num[9]),
            (img.width + ran_num[10], 0 + ran_num[11]),
            (img.width + ran_num[12], img.height + ran_num[13]),
            (0 + ran_num[14], img.height + ran_num[15]),
        )
        img = img.transform(
            (img.width, img.height),
            method=Image.PERSPECTIVE,
            data=coeff)
        mask = mask.transform(
            (mask.width, mask.height),
            method=Image.PERSPECTIVE,
            data=coeff)
    return img, mask


def color_enhance(img, mask, u=random.random()):
    if u >= 0.1 and u <= 0.2:
        rate_ = random.normalvariate(1, 0.1)
        img = ImageEnhance.Brightness(img).enhance(rate_)
    if u >= 0.7 and u <= 0.8:
        rate_ = random.normalvariate(1, 0.1)
        img = ImageEnhance.Color(img).enhance(rate_)
    return img, mask


if __name__ == '__main__':
    from load_data import jpg_loader, gif_loader

    img = jpg_loader("/home/wuliang/wuliang/CIMC/car_mask/dataset/train/0cdf5b5d0ce1_04.jpg")
    mask = gif_loader("/home/wuliang/wuliang/CIMC/car_mask/dataset/train_masks/0cdf5b5d0ce1_04_mask.gif")
    # img, mask = random_flip(img, mask, u=1)
    # img, mask = peroective(img, mask, u=1)
    import matplotlib.pyplot as plt

    # print ran_num
    img, mask = color_enhance(img, mask, u=0.7)
    plt.imshow(img)
    # plt.imshow(mask, alpha=0.5)
    plt.show()
