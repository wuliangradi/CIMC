#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Contact:wuliangwuwu@126.com
"""

from torchvision import datasets, transforms

IMG_PATH = "/home/wuliang/wuliang/CIMC/car_mask/dataset/train"
IMG_MASK_PATH = "/home/wuliang/wuliang/CIMC/car_mask/dataset/train_masks"
IMG_LABEL_PATH = "/home/wuliang/wuliang/CIMC/car_mask/dataset/train_masks.csv"

IMG_TEST_PATH = "/home/wuliang/wuliang/CIMC/car_mask/dataset/test"

BATH_SIZE = 8
NUM_EPOCHES = 3
EPOCH_VALID = 1