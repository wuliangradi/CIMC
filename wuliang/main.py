#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from time import time

from torch.utils.data import DataLoader

import log
from config import BATH_SIZE
from config import IMG_PATH, IMG_LABEL_PATH, IMG_MASK_PATH
from scripts import CarDataSet


def train():
    output_path = "/home/wuliang/wuliang/CIMC/wuliang/output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger = log.init_log(logfile="./train_log.txt", log_name="train data")
    logger.info("start")

    logger.info("训练数据集--图片存储路径是{}".format(IMG_PATH))
    logger.info("训练数据集--掩膜图片存储路径是{}".format(IMG_MASK_PATH))
    logger.info("训练数据集--标签csv路径是{}".format(IMG_LABEL_PATH))
    logger.info("读取数据集...")

    data_set = CarDataSet([IMG_PATH,
                           IMG_LABEL_PATH,
                           IMG_MASK_PATH])
    train_data_loader = DataLoader(data_set,
                                   batch_size=BATH_SIZE,
                                   shuffle=False,
                                   drop_last=True,
                                   num_workers=4)
    logger.info("train data sample counts {}".format(len(data_set)))
    logger.info("train data set batch size {}".format(BATH_SIZE))
    logger.info("train data batch counts {}".format(len(train_data_loader)))

    # data_iter = iter(train_data_loader)
    # img_tensor, label, img_mask_tensor = data_iter.next()
    # start_time_all = time()


def predict():
    pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "please input your arguments like 'train' or 'predict'"
        exit(-1)
    print "Main function"
    if sys.argv[1] == "train":
        train()
    if sys.argv[1] == "predict":
        predict()
    print "\nprogram run succeed"
