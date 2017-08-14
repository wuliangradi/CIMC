#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import log


def train():
    output_path = "/Users/baidu/wuliang/CIMC/wuliang/output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger = log.init_log(logfile="train_log.txt", log_name="train data")
    logger.info("start")

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
