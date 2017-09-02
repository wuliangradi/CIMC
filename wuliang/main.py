#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import log
from config import BATH_SIZE
from config import IMG_PATH, IMG_LABEL_PATH, IMG_MASK_PATH
from config import NUM_EPOCHES
from net import CarUNet
from net import criterion, dice_loss
from scripts import CarDataSet
from scripts import get_train_valid
from scripts import random_resize


def predict_and_evaluate(net, test_loader, H, W):
    test_acc = 0
    test_loss = 0
    test_num = 0
    for it, (img_tensor, label, img_mask_tensor) in enumerate(test_loader, 0):
        image_ = Variable(img_tensor.cuda(), volatile=True)
        label_ = Variable(img_mask_tensor.cuda(), volatile=True)

        logits = net(image_)
        probs = F.sigmoid(logits)
        masks = (probs > 0.5).float()

        loss = criterion(logits, label_)
        acc = dice_loss(masks, label_)
        batch_size = len(img_mask_tensor)
        test_num += batch_size
        test_loss += batch_size * loss.data[0]
        test_acc += batch_size * acc.data[0]

    test_loss = test_loss / test_num
    test_acc = test_acc / test_num
    return test_loss, test_acc


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
                           IMG_MASK_PATH],
                          transform=[lambda x, y: random_resize(x, y, 1280, 1918, 512, 512)])

    valid_list, train_list = get_train_valid(logger=logger, proportion_valid=0.2, num_all=len(data_set))
    train_data_loader = DataLoader(data_set,
                                   batch_size=BATH_SIZE,
                                   sampler=train_list[:],
                                   shuffle=False,
                                   drop_last=True,
                                   num_workers=4)

    valid_data_loader = DataLoader(data_set,
                                   sampler=valid_list[:],
                                   shuffle=False,
                                   drop_last=True,
                                   num_workers=4)

    logger.info("all data sample counts {}".format(len(data_set)))
    logger.info("train data set batch size {}".format(BATH_SIZE))
    logger.info("train data sample counts {}".format(len(train_data_loader) * BATH_SIZE))
    logger.info("valid data sample counts {}".format(len(valid_data_loader)))

    num_channel = 3
    width, high = (data_set.high, data_set.width)
    net = CarUNet(in_shape=(num_channel, width, high), num_classes=1)
    net.cuda()

    logger.info("{}\n\n".format(type(net)))
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    smooth_loss = 0.0
    smooth_acc = 0.0
    sum_smooth_loss = 0
    sum_smooth_acc = 0
    sum_iter = 0
    it_smooth = 30

    epoch_valid = 1

    for epoch in range(0, NUM_EPOCHES):
        net.train()

        for it, (img_tensor, label, img_mask_tensor) in enumerate(train_data_loader, 0):
            image_ = Variable(img_tensor.cuda())
            label_ = Variable(img_mask_tensor.cuda())

            logits = net(image_)
            probs = F.sigmoid(logits)
            masks = (probs > 0.5).float()

            loss = criterion(logits, label_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = dice_loss(masks, label_)

            sum_smooth_loss += loss.data[0]
            sum_smooth_acc += acc.data[0]
            sum_iter += 1
            if it % it_smooth == 0:
                smooth_loss = sum_smooth_loss / sum_iter
                smooth_acc = sum_smooth_acc / sum_iter
                sum_smooth_loss = 0.0
                sum_smooth_acc = 0.0
                sum_iter = 0

            if it % it_smooth == 0 or it == len(train_data_loader) - 1:
                train_acc = acc.data[0]
                train_loss = loss.data[0]

                logger.info("epoch: {epoch} batch_num: {iter_num} lr: {lr} loss: {smooth_loss} acc: {smooth_acc}%  "
                            "train_loss: {train_loss} train_acc: {train_acc}%".
                            format(epoch=epoch, iter_num=it, lr=0, smooth_loss=round(smooth_loss, 4),
                                   smooth_acc=round(smooth_acc, 4) * 100, train_loss=round(train_loss, 4),
                                   train_acc=round(train_acc, 4) * 100))
                # data_iter = iter(train_data_loader)
                # img_tensor, label, img_mask_tensor = data_iter.next()
                # start_time_all = time()
        if epoch % epoch_valid == 0 or epoch == 0 or epoch == NUM_EPOCHES - 1:
            net.eval()
            valid_loss, valid_acc = predict_and_evaluate(net, valid_data_loader, high, width)
            logger.info("epoch: {epoch} valid_loss: {valid_loss} valid_acc: {valid_acc}%".
                        format(epoch=epoch, valid_loss=round(valid_loss, 4), valid_acc=round(valid_acc, 4)*100))


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
