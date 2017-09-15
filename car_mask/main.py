#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

import log
from config import BATH_SIZE
from config import EPOCH_VALID
from config import IMG_PATH, IMG_LABEL_PATH, IMG_MASK_PATH
from config import IMG_TEST_PATH
from config import NUM_EPOCHES
from net import CarUNet
from net import criterion, dice_loss
from scripts import CarDataSet
from scripts import get_train_valid
from scripts import run_length_encode

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')


best_prec1 = 0


def validate(epoch, model, test_loader):
    test_acc = 0
    test_loss = 0
    test_num = 0
    for it, (img_tensor, label, img_mask_tensor) in enumerate(test_loader):
        image_ = Variable(img_tensor.cuda(), volatile=True)
        label_ = Variable(img_mask_tensor.cuda(), volatile=True)

        logits = model(image_)
        probs = F.sigmoid(logits)
        masks = (probs > 0.5).float()

        loss = criterion(logits, label_)
        acc = dice_loss(masks, label_)
        batch_size = len(img_mask_tensor)
        test_num += batch_size
        test_loss += batch_size * loss.data[0]
        test_acc += batch_size * acc.data[0]

    valid_loss = test_loss / test_num
    valid_acc = test_acc / test_num
    if epoch % EPOCH_VALID == 0 or epoch == 0 or epoch == NUM_EPOCHES - 1:
        logger.info("Validate=>\n"
                    "Epoch: {epoch}\t"
                    "Valid_loss: {valid_loss:.3f}\t"
                    "Valid_acc: {valid_acc:.3f}%"
                    .format(epoch=epoch, valid_loss=round(valid_loss, 4),
                            valid_acc=round(valid_acc, 4) * 100))

    return valid_loss, valid_acc


def train(epoch, net, optimizer, train_data_loader):
    """
    :param epoch:
    :param net:
    :param optimizer:
    :param train_data_loader:
    :return:
    """
    smooth_loss = 0.0
    smooth_acc = 0.0
    sum_smooth_loss = 0
    sum_smooth_acc = 0
    sum_iter = 0
    it_smooth = 30

    for it, (img_tensor, label, img_mask_tensor) in enumerate(train_data_loader):
        image_ = Variable(img_tensor.cuda())
        label_ = Variable(img_mask_tensor.cuda())

        # compute output
        logits = net(image_)
        probs = F.sigmoid(logits)
        masks = (probs > 0.5).float()
        loss = criterion(logits, label_)

        # compute gradient and do SGD step
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

            logger.info("Train=>\n"
                        "Epoch: {epoch}\t"
                        "Batch_num: {iter_num}\t"
                        "lr: {lr:.3f}\t"
                        "loss: {smooth_loss:.3f}\t"
                        "acc: {smooth_acc:.3f}%\t"
                        "train_loss: {train_loss}\t"
                        "train_acc: {train_acc}%".
                        format(epoch=epoch, iter_num=it, lr=0, smooth_loss=round(smooth_loss, 4),
                               smooth_acc=round(smooth_acc, 4) * 100, train_loss=round(train_loss, 4),
                               train_acc=round(train_acc, 4) * 100))


def predict():
    global args, best_prec1, logger
    args = parser.parse_args()
    data_set = CarDataSet([IMG_TEST_PATH],
                          is_predict=True)
    logger = log.init_log(logfile="./test_log.txt", log_name="test data")
    logger.info("Loading dataset...")

    test_loader = DataLoader(data_set,
                             shuffle=False,
                             drop_last=True,
                             # sampler=[1],
                             num_workers=args.workers)
    num_channel = 3
    width, high = (data_set.high, data_set.width)
    model = CarUNet(in_shape=(num_channel, width, high), num_classes=1)
    model.cuda()
    checkpoint = torch.load("model_best.pth.tar")
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    import csv
    csv_file = open("submission.csv", "wb")
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["img", "rle_mask"])
    for it, (img_tensor, name) in enumerate(test_loader):
        if it % 1000 == 0:
            print it, it / 100000.0 * 100, "%"
        # if name[0] != "0d1a9caf4350_05.jpg":
        #     continue
        # print name
        # img_out = torch.torch.from_numpy(np.zeros(shape=(1280, 1918)))
        image_ = Variable(img_tensor.cuda(), volatile=True)
        logits = model(image_)
        probs = F.sigmoid(logits)
        masks = (probs > 0.5).float()
        img_out = masks.cpu()
        img_out = img_out.data.numpy()
        img_out = img_out[0][0]
        img_out = Image.fromarray(img_out)
        img_out = img_out.resize((1918, 1280), 0)
        img_out_ = np.asarray(img_out)

        # run_length_encode(img_out_)
        # plt.figure()
        # plt.imshow(img_out_, interpolation='None')
        # plt.show()
        # from scripts import gif_loader
        # mask = gif_loader("/home/wuliang/wuliang/CIMC/car_mask/dataset/train_masks/0cdf5b5d0ce1_01_mask.gif")
        rl = run_length_encode(img_out_)
        writer.writerow([name[0], rl])


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main():
    global args, best_prec1, logger
    args = parser.parse_args()

    data_set = CarDataSet([IMG_PATH,
                           IMG_LABEL_PATH,
                           IMG_MASK_PATH],
                          # transform=[lambda x, y: random_resize(x, y, 1280, 1918, 512, 512)]
                          )

    logger = log.init_log(logfile="./train_log.txt", log_name="train data")
    valid_list, train_list = get_train_valid(logger=logger, proportion_valid=0.2, num_all=len(data_set))
    logger.info("Loading dataset...")
    train_loader = DataLoader(data_set,
                              batch_size=args.batch_size,
                              sampler=train_list[:],
                              shuffle=False,
                              drop_last=True,
                              num_workers=args.workers)

    valid_loader = DataLoader(data_set,
                              sampler=valid_list[:],
                              shuffle=False,
                              drop_last=True,
                              num_workers=args.workers)

    logger.info("All data sample counts {}".format(len(data_set)))
    logger.info("Train data batch size {}".format(BATH_SIZE))
    logger.info("Train data sample counts {}".format(len(train_loader) * BATH_SIZE))
    logger.info("Valid data sample counts {}".format(len(valid_loader)))

    num_channel = 3
    width, high = (data_set.high, data_set.width)
    model = CarUNet(in_shape=(num_channel, width, high), num_classes=1)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        model.train()
        train(epoch, model, optimizer, train_loader)
        # evaluate on validation set
        model.eval()
        loss_, prec_ = validate(epoch, model, valid_loader)
        is_best = prec_ > best_prec1
        best_prec1 = max(prec_, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "u_net",
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print "please input your arguments like 'train' or 'predict'"
    #     exit(-1)
    # print "Main function"
    # if sys.argv[1] == "train":
    #     train()
    # if sys.argv[1] == "predict":
    #     predict()
    # print "\nprogram run succeed"
    # main()
    # predict()
    # model_names = sorted(name for name in models.__dict__
    #                      if name.islower() and not name.startswith("__")
    #                      and callable(models.__dict__[name]))
    # print model_names
    # model = torch.load('checkpoint.pth.tar')
    # print model
    with open("/home/wuliang/wuliang/CIMC/car_mask/output/submission.csv") as fp:
        for line in fp.readlines():
            li = line.strip().split()
            if len(li) == 2:
                print li
