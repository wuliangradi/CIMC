#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Contact:wuliangwuwu@126.com
"""

import os
import random
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils import image_to_tensor
from utils import label_to_tensor


def jpg_loader(path):
    """
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def gif_loader(path):
    """
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert("L")


def get_image_list(path):
    """
    :param path:
    :return:
    """
    image = [f for f in listdir(path) if isfile(join(path, f))]
    return image


def get_image_mask_name(path, img_name):
    """
    :param path:
    :param img_name:
    :return:
    """
    for f in listdir(path):
        if f[:-9] == img_name[:-4]:
            return f
    return ""


def get_image_label(images, csv_path):
    """
    :param images:
    :param csv_path:
    :return:
    """
    labels = []
    df = pd.read_csv(csv_path)
    for item in images:
        try:
            labels.append(df[df['img'] == item].rle_mask.item())
        except:
            labels.append("")
    return labels


def show_image():
    te = CarDataSet(["/Users/baidu/wuliang/CIMC/wuliang/dataset/train",
                     "/Users/baidu/wuliang/CIMC/wuliang/dataset/train_masks.csv",
                     "/Users/baidu/wuliang/CIMC/wuliang/dataset/train_masks"])
    pil_img = te[10][0]
    name_img = te[10][2]
    pil_img.show()
    pil_gif = jpg_loader(
        "/Users/baidu/wuliang/CIMC/wuliang/dataset/train_masks/" + name_img.split(".")[0] + "_mask.gif")
    pil_gif.show()


def random_resize(image_, label_, high, width, target_high, target_width):
    start_high = random.randint(0, high - target_high)
    start_width = random.randint(0, width - target_width)
    return image_[start_high:(start_high + target_high), start_width:(start_width + target_width), :], \
           label_[start_high:(start_high + target_high), start_width:(start_width + target_width)]


def clip_(image_, start_high, start_width, target_high, target_width):
    return image_[start_high:(start_high + target_high), start_width:(start_width + target_width), :]


class CarDataSet(Dataset):
    def __init__(self, root, transform=None, target_transform=None, is_predict=False):
        self.root_img = root[0]
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = get_image_list(root[0])
        self.is_predict = is_predict
        if not is_predict:
            self.labels = get_image_label(self.imgs, root[1])
            self.root_img_mask = root[2]
        self.width = np.asarray(jpg_loader(os.path.join(self.root_img, self.imgs[0]))).shape[1]
        self.high = np.asarray(jpg_loader(os.path.join(self.root_img, self.imgs[0]))).shape[0]

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = jpg_loader(os.path.join(self.root_img, img_name))
        img = img.resize((1024, 512), Image.ANTIALIAS)
        img_arr = np.asarray(img) / 255.0
        if not self.is_predict:
            label = self.labels[index]
            img_mask_name = get_image_mask_name(self.root_img_mask, img_name)
            img_mask = gif_loader(os.path.join(self.root_img_mask, img_mask_name))
            img_mask = img_mask.resize((1920, 1280), Image.ANTIALIAS)

            img_mask_arr = np.asarray(img_mask)

            if self.transform:
                for tran in self.transform:
                    img_arr, img_mask_arr = tran(img_arr, img_mask_arr)
            img_mask_tensor = label_to_tensor(img_mask_arr)
            img_tensor = image_to_tensor(img_arr)
            return img_tensor, label, img_mask_tensor

        else:
            if self.transform:
                for tran in self.transform:
                    img_arr = tran(img_arr)
            img_tensor = image_to_tensor(img_arr)
            return img_tensor,img_name

    def __len__(self):
        return len(self.imgs)
