#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Contact:wuliangwuwu@126.com
"""

from torch.utils.data.dataset import Dataset
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd
import os

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_image_list(path):
    image = [f for f in listdir(path) if isfile(join(path, f))]
    return image


def get_image_label(images, csv_path):
    labels = []
    df = pd.read_csv(csv_path)
    for item in images:
        try:
            labels.append(df[df['img'] == item].rle_mask.item())
        except:
            labels.append("")

    return labels


class CarDataSet(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root[0]
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = get_image_list(root[0])
        self.labels = get_image_label(self.imgs, root[1])

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        img_array = pil_loader(os.path.join(self.root, img))
        return img_array, label, img

    def __len__(self):
        pass


te = CarDataSet(["/Users/baidu/wuliang/CIMC/wuliang/dataset/train",
                 "/Users/baidu/wuliang/CIMC/wuliang/dataset/train_masks.csv"])
# print os.path.join("/Users/baidu/wuliang/CIMC/wuliang/dataset/train", "666")
print te[10]


