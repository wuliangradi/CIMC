#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    The way of AI
    Contact:wuliangwuwu@126.com
"""

import numpy as np
import torch


def image_to_tensor(image, mean=0, std=1.):
    image = image.astype(np.float32)
    image = (image - mean)/std
    image = image.transpose((2,0,1))
    img_tensor = torch.from_numpy(image)
    return img_tensor


def label_to_tensor(label, threshold=0.5):
    print label.shape
    label = (label > threshold).astype(np.float32)
    lab_tensor = torch.from_numpy(label).type(torch.FloatTensor)
    return lab_tensor