#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    The way of AI
    Contact:wuliangwuwu@126.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class CarUNet(nn.Module):
    def __init__(self, in_shape, num_classes):
        super(CarUNet, self).__init__()
        in_channels, height, width = in_shape

        # 512
        self.down1 = nn.Sequential(
            *(make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1))
        )
        # 256

        # 256
        self.down2 = nn.Sequential(
            *(make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1))
        )
        # 128

        self.down3 = nn.Sequential(
            *(make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1))
        )
        # 64

        self.down4 = nn.Sequential(
            *(make_conv_bn_relu(64, 128, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1))
        )
        # 32

        self.down5 = nn.Sequential(
            *(make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1))
        )
        # 16

        self.down6 = nn.Sequential(
            *(make_conv_bn_relu(256, 512, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1))
        )
        # 8

        self.center = nn.Sequential(
            *(make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(1024, 1024, kernel_size=3, stride=1, padding=1))
        )

        # 16
        self.up6 = nn.Sequential(
            *(make_conv_bn_relu(512 + 1024, 512, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1))
            # nn.Dropout(p=0.10),
        )
        # 16

        self.up5 = nn.Sequential(
            *(make_conv_bn_relu(256 + 512, 256, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1))
        )
        # 32

        self.up4 = nn.Sequential(
            *(make_conv_bn_relu(128 + 256, 128, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1))
        )
        # 64

        self.up3 = nn.Sequential(
            *(make_conv_bn_relu(64 + 128, 64, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1))
        )
        # 128

        self.up2 = nn.Sequential(
            *(make_conv_bn_relu(32 + 64, 32, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1))
        )
        # 128
        # -------------------------------------------------------------------------

        self.up1 = nn.Sequential(
            *(make_conv_bn_relu(16 + 32, 16, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1))
        )
        # 128

        self.up0 = nn.Sequential(
            *(make_conv_bn_relu(3 + 16, 8, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=1) +
              make_conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=1))
        )
        # 128

        self.classify = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 512
        down1 = self.down1(x)
        out = F.max_pool2d(down1, kernel_size=2, stride=2)  # 64

        down2 = self.down2(out)
        out = F.max_pool2d(down2, kernel_size=2, stride=2)  # 64

        down3 = self.down3(out)
        out = F.max_pool2d(down3, kernel_size=2, stride=2)  # 64

        down4 = self.down4(out)
        out = F.max_pool2d(down4, kernel_size=2, stride=2)  # 32

        down5 = self.down5(out)
        out = F.max_pool2d(down5, kernel_size=2, stride=2)  # 16

        down6 = self.down6(out)
        out = F.max_pool2d(down6, kernel_size=2, stride=2)  # 8

        out = self.center(out)

        out = F.upsample_bilinear(out, scale_factor=2)  # 16
        out = torch.cat([down6, out], 1)
        out = self.up6(out)

        out = F.upsample_bilinear(out, scale_factor=2)  # 32
        out = torch.cat([down5, out], 1)
        out = self.up5(out)

        out = F.upsample_bilinear(out, scale_factor=2)  # 64
        out = torch.cat([down4, out], 1)
        out = self.up4(out)

        out = F.upsample_bilinear(out, scale_factor=2)  # 128
        out = torch.cat([down3, out], 1)
        out = self.up3(out)

        out = F.upsample_bilinear(out, scale_factor=2)  # 128
        out = torch.cat([down2, out], 1)
        out = self.up2(out)

        out = F.upsample_bilinear(out, scale_factor=2)  # 256
        out = torch.cat([down1, out], 1)
        out = self.up1(out)

        out = F.upsample_bilinear(out, scale_factor=2)  # 1024
        x = F.upsample_bilinear(x, scale_factor=2)
        out = torch.cat([x, out], 1)
        out = self.up0(out)

        out = self.classify(out)

        return out


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


if __name__ == '__main__':

    batch_size = 10
    C, H, W = 3, 256, 256

    if 1:
        num_classes = 1

        inputs = torch.randn(batch_size, C, H, W)
        labels = torch.LongTensor(batch_size, 2 * H, 2 * W).random_(1).type(torch.FloatTensor)

        net = CarUNet(in_shape=(C, H, W), num_classes=1).cuda().train()
        x = Variable(inputs).cuda()
        y = Variable(labels).cuda()
        logits = net.forward(x)

        loss = BCELoss2d()(logits, y)
        loss.backward()

        print(type(net))
        print(net)

        print('logits')
        print(logits)
