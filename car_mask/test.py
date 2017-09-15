#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Contact:wuliangwuwu@126.com
"""

# import PIL
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# img = mpimg.imread('/Users/baidu/wuliang/CIMC/wuliang/dataset/train/0cdf5b5d0ce1_01.jpg')
# plt.figure()
# plt.imshow(img)
# plt.show()

count = 1
with open("/home/wuliang/wuliang/CIMC/car_mask/output/submission.csv") as fp:
    for line in fp.readlines():
        li = line.strip().split(",")
        count = count + 1
        if len(li[1].split(" "))%2 !=0:
            print li
print count