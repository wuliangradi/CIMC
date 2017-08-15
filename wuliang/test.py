#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Contact:wuliangwuwu@126.com
"""

import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img = mpimg.imread('/Users/baidu/wuliang/CIMC/wuliang/dataset/train/0cdf5b5d0ce1_01.jpg')
plt.figure()
plt.imshow(img)
plt.show()