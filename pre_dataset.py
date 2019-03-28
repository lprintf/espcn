#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 2019/2/8 21:37
# @Author  : ylin
# Description:
# for prepare better dataset.ps:the test dataset is defective.
import matplotlib.pyplot as plt

perfect_hr = r'C:\Users\54564\jupyter_repertory\Single Image Super Resolution Challenge\etc\train\HR/'
perfect_x4 = r'C:\Users\54564\jupyter_repertory\Single Image Super Resolution Challenge\etc\train\X4/'
defective = r'C:\Users\54564\jupyter_repertory\Single Image Super Resolution Challenge\etc\office_dataset\high/'
raw_root = r'D:\54564\Downloads\benchmark'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])


def gatherer():
    pass


if __name__ == '__main__':
    pass
