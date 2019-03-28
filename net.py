#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 2019/2/8 11:45
# @Author  : ylin
# Description:
# 
import torch.nn as nn
import torch


# 这个网络很不ESPCN,但有一定价值,也许和下面的结合下可以获得更好效果.
class Net(nn.Module):
    def __init__(self, upscale_factor=4, kernel=3):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3 * 64, kernel_size=kernel, stride=1,
                               padding=int((kernel - 1) / 2))
        self.conv2 = nn.Conv2d(in_channels=3 * 64, out_channels=3 * 64, kernel_size=kernel, stride=1,
                               padding=int((kernel - 1) / 2))
        self.conv3 = nn.Conv2d(in_channels=3 * 64, out_channels=3 * 32, kernel_size=kernel, stride=1,
                               padding=int((kernel - 1) / 2))
        self.conv4 = nn.Conv2d(in_channels=3 * 32, out_channels=3 * (upscale_factor ** 2), kernel_size=kernel, stride=1,
                               padding=int((kernel - 1) / 2))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn1 = nn.BatchNorm2d(3 * 64)
        self.bn2 = nn.BatchNorm2d(3 * 64)
        self.bn3 = nn.BatchNorm2d(3 * 32)
        self.bn4 = nn.BatchNorm2d(3 * (upscale_factor ** 2))
        self.bn5 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = torch.tanh(self.bn3(self.conv3(x)))
        # x = sigmoid(self.pixel_shuffle(self.conv4(x)))  # 估计会有部分超出敏感范围
        # x = sigmoid(self.pixel_shuffle(self.bn4(self.conv4(x))))  # 放大前归一化似乎不太合理
        x = torch.sigmoid(self.bn5(self.pixel_shuffle(self.conv4(x))))
        return x


# https://blog.csdn.net/zuolunqiang/article/details/52401802
class ESPCN(nn.Module):
    def __init__(self, upscale_factor=4, kernel=3):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3 * 64, kernel_size=kernel, stride=1,
                               padding=int((kernel - 1) / 2))
        self.conv2 = nn.Conv2d(in_channels=3 * 64, out_channels=3 * 64, kernel_size=kernel, stride=1,
                               padding=int((kernel - 1) / 2))
        self.conv3 = nn.Conv2d(in_channels=3 * 64, out_channels=3 * 32, kernel_size=kernel, stride=1,
                               padding=int((kernel - 1) / 2))
        self.conv4 = nn.Conv2d(in_channels=3 * 32, out_channels=3 * (upscale_factor ** 2), kernel_size=kernel, stride=1,
                               padding=int((kernel - 1) / 2))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.tanh((self.conv1(x)))
        x = torch.tanh((self.conv2(x)))
        x = torch.tanh((self.conv3(x)))
        x = torch.sigmoid((self.pixel_shuffle(self.conv4(x))))
        return x
