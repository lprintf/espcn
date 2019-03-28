#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 2019/2/8 11:49
# @Author  : ylin
# Description:
#
import torch


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, img1, img2):
        return torch.mean((img1 - img2) ** 2)
