#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 2019/2/8 11:39
# @Author  : ylin
# Description:
# 
from torch.utils.data.dataset import Dataset
from os.path import join
from os import listdir
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = dataset_dir + 'X4/'
        self.target_dir = dataset_dir + 'HR/'
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.target_filenames = [join(self.target_dir, x) for x in listdir(self.target_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert('RGB')
        target = Image.open(self.target_filenames[index]).convert('RGB')
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.image_filenames)
