#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 2019/2/27 21:12
# @Author  : ylin
# Description:
#

import os
import sys

# region select which net loading mode will be used.
# net_mode = 'new'
net_mode = 'from_params'
# net_mode = 'from_intact'
# endregion

# region work mode
work_modes = ['test']
# work_mode = 'train'
# work_mode = 'test'
# work_mode = 'deal'
# work_mode = 'intact_save'
# endregion


lr = 0.00001
train_time = 1

dir_root = sys.path[0]
# the root directory of the train data
dir_train = dir_root + '/etc/train/'
# the root directory of the test data
# dir_train = dir_root + '/etc/test/'
dir_test = dir_root + '/etc/test/'
# the output directory when the script run as 'deal' mode.
dir_output = dir_root + '/etc/output/'
# model config
dir_model = os.path.join(os.path.split(dir_root)[0], 'model/')
used_params_file = dir_model + 'net_params_5_1000.pkl'
save_params_file = dir_model + 'net_params_5_1000.pkl'
used_intact_file = dir_model + 'net.pkl'
save_intact_file = dir_model + 'net.pkl'
