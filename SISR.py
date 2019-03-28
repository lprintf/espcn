#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 2019/2/8 11:22
# @Author  : ylin
# Description:
# train script,base on ESPCN

# reference:
# blog:   https://blog.csdn.net/aBlueMouse/article/details/78710553
# paper:  https://arxiv.org/abs/1609.05158
# code:   https://github.com/leftthomas/ESPCN

# dataset row:
# benchmark
# DIV2K
# etc
import time

import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms as T
import torch
from os import listdir
from dataloader import DatasetFromFolder
from loss import MyLoss
from net import ESPCN as Net
from config import used_params_file, used_intact_file, save_intact_file, lr, train_time, net_mode, \
    dir_train, dir_test, dir_output, save_params_file, work_modes
import sys


class SISR:
    def __init__(self, net_mode='from_params'):
        if net_mode == 'from_intact':
            self.net = torch.load(used_intact_file)
            self.net.cuda()
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        elif net_mode == 'new':
            self.net = Net()
            self.net.cuda()
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        elif net_mode == 'from_params':
            self.net = Net()
            self.net.cuda()
            checkpoint = torch.load(used_params_file)
            self.net.load_state_dict(checkpoint['net'])
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            checkpoint['optimizer']['param_groups'][0]['lr'] = lr
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(self.optimizer.state_dict())
        else:
            print('please select a useful mode to contract neural net')
            sys.exit()

    def train(self, train_time, visualization=False, dataset_dir=dir_train):
        self.net.train()
        dataset = DatasetFromFolder(dataset_dir, T.ToTensor(), T.ToTensor())
        train_loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, pin_memory=True)
        if visualization:
            plt.ion()  # for ide test
            plt.show()
        loss_func = MyLoss()
        for epoch in range(train_time):
            print(f"epoch {epoch + 1}")
            for step, (x, y) in enumerate(train_loader):
                # for x, y in train_loader:
                b_x, b_y = x.cuda(), y.cuda()
                output = self.net(b_x)
                loss = loss_func(output, b_y)
                if step % 9 == 2 and visualization:
                    # print(output[0][2:3, 2:3, 40:45])
                    # print(b_y[0][2:3, 2:3, 40:45])
                    self.n_img_show((output.cpu()).detach().numpy()[0])
                    time.sleep(0.5)  # for my pitiful laptop
                    print(loss)
                self.optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients
                # time.sleep(0.1)
            self.tmp_save()

    def deal(self, input_dir=dir_test + 'X4/', output_dir=dir_output):
        self.net.eval()
        for filename in listdir(input_dir):
            # img2 = plt.imread(input_dir + filename)
            # img2 = torch.from_numpy(np.transpose(img2, (2, 0, 1))).unsqueeze(0)
            if self.is_image_file(filename):
                img = T.ToTensor()(Image.open(input_dir + filename).convert('RGB')).cuda()
                out_img = self.net(img.unsqueeze(0))
                n_img = (out_img.cpu()).detach().numpy()[0]
                self.n_img_save(n_img, output_dir + filename)

    def test(self, dataset_dir=dir_test):
        self.net.eval()
        dataset = DatasetFromFolder(dataset_dir, T.ToTensor(), T.ToTensor())
        test_loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        plt.ion()  # for ide test
        plt.show()
        loss_func = MyLoss()
        for x, y in test_loader:
            #         print(b_x.size(),b_y)
            b_x, b_y = x.cuda(), y.cuda()
            output = self.net(b_x)
            self.n_img_show((output.cpu()).detach().numpy()[0])
            # print(output[0][2:3, 2:3, 40:45])
            # print(b_y[0][2:3, 2:3, 40:45])
            time.sleep(2)

            loss = loss_func(output, b_y)
            print(loss)

    # region function
    @staticmethod
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])

    @staticmethod
    def n_img_show(img):
        plt.cla()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.axis('off')
        plt.show()
        plt.pause(0.1)

    @staticmethod
    def n_img_save(img, filename):
        plt.imsave(filename, np.transpose(img, (1, 2, 0)), format="png")

    def submit_save(self, ):
        torch.save(self.net, save_intact_file)

    def tmp_save(self, ):
        state = {'net': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, save_params_file)

    # endregion


if __name__ == '__main__':
    # params_file = 'net_params_4.pkl'
    # lr = 0.000001
    # if len(sys.argv) == 4:
    #     net_mode, work_mode, train_time = sys.argv[1:]
    #     train_time = int(train_time)
    # else:
    #     print('please check parameter')
    #     sys.exit()
    s = SISR(net_mode)
    for work_mode in work_modes:
        if work_mode == 'train':
            s.train(train_time, True)
        elif work_mode == 'deal':
            s.deal()
        elif work_mode == 'test':
            s.test()
        elif work_mode == 'intact_save':
            s.submit_save()
        else:
            print('please select todo_mode in train or deal or test')
