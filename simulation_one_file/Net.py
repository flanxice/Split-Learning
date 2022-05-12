# -*- coding = utf-8 -*-
# @Time : 2021/11/11 18:39
# @Author : SBP
# @File : Net.py
# @Software : PyCharm

from torch import nn
from collections import OrderedDict
from seed import *
# from CONTROL import *


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


# input batch_size * 1 * 28 * 28
NetList = [
    ('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)),
    ('relu1', nn.ReLU()),
    ('pool1', nn.MaxPool2d(kernel_size=2)),
    # output batch_size * 16 * 14 *14
    ('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
    ('relu2', nn.ReLU()),
    ('pool2', nn.MaxPool2d(kernel_size=2)),
    # output batch_size * 32 * 7 * 7
    ('flatten', Flatten()),
    # output batch_size * (32 * 7 * 7)
    ('liner3', nn.Linear(32 * 7 * 7, 700)),
    ('relu3', nn.ReLU()),
    ('liner4', nn.Linear(700, 100)),
    ('relu4', nn.ReLU()),
    ('liner5', nn.Linear(100, 10))]

MyModel_Whole = nn.Sequential(OrderedDict(NetList))


def getMyModel(split_point=3, Net_List=NetList):  # split_point = 3, 6, 7, 9, 11
    NetList_Device = Net_List[0:split_point]
    NetList_Server = Net_List[split_point:]
    MyModel_Device = nn.Sequential(OrderedDict(NetList_Device))
    MyModel_Sever = nn.Sequential(OrderedDict(NetList_Server))
    return MyModel_Device, MyModel_Sever


# def modelinitsave(Net_List=NetList):
#     length = len(Net_List)
#     for i in range(1, length):
#         model_device, model_server = getMyModel(split_point=i)
#         torch.save(model_device.state_dict(), './net_init/net_device_init_sp' + str(i) + '.pkl')
#         torch.save(model_server.state_dict(), './net_init/net_server_init_sp' + str(i) + '.pkl')


if __name__ == '__main__':
    MyModel_Device, MyModel_Sever = getMyModel()
    inputs = torch.randn(16, 1, 28, 28)
    tempout = MyModel_Device(inputs)
    print(tempout.size())
    out = MyModel_Sever(tempout)
    print(out.size())
