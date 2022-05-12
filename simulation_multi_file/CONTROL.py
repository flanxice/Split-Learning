# -*- coding = utf-8 -*-
# @Time : 2021/11/19 23:06
# @Author : SBP
# @File : CONTROL.py
# @Software : PyCharm

from pickle import TRUE
import torch
import numpy as np
import random
import os

ADDRESS = 13
IFConvergence = True  # if False need same plit_point
IFTestSplit = True    # 测试集是否分割

seed = 1234
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def getGPUID(index, SAME=False):
    if SAME:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        if index <= 5:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        elif index <= 10:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def DeviceGPU(index, SAMEGPU=False, CPU=True):
    if SAMEGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif CPU:
        DEVICE = "cpu"
    else:
        if index <= 5:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif index <= 10:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:  
            DEVICE = "cpu"
    print(DEVICE)
    return DEVICE
