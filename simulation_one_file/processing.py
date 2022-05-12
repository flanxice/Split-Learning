# -*- coding = utf-8 -*-
# @Time : 2022/3/27 14:21
# @Author : SBP
# @File : processing.py
# @Software : PyCharm

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from Net import *


def DataSplit(device_num, data_split, aveblock, datas, labels):
    store_data, store_label = [], []
    data_split = [aveblock * item for item in data_split]
    sum_split = [0]
    mysum = 0
    for item in data_split:
        mysum += item
        sum_split.append(mysum)
    for i in range(device_num):
        data_temp = datas[sum_split[i]:sum_split[i + 1]]
        label_temp = labels[sum_split[i]:sum_split[i + 1]]
        store_data.append(data_temp)
        store_label.append(label_temp)
    return store_data, store_label


def processing(num_device, train_split, test_split, IFTestSplit=True):
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    train_dataset_data = train_loader.dataset.data
    train_dataset_data = (train_dataset_data - 127.5) / 255
    train_dataset_data = train_dataset_data.unsqueeze(dim=1).float()
    train_dataset_label = train_loader.dataset.targets
    length_train = len(train_dataset_label)
    aveblock = length_train // sum(train_split)
    store_train_data, store_train_label = DataSplit(num_device, train_split, aveblock, train_dataset_data,
                                                    train_dataset_label)
    # save train data & label
    torch.save({'train_data': store_train_data, 'train_label': store_train_label}, './train_process')
    test_dataset_data = test_loader.dataset.data
    test_dataset_data = (test_dataset_data - 127.5) / 255
    test_dataset_data = test_dataset_data.unsqueeze(dim=1).float()
    test_dataset_label = test_loader.dataset.targets
    if IFTestSplit:
        length_test = len(test_dataset_label)
        aveblock = length_test // sum(test_split)
        store_test_data, store_test_label = DataSplit(num_device, test_split, aveblock, test_dataset_data,
                                                      test_dataset_label)
    else:
        store_test_data, store_test_label = test_dataset_data, test_dataset_label

    # save test data & label
    torch.save({'test_data': store_test_data, 'test_label': store_test_label}, './test_process')
    print('----------------------- data process over ----------------------')


if __name__ == '__main__':
    processing(5, [1] * 5, [1] * 5)
