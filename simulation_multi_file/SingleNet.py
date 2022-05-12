# -*- coding = utf-8 -*-
# @Time : 2021/11/19 9:20
# @Author : SBP
# @File : SingleNet.py
# @Software : PyCharm

from Net import *
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
import os
from CONTROL import *

lr = 0.0001
num_epoch = 60
train_batch_size = 64
test_batch_size = 128
print(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
print(DEVICE)

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
myNN = MyModel_Whole.to(DEVICE)
optimizer = optim.Adam(myNN.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for i in range(num_epoch):
    for data in train_loader:
        img, label = data
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        # print(img.shape)
        out = myNN(img)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    num_correct = 0
    loss_test = 0
    for data in test_loader:
        img, label = data
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        out = myNN(img)
        loss = criterion(out, label)
        loss_test += loss
        _, pred = torch.max(out, 1)
        num_correct += (pred == label).sum()
    if (num_correct / len(test_loader) / test_batch_size) >= 0.7:
        torch.save(myNN.state_dict(), './myNN_Single_Net.pkl')
        print('save')
    print('epoch = {}'.format(i))
    print('test loss = {}'.format(loss_test / len(test_loader)))
    print('test acc = {}'.format(num_correct / len(test_loader) / test_batch_size))
