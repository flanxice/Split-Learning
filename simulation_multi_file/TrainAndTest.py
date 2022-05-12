# -*- coding = utf-8 -*-
# @Time : 2021/9/24 11:15
# @Author : SBP
# @File : TrainAndTest.py
# @Software : PyCharm

import torch


def showLoss(loss, step, step_gap=50):
    if step % step_gap == 0:
        print('iteration = {}, loss_Device = {}'.format(step, loss))


def averageGrads(myNN, grads, train_batchsize, device_num, DEVICE):
    for index, param in enumerate(myNN.parameters()):
        param.grad = (grads[0][index] * train_batchsize[0]).to(DEVICE)
        for i in range(1, device_num):
            param.grad = param.grad + (grads[i][index] * train_batchsize[i]).to(DEVICE)
        param.grad = param.grad / sum(train_batchsize)
    return myNN


def getParamGrad(myNN):
    grads_list = []
    for index, (name, param) in enumerate(myNN.named_parameters()):
        grads_list.append(param.grad)
    return grads_list


def makesplitNet(myNN_device, myNN_server, grads_device, grads_server):
    for index, param in enumerate(myNN_device.parameters()):
        param.grad = grads_device[index]
    for index, param in enumerate(myNN_server.parameters()):
        param.grad = grads_server[index]
    return myNN_device, myNN_server


def saveParamGrad(myNN, path):
    grads_list = []
    for index, (name, param) in enumerate(myNN.named_parameters()):
        grads_list.append(param.grad)
    torch.save(grads_list, path)


def splitbatch(batch_size, datas, labels):
    i = 0
    length_data = len(datas)
    batch_num = length_data // batch_size
    datas_out, labels_out = [], []
    while True:
        if (i + 1) * batch_size > length_data:
            break
        data_temp = datas[i * batch_size:(i + 1) * batch_size]
        label_temp = labels[i * batch_size:(i + 1) * batch_size]
        datas_out.append(data_temp)
        labels_out.append(label_temp)
        i += 1
    return datas_out, labels_out, batch_num


