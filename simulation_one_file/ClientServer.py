# -*- coding = utf-8 -*-
# @Time : 2022/3/27 14:33
# @Author : SBP
# @File : ClientServer.py
# @Software : PyCharm

from Net import *


def split_data_label(batch_size, datas, labels):
    i = 0
    length = len(labels)
    batch_num = length // batch_size
    datas_out, labels_out = [], []
    while True:
        if (i + 1) * batch_size > length:
            break
        data_temp = datas[i * batch_size:(i + 1) * batch_size]
        label_temp = labels[i * batch_size:(i + 1) * batch_size]
        datas_out.append(data_temp)
        labels_out.append(label_temp)
        i += 1
    return datas_out, labels_out, batch_num


def getParamGrad(myNN):
    grads_list = []
    for index, (name, param) in enumerate(myNN.named_parameters()):
        grads_list.append(param.grad)
    return grads_list


def net_average_grads(net, batch_size_train, grads, device_num, DEVICE='cpu'):
    for index, param in enumerate(net.parameters()):
        param.grad = (grads[0][index] * batch_size_train[0]).to(DEVICE)
        for i in range(1, device_num):
            param.grad = param.grad + (grads[i][index] * batch_size_train[i]).to(DEVICE)
        param.grad = param.grad / sum(batch_size_train)
    return net


class Client:
    def __init__(self, split_point, index_device, IF_testall, batch_size_train, batch_size_test):
        data_label_train = torch.load('./train_process')
        data_label_test = torch.load('./test_process')
        data_train = data_label_train['train_data'][index_device]
        label_train = data_label_train['train_label'][index_device]
        if IF_testall:
            data_test = data_label_test['test_data']
            label_test = data_label_test['test_label']
        else:
            data_test = data_label_test['test_data'][index_device]
            label_test = data_label_test['test_label'][index_device]
        data_train, label_train, times_train = split_data_label(batch_size_train, data_train, label_train)
        data_test, label_test, times_test = split_data_label(batch_size_test, data_test, label_test)
        self.data_train = data_train
        self.label_train = label_train
        self.times_train = times_train
        self.data_test = data_test
        self.label_test = label_test
        self.times_test = times_test
        net, _ = getMyModel(split_point=split_point)
        self.net = net.to(DEVICE)

    def net_forward_train(self, iteration):
        r = iteration % self.times_train
        data_train_in = self.data_train[r].to(DEVICE)
        label_train_in = self.label_train[r].to(DEVICE)
        tempout = self.net(data_train_in)
        return tempout, label_train_in

    def net_forward_test(self, r):
        data_test_in = self.data_test[r].to(DEVICE)
        label_test = self.label_test[r].to(DEVICE)
        tempout = self.net(data_test_in)
        return tempout, label_test

    def net_backward(self, tempout, temp_grad):
        self.net.zero_grad()
        tempout.backward(temp_grad)


    def net_get_grad(self):
        net_grads = []
        for param in self.net.parameters():
            net_grads.append(param.grad)
        return net_grads

    def update_net(self, net_grad):
        self.net.zero_grad()
        for index, param in enumerate(self.net.parameters()):
            param.grad = net_grad[index]


class Server:
    def __init__(self, net):
        self.net = net.to(DEVICE)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_correct = 0

    def net_forward_backward(self, tempout, label):
        tempout = tempout.to(DEVICE)
        label = label.long().to(DEVICE)
        self.net.zero_grad()
        out = self.net(tempout)
        loss = self.criterion(out, label)
        loss_a_batch = loss * label.size(0)
        loss.backward(retain_graph=False)
        return loss, loss_a_batch, tempout.grad

    def forward_predict(self, temp, label):
        temp = temp.to(DEVICE)
        label = label.long().to(DEVICE)
        self.net.zero_grad()
        out = self.net(temp)
        _, pred = torch.max(out, 1)
        self.num_correct += (pred == label).sum()
        loss = self.criterion(out, label.long())
        loss_test_batch = loss * label.size(0)
        return loss, loss_test_batch

    def net_get_grad(self):
        net_grads = []
        for param in self.net.parameters():
            net_grads.append(param.grad)
        return net_grads

    def net_update(self, net_grad):
        self.net.zero_grad()
        for index, param in enumerate(self.net.parameters()):
            param.grad = net_grad[index]


if __name__ == '__main__':
    pass
