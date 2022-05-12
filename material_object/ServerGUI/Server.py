# -*- coding = utf-8 -*-
# @Time : 2022/1/16 15:04
# @Author : SBP
# @File : Server.py
# @Software : PyCharm
import numpy as np
import torch

from Communication_Net import *
from Net import *
from copy import deepcopy
import threading


class MyServer:
    def __init__(self, num_device, LR, split_point, num_iteration_end, step, batch_size_train, batch_size_test,
                 IF_testall=0, DEVICE=torch.device('cpu')):
        self.DEVICE = DEVICE
        self.num_device = num_device
        host = '0.0.0.0'
        port_start = 50000
        self.LR = LR
        self.split_point = split_point
        self.num_iteration_end, self.step = num_iteration_end, step
        self.IF_testall = IF_testall  # False
        self.batch_size_train = [batch_size_train] * self.num_device
        self.batch_size_test = [batch_size_test] * self.num_device
        self.batch_size_train_str = [str(i) + 'q' for i in self.batch_size_train]
        self.batch_size_test_str = [str(i) + 'q' for i in self.batch_size_test]
        # net init
        net_device, net_sever = getMyModel(split_point=split_point)
        self.net_device, self.net_sever = net_device.to(DEVICE), net_sever.to(DEVICE)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net_sever.parameters(), lr=self.LR)

        # Server conn create
        self.ServerConns = []
        for i in range(num_device):
            conn = ServerConn()
            conn.get_conn_connect(port=port_start + i, host=host)
            self.ServerConns.append(conn)

        # send 1,split point; 2,batch_size_train; 3,batch_size_test
        for i in range(num_device):
            self.ServerConns[i].conn_send_code(self.batch_size_train_str[i])
            self.ServerConns[i].conn_send_code(self.batch_size_test_str[i])
            self.ServerConns[i].conn_send_code(str(split_point) + 'q')
            self.ServerConns[i].conn_send_code(str(num_iteration_end) + 'q')
            self.ServerConns[i].conn_send_code(str(step) + 'q')
            self.ServerConns[i].conn_send_code(str(IF_testall) + 'q')

        self.times_test = []
        self.num_test = []
        for i in range(self.num_device):
            time_test = int(self.ServerConns[i].conn_recv_uq())
            self.times_test.append(time_test)
            self.num_test.append(time_test * self.batch_size_test[i])

        self.iteration = 0

    def test(self):
        print('\033[0;33;40m\tTest\033[0m')
        net_temp = ServerNet(self.net_sever, self.criterion)
        # add update net here!!!!
        loss_all_device = 0
        num_correct_all = 0
        loss_test_every_device_once = []  # 每个用户的loss
        acc_test_every_device_once = []  # 每个用户的准确率
        for i in range(self.num_device):
            loss_all_one = 0
            net_temp.num_correct = 0
            for j in range(self.times_test[i]):
                data = self.ServerConns[i].conn_recv_pickle()
                temp = data['data'].to(self.DEVICE)
                label = data['label'].to(self.DEVICE)
                loss_once, loss_batch_once = net_temp.forward_predict(temp, label)
                loss_all_one += loss_once
                loss_all_device += loss_batch_once
            loss_all_one = loss_all_one / self.times_test[i]
            num_correct_all += net_temp.num_correct
            acc_one = net_temp.num_correct / (self.num_test[i])
            print('device ' + str(i) + ' : ' + 'loss = {}; acc = {}'.format(loss_all_one, acc_one))
            loss_test_every_device_once.append(loss_all_one.item())
            acc_test_every_device_once.append(acc_one.item())
        loss_all_device = loss_all_device / sum(self.num_test)
        acc_all = num_correct_all / sum(self.num_test)
        print('all device : loss = {}; acc = {}'.format(loss_all_device, acc_all))
        return torch.Tensor(loss_test_every_device_once), torch.tensor(
            acc_test_every_device_once), loss_all_device, acc_all

    def train(self):
        print('\033[0;31;40m\tTrain\033[0m')
        net_grads_all = []
        loss_train_every_device = []
        loss_a_iteration = 0
        Server_net_temp = ServerNet(self.net_sever, self.criterion)
        for i in range(self.num_device):
            data = self.ServerConns[i].conn_recv_pickle()
            temp = data['data'].to(self.DEVICE)
            label = data['label'].long().to(self.DEVICE)
            loss_one, loss_a_batch, _ = Server_net_temp.net_forward_backward(data=temp, label=label)
            loss_a_iteration += loss_a_batch
            net_grads = deepcopy(Server_net_temp.net_get_grad())
            net_grads_all.append(net_grads)
            print('device ' + str(i) + ' : ' + 'loss = {}'.format(loss_one))
            loss_train_every_device.append(loss_one.item())
        loss_a_iteration = loss_a_iteration / sum(self.batch_size_train)
        print('all device loss = {}'.format(loss_a_iteration.item()))
        # update net
        self.optimizer.zero_grad()
        self.net_sever.zero_grad()
        self.net_sever = net_average_grads(self.net_sever, self.batch_size_train, net_grads_all, self.num_device,
                                           self.DEVICE)
        self.optimizer.step()
        self.iteration += 1
        return torch.tensor(loss_train_every_device), loss_a_iteration.item()


if __name__ == '__main__':

    c = MyServer(num_device=2, LR=0.001, split_point=3, num_iteration_end=800, step=1,
                 batch_size_train=128, batch_size_test=128, IF_testall=0, DEVICE=torch.device('cpu'))
    while True:
        if c.iteration > c.num_iteration_end:
            print('over')
            break
        if c.iteration % c.step == c.step - 1:
            c.test()
        c.train()
