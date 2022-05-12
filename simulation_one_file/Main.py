# -*- coding = utf-8 -*-
# @Time : 2022/3/27 14:17
# @Author : SBP
# @File : accumulate.py
# @Software : PyCharm

from processing import *
from ClientServer import *
from copy import deepcopy
import time

learning_rate = 0.001
num_device = 10  # 设备个数
num_iterations = 800  # 迭代次数 大概1epoch = 60000/1/64
step = 1  # 每隔1测试一次
split_points = [12] * 10
train_data_num = [1] * 10  # 训练数据量 1:1 # 一共60000
test_data_num = [1] * 10  # 测试集数据量比值 # 一共10000
train_batch_size = [500] * 10
test_batch_size = [200] * 10
IFTestSplit = True
IFAccumulate = True

net_client_main, net_sever_mian = getMyModel(split_point=split_points[0])
optimizer_server = torch.optim.Adam(net_sever_mian.parameters(), lr=learning_rate)
optimizer_client = torch.optim.Adam(net_client_main.parameters(), lr=learning_rate)

processing(num_device, train_data_num, test_data_num, IFTestSplit=IFTestSplit)

clients = []
for i in range(num_device):
    client_temp = Client(split_points[i], i, not IFTestSplit, train_batch_size[i], test_batch_size[i])
    clients.append(client_temp)

# need save
loss_test, acc_test = [], []
loss_train = []
time1, time2, time3, time4 = [], [], [], []

iteration = 0
while True:
    print('iteration = {}\n'.format(iteration))
    if iteration > num_iterations:
        print('over')
        break
    if iteration % step == step - 1:
        print('\ntest')
        # update net here!
        loss_all = 0
        num_correct_all = 0
        test_length_all = 0
        for i in range(num_device):
            num_correct = 0
            loss_one_device = 0
            server_temp = Server(net_sever_mian)
            for j in range(clients[i].times_test):
                temp_out, label = clients[i].net_forward_test(j)
                _, loss_test_batch = server_temp.forward_predict(temp_out, label)
                loss_one_device += loss_test_batch
                loss_all += loss_test_batch
            num_correct = server_temp.num_correct
            num_correct_all += num_correct
            test_length = test_batch_size[i] * clients[i].times_test
            test_length_all += test_length
            loss_one_device = loss_one_device / test_length
            acc_one = num_correct / test_length
            print('DEVICE ' + str(i) + ' : \n loss = {}, acc = {}'.format(loss_one_device, acc_one))
        loss_all = loss_all / test_length_all
        acc_all = num_correct_all / test_length_all
        print('loss_all = {}, acc_all = {}'.format(loss_all, acc_all))
        loss_test.append([iteration, loss_all.item()])
        acc_test.append([iteration, acc_all.item()])
    print('\ntrain')
    # 正向传播与反向传播 => 网络梯度
    loss_all = 0
    net_grads_all_server = []
    net_grads_all_client = []
    for i in range(num_device):
        server_temp = Server(net_sever_mian)
        starttime = time.time()
        tempout, label = clients[i].net_forward_train(iteration)
        endtime = time.time()
        time1.append([iteration, i, endtime - starttime])
        temp_out = deepcopy(tempout.data)
        temp_out.requires_grad = True
        starttime = time.time()
        _, loss_a_batch, temp_grad = server_temp.net_forward_backward(temp_out, label)
        endtime = time.time()
        time2.append([iteration, i, endtime - starttime])
        loss_all += loss_a_batch
        net_grads_server = deepcopy(server_temp.net_get_grad())
        net_grads_all_server.append(net_grads_server)
        if IFAccumulate:
            starttime = time.time()
            clients[i].net_backward(tempout, deepcopy(temp_grad))
            endtime = time.time()
            time3.append([iteration, i, endtime - starttime])
            net_grads_client = deepcopy(clients[i].net_get_grad())
            net_grads_all_client.append(net_grads_client)
        else:
            pass
    loss_all = loss_all / sum(train_batch_size)
    print('loss = {}'.format(loss_all))
    loss_train.append([iteration, loss_all.item()])
    # 平均网络梯度
    starttime = time.time()
    optimizer_server.zero_grad()
    net_sever_mian.zero_grad()
    net_sever_main = net_average_grads(net_sever_mian, train_batch_size, net_grads_all_server, num_device, DEVICE)
    optimizer_server.step()
    if IFAccumulate:
        optimizer_client.zero_grad()
        net_client_main.zero_grad()
        net_client_main = net_average_grads(net_client_main, train_batch_size, net_grads_all_client, num_device, DEVICE)
        optimizer_client.step()
    endtime = time.time()
    time4.append([iteration, endtime - starttime])
    iteration += 1

torch.save({'loss_test': loss_test, 'acc_test': acc_test, 'loss_train': loss_train, 'time1': time1, 'time2': time2,
            'time3': time3, 'time4': time4},
           './result_' + str(split_points[0]) + '_' + str(num_device) + '_' + str(train_batch_size[0]) + '_' + str(
               test_batch_size[0]) + '_' + str(IFTestSplit) + '_' + str(IFAccumulate))
