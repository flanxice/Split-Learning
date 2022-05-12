# -*- coding = utf-8 -*-
# @Time : 2022/1/16 15:04
# @Author : SBP
# @File : Server.py
# @Software : PyCharm

from Communication_Net import *
from Net import *
from copy import deepcopy



DEVICE = torch.device('cpu')
num_device = 2
host = '0.0.0.0'
port_start = 50000
LR = 0.001
split_point = 3
num_iteration_end, step = 800, 1
IF_testall = 0  # False
batch_size_train = [128] * num_device
batch_size_test = [128] * num_device
batch_size_train_str = [str(i) + 'q' for i in batch_size_train]
batch_size_test_str = [str(i) + 'q' for i in batch_size_test]
# net init
net_device, net_sever = getMyModel(split_point=split_point)
net_device, net_sever = net_device.to(DEVICE), net_sever.to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_sever.parameters(), lr=LR)

        # Server conn create
ServerConns = []
for i in range(num_device):
    conn = ServerConn()
    conn.get_conn_connect(port=port_start + i, host=host)
    ServerConns.append(conn)

# send 1,split point; 2,batch_size_train; 3,batch_size_test
for i in range(num_device):
    ServerConns[i].conn_send_code(batch_size_train_str[i])
    ServerConns[i].conn_send_code(batch_size_test_str[i])
    ServerConns[i].conn_send_code(str(split_point) + 'q')
    ServerConns[i].conn_send_code(str(num_iteration_end) + 'q')
    ServerConns[i].conn_send_code(str(step) + 'q')
    ServerConns[i].conn_send_code(str(IF_testall) + 'q')

times_test = []
num_test = []
for i in range(num_device):
    time_test = int(ServerConns[i].conn_recv_uq())
    times_test.append(time_test)
    num_test.append(time_test * batch_size_test[i])

iteration = 0
while True:
    if iteration > num_iteration_end:
        print('over')
        break
    if iteration % step == step - 1:
        print('\033[0;33;40m\tTest\033[0m')
        net_temp = ServerNet(net_sever, criterion)
        # add update net here!!!!
        loss_all_device = 0
        num_correct_all = 0
        for i in range(num_device):
            loss_all_one = 0
            net_temp.num_correct = 0
            for j in range(times_test[i]):
                data = ServerConns[i].conn_recv_pickle()
                temp = data['data'].to(DEVICE)
                label = data['label'].to(DEVICE)
                loss_once, loss_batch_once = net_temp.forward_predict(temp, label)
                loss_all_one += loss_once
                loss_all_device += loss_batch_once
            loss_all_one = loss_all_one / times_test[i]
            num_correct_all += net_temp.num_correct
            acc_one = net_temp.num_correct / (num_test[i])
            print('device ' + str(i) + ' : ' + 'loss = {}; acc = {}'.format(loss_all_one, acc_one))
        loss_all_device = loss_all_device / sum(num_test)
        acc_all = num_correct_all / sum(num_test)
        print('all device : loss = {}; acc = {}'.format(loss_all_device, acc_all))
    print('\033[0;31;40m\tTrain\033[0m')
    net_grads_all = []
    loss_a_iteration = 0
    Server_net_temp = ServerNet(net_sever, criterion)
    for i in range(num_device):
        data = ServerConns[i].conn_recv_pickle()
        temp = data['data'].to(DEVICE)
        label = data['label'].long().to(DEVICE)
        loss_one, loss_a_batch, _ = Server_net_temp.net_forward_backward(data=temp, label=label)
        loss_a_iteration += loss_a_batch
        net_grads = deepcopy(Server_net_temp.net_get_grad())
        net_grads_all.append(net_grads)
        print('device ' + str(i) + ' : ' + 'loss = {}'.format(loss_one))
    # update net
    optimizer.zero_grad()
    net_sever.zero_grad()
    net_sever = net_average_grads(net_sever, batch_size_train, net_grads_all, num_device, DEVICE)
    optimizer.step()
    iteration += 1
