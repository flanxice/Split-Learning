# -*- coding = utf-8 -*-
# @Time : 2022/1/16 15:04
# @Author : SBP
# @File : Client.py
# @Software : PyCharm

import torch
from Communication_Net import *
from Net import *

index_device = 1
DEVIVE = torch.device('cpu')
PORT = 50000 + index_device
HOST = '10.181.222.235'

conn = ClientNet(0)
conn.get_conn_connect(port=PORT, host=HOST)
batch_size_train = int(conn.conn_recv_uq())
batch_size_test = int(conn.conn_recv_uq())
split_point = int(conn.conn_recv_uq())
num_iterations = int(conn.conn_recv_uq())
step = int(conn.conn_recv_uq())
IF_testall = bool(int(conn.conn_recv_uq()))

net_device, _ = getMyModel(split_point=split_point)
print('device index = {}'.format(index_device))
print('split point = {}; batch_size_train = {}; batch_size_test = {}'.format(split_point, batch_size_train,
                                                                             batch_size_test))
conn.net = net_device

# load data and label
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

conn.conn_send_code(str(times_test) + 'q')

iteration = 0
while True:
    if iteration > num_iterations:
        print('over')
        break
    if iteration % step == step - 1:
        print('test')
        for i in range(times_test):
            data_test_in = data_test[i].to(DEVIVE)
            label_in = label_test[i].to(DEVIVE)
            temp_out = conn.net_forward(data_test_in)
            send_buffer = {'data': temp_out, 'label': label_in}
            conn.conn_send_pickle(send_buffer)
    print('train')
    r = iteration % times_train
    data_train_in = data_train[r].to(DEVIVE)
    label_train_in = label_train[r].to(DEVIVE)
    temp_out = conn.net_forward(data_train_in)
    send_buffer = {'data': temp_out, 'label': label_train_in}
    conn.conn_send_pickle(send_buffer)
    iteration += 1
