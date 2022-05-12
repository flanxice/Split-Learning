# -*- coding = utf-8 -*-
# @Time : 2021/10/12 15:34
# @Author : SBP
# @File : Device1.py
# @Software : PyCharm


from multiprocessing.connection import Client

import torch

from Net import *
from TrainAndTest import *
from CONTROL import *
import time

index_device = 9
DEVICE = DeviceGPU(index_device)

address2 = ('localhost', 7400 + index_device + ADDRESS)
conn2 = Client(address2, authkey=b"password")  # server_center

address3 = ('localhost', 6900 + index_device + ADDRESS)
conn3 = Client(address3, authkey=b"password")  # server_separate

split_point = int(conn2.recv())
train_batch_size = int(conn2.recv())
test_batch_size = int(conn2.recv())
num_iterations = int(conn2.recv())
step = int(conn2.recv())
print('split_point = {}, train_batch_size = {},\ntest_batch_size = {}, num_iterations = {}'.format(split_point,
                                                                                                   train_batch_size,
                                                                                                   test_batch_size,
                                                                                                   num_iterations))
print('step = {}'.format(step))
# load train and test data and label
data_label = torch.load('./train_process')
train_data = data_label['train_data'][index_device - 1]
train_label = data_label['train_label'][index_device - 1]
data_label = torch.load('./test_process')
if IFTestSplit:
    test_data = data_label['test_data'][index_device - 1]
    test_label = data_label['test_label'][index_device - 1]
else:
    test_data = data_label['test_data']
    test_label = data_label['test_label']

# create train loader and test loader
train_data, train_label, train_batch_num = splitbatch(train_batch_size, train_data, train_label)
test_data, test_label, test_batch_num = splitbatch(test_batch_size, test_data, test_label)
train_num_act, test_num_act = train_batch_num * train_batch_size, test_batch_num * test_batch_size
print(train_num_act, test_num_act)
torch.save(train_num_act, './data_num_device/train_num' + str(index_device))
torch.save(test_num_act, './data_num_device/test_num' + str(index_device))

# define net and loss function
MyNN_Device, _ = getMyModel(split_point=split_point, Net_List=NetList)
MyNN_Device = MyNN_Device.to(DEVICE)
# MyNN_Device.load_state_dict(torch.load('./net_init/net_device_init_sp' + str(split_point) + '.kpl'))
criterion = nn.CrossEntropyLoss()
torch.save(MyNN_Device.state_dict(), './Net_device/net_device' + str(index_device) + '.pkl')

print('-------------------preprocess done----------------')
conn3.send('preprocess done')
conn2.send('preprocessdone')

iteration = 0
time_forward_list, time_backward_list = [], []
while True:
    if iteration > num_iterations:  # over
        conn3.send('-1')
        torch.save({'time_forward_list': time_forward_list, 'time_backward_list': time_backward_list},
                   './result/time_device' + str(index_device))
        break
    if iteration % step == step - 1:  #
        conn3.send('test')
        MyNN_Device.load_state_dict(torch.load('./Net_device/net_device' + str(index_device) + '.pkl'))
        MyNN_Device = MyNN_Device.to(DEVICE)
        for i in range(test_batch_num):
            data_input = test_data[i].to(DEVICE)
            label_input = test_label[i].to(DEVICE)
            test_temp_out = MyNN_Device(data_input)
            torch.save({'test_temp_out': test_temp_out, 'test_label': label_input},
                       './test/test_temp' + str(index_device))
            conn3.send('testbegin')
            recall = conn3.recv()
            if recall == 'testnext':
                pass
            else:
                print('error test')
                break
        conn3.send('testover')
    conn3.send('begin')
    # load model
    if IFConvergence:
        MyNN_Device.load_state_dict(torch.load('./Net_device/net_device' + str(index_device) + '.pkl'))
    else:
        pass
    index_data = iteration % train_batch_num
    starttime_forward = time.time()
    data_input = train_data[index_data].to(DEVICE)
    label_input = train_label[index_data].to(DEVICE)
    temp_out = MyNN_Device(data_input)
    # temp_out_flatten = temp_out.view(temp_out.size(0), -1)
    # loss1 = criterion(temp_out_flatten, label_input.long())
    # showLoss(loss1, iteration, step_gap=50)
    endtime_foward = time.time()
    time_forward_list.append(endtime_foward - starttime_forward)
    torch.save({'temp_out': temp_out, 'label': label_input}, './temp/temp_out' + str(index_device))
    conn3.send('Forward1')
    recall = conn3.recv()
    if recall == 'Backward2':
        pass
    else:
        print('error!')
        break
    if IFConvergence:  # if device net converge
        MyNN_Device.zero_grad()
        temp_out.grad = torch.load('./temp/temp_grad' + str(index_device))
        # loss2 = torch.load('./temp/loss2' + str(index_device))
        starttime_back = time.time()
        temp_out.backward(temp_out.grad)
        endtime_back = time.time()
        saveParamGrad(MyNN_Device, path='./Net_device/net_device' + str(index_device))
        time_backward_list.append(endtime_back - starttime_back)
    else:
        pass
    conn2.send('train_d')
    recall = conn2.recv()
    if recall == 'updateover':
        pass
    else:
        print('error update')
        break
    iteration += 1
conn2.close()
conn3.close()
