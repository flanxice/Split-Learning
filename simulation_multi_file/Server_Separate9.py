# -*- coding = utf-8 -*-
# @Time : 2021/10/12 15:22
# @Author : SBP
# @File : Server_Separate.py
# @Software : PyCharm

from multiprocessing.connection import Client, Listener
from Net import *
from TrainAndTest import *
from CONTROL import *
import time

index_server = 9
# getGPUID(index=index_server)
DEVICE = torch.device("cpu")
print(DEVICE)
# client 1 receive from server center
address1 = ('localhost', 6400 + index_server + ADDRESS)
conn1 = Client(address1, authkey=b"password")

# listener 3 communicate with device1
address3 = ('localhost', 6900 + index_server + ADDRESS)
listener3 = Listener(address3, authkey=b"password")

split_point = int(conn1.recv())
print('split_point = {}'.format(split_point))

print('---------------built connect--------------')
conn3 = listener3.accept()
print("connection accepted from:", listener3.last_accepted)

# model receive from device1
_, MyNN_Server = getMyModel(split_point=split_point, Net_List=NetList)
MyNN_Server = MyNN_Server.to(DEVICE)
criterion = nn.CrossEntropyLoss()
# MyNN_Server.load_state_dict(torch.load('./net_init/net_server_init_sp' + str(split_point) + '.pkl'))
torch.save(MyNN_Server.state_dict(), './Net_server_s/net_server_s' + str(index_server) + '.pkl')

# wait for device preparing
processdata = conn3.recv()
print(processdata)

# start train and test
test_loss_list, test_accnum_list = [], []
train_loss_list = []
time_forward_list, time_backward_list = [], []
while True:
    target = conn3.recv()
    # print(target)
    if target == '-1':
        conn1.send('finish')  # to SERVER
        print('train over')
        torch.save({'test_loss_list': test_loss_list, 'test_accnum_list': test_accnum_list},
                   './result/test_loss_acc' + str(index_server))  # 这里是乘过batch_size的loss
        torch.save(train_loss_list, './result/train_loss' + str(index_server))
        torch.save({'time_forward_list': time_forward_list, 'time_backward_list': time_backward_list},
                   './result/time_server' + str(index_server))
        break
    elif target == 'begin':
        MyNN_Server.load_state_dict(torch.load('./Net_server_s/net_server_s' + str(index_server) + '.pkl'))
    elif target == 'Forward1':
        temp = torch.load('./temp/temp_out' + str(index_server))
        temp_out, label = temp['temp_out'], temp['label']
        starttime_forward = time.time()
        out = MyNN_Server(temp_out.to(DEVICE))
        loss2 = criterion(out, label.long().to(DEVICE))
        endtime_forward = time.time()
        time_forward_list.append(endtime_forward - starttime_forward)
        # backforward
        MyNN_Server.zero_grad()
        loss2.backward(retain_graph=False)  # retain_graph=True
        endtime_back = time.time()
        time_backward_list.append(endtime_back - endtime_forward)
        if IFConvergence:  # if device net converge
            torch.save(temp_out.grad, './temp/temp_grad' + str(index_server))  # save backward grad
            # torch.save(loss2.cpu(), './temp/loss2' + str(index_server))
        saveParamGrad(MyNN_Server, path='./Net_server_s/net_server_s' + str(index_server))
        # torch.save(MyNN_Server.state_dict(), './Net_server_s/net_server_s' + str(index_server) + '.pkl')
        torch.save(loss2, './loss/loss_server' + str(index_server))
        train_loss_list.append(loss2.item())
        conn1.send('train_s')
        conn3.send('Backward2')
    elif target == 'test':
        print('test')
        MyNN_Server.load_state_dict(torch.load('./Net_server_s/net_server_s' + str(index_server) + '.pkl'))
        num_correct = 0
        loss_test = 0
    elif target == 'testbegin':
        temp = torch.load('./test/test_temp' + str(index_server))
        test_temp, test_label = temp['test_temp_out'], temp['test_label']
        out = MyNN_Server(test_temp.to(DEVICE))
        loss_test_temp = criterion(out, test_label.long().to(DEVICE))
        loss_test += loss_test_temp.data.item() * test_label.size(0)
        _, pred = torch.max(out, 1)
        num_correct += (pred == test_label.to(DEVICE)).sum()
        conn3.send('testnext')
    elif target == 'testover':
        torch.save({'loss_test': loss_test, 'num_correct': num_correct}, './loss_acc_s/loss_acc_s' + str(index_server))
        test_loss_list.append(loss_test)
        test_accnum_list.append(num_correct)
        conn1.send('testover')
    else:
        print('ERROR!')
        break

conn1.close()
conn3.close()
