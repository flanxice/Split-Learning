# -*- coding = utf-8 -*-
# @Time : 2021/11/12 10:12
# @Author : SBP
# @File : Server_Center.py
# @Software : PyCharm

from torch import optim
from MyFunctions import *
from TrainAndTest import *
from Net import *
from twilio.rest import Client as Client_m
from CONTROL import *

learning_rate = 0.001
num_device = 10  # 设备个数
num_iterations = 800  # 迭代次数 大概1epoch = 60000/1/64
step = 1  # 每隔1测试一次
split_points = [3] * 10
train_data_num = [1] * 10  # 训练数据量 1:1 # 一共60000
test_data_num = [1] * 10  # 测试集数据量比值 # 一共10000
train_batch_size = [256] * 10
test_batch_size = [200] * 10

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print(DEVICE)

# 运行 Processing 分割数据
makedir()
os.system("bash ./Processing.sh")

print("------------create connect of processing--------------")
address0 = ("localhost", 5400 + ADDRESS)
listener0 = Listener(address0, authkey=b"password")

print("---------------built connect--------------")
conn_process = listener0.accept()
print("connection accepted from:", listener0.last_accepted)
torch.save({"train_data_num": train_data_num, "test_data_num": test_data_num}, "./Processing_data_split")
conn_process.send(str(num_device))

try:
    recall = conn_process.recv()
    print("---------------- process done ---------------")
except:
    print("------------- process fail -------------------")
    exit()
if IFConvergence:
    MyModel_C = MyModel_Whole.to(DEVICE)
else:
    _, MyModel_C = getMyModel(split_point=split_points[0])
    MyModel_C = MyModel_C.to(DEVICE)
optimizer = optim.Adam(MyModel_C.parameters(), lr=learning_rate)

Listeners1 = CreateListener(num_device, address_number=6400 + ADDRESS)
Listeners2 = CreateListener(num_device, address_number=7400 + ADDRESS)

for i in range(1, num_device + 1):
    os.system("bash ./Server_Separate" + str(i) + ".sh&")
for i in range(1, num_device + 1):
    os.system("bash ./Device" + str(i) + ".sh&")

print("---------------- build connect ---------------")
# auto print
conns1 = buildConnect(Listeners1)
conns2 = buildConnect(Listeners2)
print("---------------- send split point ---------------")
connSendMore(conns1, split_points)
connSendMore(conns2, split_points)
print("---------------- send train_batch_size, test_batch_size ---------------")
connSendMore(conns2, train_batch_size)
connSendMore(conns2, test_batch_size)
print("---------------- send num_iterations and step---------------")
connSendSingle(conns2, num_iterations)
connSendSingle(conns2, step)
print("---------------- send over ---------------")

# 存储每个用户实际训练和测试数据量
signals = connRecv(conns2)
target_process = [item == "preprocessdone" for item in signals]
train_num, test_num = [], []
train_all_num, test_all_num = 1, 1
if all(target_process) == True:
    for num in range(num_device):
        temp_train = torch.load("./data_num_device/train_num" + str(num + 1))
        temp_test = torch.load("./data_num_device/test_num" + str(num + 1))
        train_num.append(temp_train)
        test_num.append(temp_test)
    train_all_num = sum(train_num)
    test_all_num = sum(test_num)

print("---------------- start control ---------------")

test_step = 0
test_loss_list, test_acc_list = [], []
train_loss_list = []
# print(conns1)
while True:
    # print('111')
    signals = connRecv(conns1)
    # print(signals)
    target_finish = [item == "finish" for item in signals]
    target_testover = [item == "testover" for item in signals]
    target_train_s = [item == "train_s" for item in signals]
    if all(target_finish) == True:
        print("train over")
        torch.save({"test_loss_list": test_loss_list, "test_acc_list": test_acc_list}, "./result/test_acc_loss_all")
        torch.save(train_loss_list, "./result/train_loss_all")
        break
    elif all(target_testover) == True:
        print("test")
        test_step += step
        loss_sum, num_acc_sum = 0, 0
        for num in range(num_device):
            loss_acc = torch.load("./loss_acc_s/loss_acc_s" + str(num + 1))
            loss = loss_acc["loss_test"]
            num_correct = loss_acc["num_correct"]
            print("device name = {}".format(num + 1))
            print("loss = {}, acc = {}".format(loss / test_num[num], num_correct / test_num[num]))
            loss_sum += loss
            num_acc_sum += num_correct
        print("step = {} => ".format(test_step), end="")
        loss_all, acc_all = loss_sum / test_all_num, num_acc_sum / test_all_num
        print("loss = {}, acc = {}".format(loss_all, acc_all))
        test_loss_list.append(loss_all)
        test_acc_list.append(acc_all)

    elif all(target_train_s) == True:
        signals_device = connRecv(conns2)
        target_train_d = [item == "train_d" for item in signals_device]
        if all(target_train_d) == True:
            pass
        else:
            print("error")
            break
        grads_net = []
        loss_train = 0
        length_device = []
        optimizer.zero_grad()
        for num in range(num_device):
            grads_server_temp = torch.load("./Net_server_s/net_server_s" + str(num + 1))
            loss_temp = torch.load("./loss/loss_server" + str(num + 1))
            loss_train += loss_temp.item() * train_batch_size[num]
            if IFConvergence:
                grads_device_temp = torch.load("./Net_device/net_device" + str(num + 1))
                length_device_temp = len(grads_device_temp)
                length_device.append(length_device_temp)
            else:
                grads_device_temp = []
            grads_device_temp.extend(grads_server_temp)
            grads_net.append(grads_device_temp)
        loss_train = loss_train / sum(train_batch_size)
        # print('loss_train = {}'.format(loss_train))  #输出每次迭代 loss_train
        train_loss_list.append(loss_train)
        # average
        if IFConvergence:
            MyModel_C = averageGrads(MyModel_C, grads_net, train_batch_size, num_device, DEVICE)
            optimizer.step()
            grads_net_new = getParamGrad(myNN=MyModel_C)
            for num in range(num_device):
                grads_device_new = grads_net_new[0: length_device[num]]
                grads_server_new = grads_net_new[length_device[num]:]
                MyModel_Device_temp, MyModel_Sever_temp = getMyModel(split_point=split_points[num])
                MyModel_Device_temp, MyModel_Sever_temp = makesplitNet(MyModel_Device_temp,
                                                                       MyModel_Sever_temp,
                                                                       grads_device_new,
                                                                       grads_server_new)
                torch.save(MyModel_Device_temp.state_dict(), "./Net_device/net_device" + str(num + 1) + ".pkl")
                torch.save(MyModel_Sever_temp.state_dict(), "./Net_server_s/net_server_s" + str(num + 1) + ".pkl")
        else:
            MyModel_C = averageGrads(MyModel_C, grads_net, train_batch_size, num_device, DEVICE)
            optimizer.step()
            for num in range(num_device):
                torch.save(MyModel_C.state_dict(), "./Net_server_s/net_server_s" + str(num + 1) + ".pkl")
        # print('update over')
        connSendSingle(conns2, "updateover")
    else:
        print("error")
        break

conn_process.close()
listener0.close()
closeconn(conns1)
closeListener(Listeners1)
closeconn(conns2)
closeListener(Listeners2)

sid = 'ACe6aec27da029cd81f715c49a21ce9bb2'
token = '83ca6e4d2e43cf1a7b7d307f42aec843'
client_Message = Client_m(sid, token)
client_Message.messages.create(
    to='+8613646501257',
    from_='+17194972960',
    body='训练已完成!'
)
