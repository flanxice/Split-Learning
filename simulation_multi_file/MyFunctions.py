# -*- coding = utf-8 -*-
# @Time : 2021/10/12 14:23
# @Author : SBP
# @File : MyFunctions.py
# @Software : PyCharm

from multiprocessing.connection import Client, Listener
import os

def CreateListener(num_device, address_number):
    Listeners = []
    for i in range(num_device):
        address = ('localhost', address_number + 1 + i)
        listener = Listener(address, authkey=b"password")
        Listeners.append(listener)
    return Listeners


def buildConnect(Listeners):
    conns = []
    for i in range(len(Listeners)):
        conn = Listeners[i].accept()
        print('connection ' + ' accepted from:', Listeners[i].last_accepted)
        conns.append(conn)
    return conns


def connSendSingle(conns, data):
    for item in conns:
        item.send(str(data))


def connSendMore(conns, data_list):
    for i in range(len(data_list)):
        conns[i].send(str(data_list[i]))


# 接收所有消息
def connRecv(conns):
    datas = []
    for i in range(len(conns)):
        data = conns[i].recv()
        datas.append(data)
        # print("device " + str(i + 1) + " : {}".format(data))
    return datas


def closeconn(conns):
    for item in conns:
        item.close()


def closeListener(Listeners):
    for item in Listeners:
        item.close()

def makedir():
    if not os.path.exists("./log/"):
        os.makedirs("./log/")
    if not os.path.exists("./loss/"):
        os.makedirs("./loss/")
    if  not os.path.exists("./data_num_device/"):
        os.makedirs("./data_num_device/")
    if not os.path.exists("./loss_acc_s/"):
        os.makedirs("./loss_acc_s/")
    if not os.path.exists("./Net_device/"):
        os.makedirs("./Net_device/")
    if not os.path.exists("./Net_server_s/"):
        os.makedirs("./Net_server_s/")
    if not os.path.exists("./result/"):
        os.makedirs("./result/")
    if not os.path.exists("./temp/"):
        os.makedirs("./temp/")
    if not os.path.exists("./test/"):
        os.makedirs("./test/")