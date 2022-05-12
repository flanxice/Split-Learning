# -*- coding = utf-8 -*-
# @Time : 2022/1/16 13:47
# @Author : SBP
# @File : Communication_Net.py
# @Software : PyCharm

from socket import socket, AF_INET, SOCK_STREAM
import pickle
import torch


def recvall(self, length):
    data = b''
    while len(data) < length:  # 循环接收数据
        data += self.recv(length - len(data))
    return data


socket.recvall = recvall  # 给socket类添加recvall方法


def recv_decode_data(self, buffer_size):
    data_b = self.recvall(buffer_size)
    data = data_b.decode('utf-8')
    return data


def send_encode_data(self, data):
    length = self.send(bytes(str(data), 'utf-8'))
    return length


def recv_until_q(self):
    data = b''
    while True:
        data_temp = self.recvall(1)
        # print(data_temp)
        if data_temp != b'q':
            data += data_temp
            # print(data)
        else:
            break
    return data


socket.recv_until_q = recv_until_q
socket.send_encode_data = send_encode_data
socket.recv_decode_data = recv_decode_data


def send_pickle_data(self, data):
    data_b = pickle.dumps(data)
    length = len(data_b)
    self.send_encode_data(str(length) + 'q')
    length = self.send(data_b)
    return length


def recv_pickle_data(self):
    length = int(self.recv_until_q())
    data_b = self.recvall(length)
    data = pickle.loads(data_b)
    return data


socket.recv_pickle_data = recv_pickle_data
socket.send_pickle_data = send_pickle_data


class ClientNet:
    def __init__(self, net):
        self.net = net
        self.conn = None

    def get_conn_connect(self, port, host):
        addr = (host, port)
        conn = socket(AF_INET, SOCK_STREAM)
        while True:
            try:
                conn.connect(addr)
                print('connect successfully')
                break
            except:
                pass
        self.conn = conn

    def conn_send_code(self, data):
        length = self.conn.send_encode_data(data)
        return length

    def conn_send_pickle(self, data):
        length = self.conn.send_pickle_data(data)
        return length

    def conn_recv_code(self, length):
        data = self.conn.recv_decode_data(length)
        return data

    def conn_recv_pickle(self):
        data = self.conn.recv_pickle_data()
        return data

    def conn_recv_uq(self):
        data = self.conn.recv_until_q()
        return data

    def net_forward(self, data):
        out = self.net(data)
        return out

    def net_backward(self, out, grad):
        self.net.zero_grad()
        out.backward(grad)

    def net_get_grad(self):
        net_grads = []
        for param in self.net.parameters():
            net_grads.append(param.grad)
        return net_grads

    def net_update(self, net_state_dict):
        self.net.load_state_dict()


class ServerConn:
    def __init__(self):
        self.conn = None

    def get_conn_connect(self, port, host='0.0.0.0'):
        addr = (host, port)
        tcpListener = socket(AF_INET, SOCK_STREAM)
        tcpListener.bind(addr)
        tcpListener.listen(1)
        conn, addr = tcpListener.accept()
        print('connect successful', addr)
        self.conn = conn

    def conn_send_code(self, data):
        length = self.conn.send_encode_data(data)
        return length

    def conn_send_pickle(self, data):
        length = self.conn.send_pickle_data(data)
        return length

    def conn_recv_code(self, length):
        data = self.conn.recv_decode_data(length)
        return data

    def conn_recv_pickle(self):
        data = self.conn.recv_pickle_data()
        return data

    def conn_recv_uq(self):
        data = self.conn.recv_until_q()
        return data


class ServerNet:
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion
        self.num_correct = 0

    def net_forward_backward(self, data, label):
        out = self.net(data)
        loss = self.criterion(out, label)
        loss_a_batch = loss * label.size(0)
        self.net.zero_grad()
        loss.backward()
        return loss, loss_a_batch, data.grad

    def net_get_grad(self):
        net_grads = []
        for param in self.net.parameters():
            net_grads.append(param.grad)
        return net_grads

    def forward_predict(self, temp, label):
        self.net.zero_grad()
        out = self.net(temp)
        _, pred = torch.max(out, 1)
        self.num_correct += (pred == label).sum()
        loss = self.criterion(out, label.long())
        loss_test_batch = loss * label.size(0)
        return loss, loss_test_batch

    def net_update(self, net_grads):
        self.net.zero_grad()
        for index, param in enumerate(self.net.parameters()):
            param.grad = net_grads[index]


def net_average_grads(net, batch_size_train, grads, device_num, DEVICE='cpu'):
    for index, param in enumerate(net.parameters()):
        param.grad = (grads[0][index] * batch_size_train[0]).to(DEVICE)
        for i in range(1, device_num):
            param.grad = param.grad + (grads[i][index] * batch_size_train[i]).to(DEVICE)
        param.grad = param.grad / sum(batch_size_train)
    return net


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
