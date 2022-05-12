# -*- coding = utf-8 -*-
# @Time : 2022/3/30 15:39
# @Author : SBP
# @File : myfunc_an.py
# @Software : PyCharm

import numpy as np
import torch


def value_all_kinds(k, W, T, Cf, Cb, batch_size, form):
    '''
    :param k: split point
    :param W: Model cutting face size
    :param T: Amount of data transmitted
    :param Cf: Calculation amount of forward propagation (float)
    :param Cb: Calculation amount of back propagation   (float)
    :param batch_size:  train batch_size = 500
    :param form: "c":converge , "n":non-converge , "f":federal-learning
    :return:
    '''
    res_C, res_T = -1, -1
    if form == 'c':
        res_C = batch_size * (sum(Cf[:k]) + sum(Cb[:k]))
        res_T = batch_size * (2 * T[k - 1] + 1) + 2 * sum(W[:k])
    elif form == 'n':
        res_C = batch_size * sum(Cf[:k])
        res_T = batch_size * (T[k - 1] + 1)
    elif form == 'f':
        res_C = batch_size * (sum(Cf) + sum(Cb))
        res_T = 2 * sum(W)
    return res_C, res_T


if __name__ == '__main__':
    W = [160, 0, 0, 4640, 0, 0, 0, 1098300, 0, 70100, 0, 1010]
    T = [12544, 12544, 3136, 6272, 6272, 1568, 1568, 700, 700, 100, 100, 10]
    Cf = [250880, 0, 0, 909440, 0, 0, 0, 1097600, 0, 70000, 0, 1000]
    Cb = [250880, 0, 0, 909440, 0, 0, 0, 1097600, 0, 70000, 0, 1000]
    n = 12  # Number of layers all
    alpha, beta = 1 / 100000000, 1 / (10e6 / 64)
    value_min = np.inf
    k_min = 0
    value_converge = [0]
    for k in range(1, n + 1):
        res_C, res_T = value_all_kinds(k, W, T, Cf, Cb, batch_size=500, form='c')
        value = alpha * res_C + beta * res_T
        value_converge.append(value)
        if value < value_min:
            value_min = value
            k_min = k
    print('converge : sp = {}'.format(k_min))
    print('time = {}\n'.format(value_min))

    value_min = np.inf
    k_min = 0
    value_non_converge = [0]
    for k in range(1, n + 1):
        res_C, res_T = value_all_kinds(k, W, T, Cf, Cb, batch_size=500, form='n')
        value = alpha * res_C + beta * res_T
        value_non_converge.append(value)
        if value < value_min:
            value_min = value
            k_min = k
    print('non-converge : sp = {}'.format(k_min))
    print('time = {}\n'.format(value_min))

    value_min = np.inf
    k_min = 0
    res_C, res_T = value_all_kinds(k, W, T, Cf, Cb, batch_size=128, form='f')
    value = alpha * res_C + beta * res_T
    print('f  : time = {}'.format(value))
    torch.save({'value_converge': value_converge, 'value_non_converge': value_non_converge, 'federal': value},
               './value_weight')
