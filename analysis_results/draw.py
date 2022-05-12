# -*- coding = utf-8 -*-
# @Time : 2022/3/28 16:20
# @Author : SBP
# @File : draw.py
# @Software : PyCharm

from matplotlib import pyplot as plt
import torch
import os
import sys

sys.path.append('./results')
dirs = os.listdir('./results')

NonConvergence_loss_test, NonConvergence_accs, NonConvergence_loss_train = [], [], []

results_temp = torch.load('./results/result_' + str(1) + '_10_500_200_True_True')
Convergence_loss_test = torch.tensor(results_temp['loss_test'])
Convergence_accs = torch.tensor(results_temp['acc_test'])
Convergence_loss_train = torch.tensor(results_temp['loss_train'])

for i in range(1, 12):
    results_temp = torch.load('./results/result_' + str(i) + '_10_500_200_True_False')
    loss_test = results_temp['loss_test']
    acc_test = results_temp['acc_test']
    loss_train = results_temp['loss_train']
    NonConvergence_loss_test.append(loss_test)
    NonConvergence_accs.append(acc_test)
    NonConvergence_loss_train.append(loss_train)

NonConvergence_loss_test = torch.tensor(NonConvergence_loss_test)
NonConvergence_accs = torch.tensor(NonConvergence_accs)
NonConvergence_loss_train = torch.tensor(NonConvergence_loss_train)


def draw1():
    plt.figure(dpi=1024)
    plt.plot(Convergence_accs[:, 0], Convergence_accs[:, 1], label='Convergence')
    plt.plot(NonConvergence_accs[2, :, 0], NonConvergence_accs[2, :, 1],
             label='split_point = 3')
    plt.plot(NonConvergence_accs[5, :, 0], NonConvergence_accs[5, :, 1],
             label='split_point = 6')
    plt.plot(NonConvergence_accs[8, :, 0], NonConvergence_accs[8, :, 1],
             label='split_point = 9')
    plt.plot(NonConvergence_accs[9, :, 0], NonConvergence_accs[9, :, 1],
             label='split_point = 10')
    plt.legend()
    plt.title('Test-Accuracy')
    plt.xlabel('iteration ')
    plt.savefig('./Test_Accuracy.png')
    # plt.show()


def draw2():
    plt.figure(dpi=1024)
    plt.plot(Convergence_loss_test[:, 0], Convergence_loss_test[:, 1], label='Convergence')
    plt.plot(NonConvergence_loss_test[2, :, 0], NonConvergence_loss_test[2, :, 1],
             label='split_point = 3')
    plt.plot(NonConvergence_loss_test[5, :, 0], NonConvergence_loss_test[5, :, 1],
             label='split_point = 6')
    plt.plot(NonConvergence_loss_test[8, :, 0], NonConvergence_loss_test[8, :, 1],
             label='split_point = 9')
    plt.plot(NonConvergence_loss_test[9, :, 0], NonConvergence_loss_test[9, :, 1],
             label='split_point = 10')
    plt.legend()
    plt.title('Test-Loss')
    plt.xlabel('iteration ')
    plt.savefig('./Test_Loss.png')
    # plt.show()


def draw3():
    plt.figure(dpi=1024)
    plt.plot(Convergence_loss_train[:, 0], Convergence_loss_train[:, 1], label='Convergence')
    plt.plot(NonConvergence_loss_train[2, :, 0], NonConvergence_loss_train[2, :, 1],
             label='split_point = 3')
    plt.plot(NonConvergence_loss_train[5, :, 0], NonConvergence_loss_train[5, :, 1],
             label='split_point = 6')
    plt.plot(NonConvergence_loss_train[8, :, 0], NonConvergence_loss_train[8, :, 1],
             label='split_point = 9')
    plt.plot(NonConvergence_loss_train[9, :, 0], NonConvergence_loss_train[9, :, 1],
             label='split_point = 10')
    plt.legend()
    plt.title('Train-Loss')
    plt.xlabel('iteration ')
    plt.savefig('./Train_Loss.png')
    # plt.show()


values = torch.load('./analysis/value_weight')
value_converge = values['value_converge']
value_non_converge = values['value_non_converge']
federal = values['federal']


## draw 4 - 6 : converge acc,loss,loss - training cost
def draw4():
    plt.figure(dpi=1024)
    plt.plot(Convergence_accs[:, 0] * value_converge[3], Convergence_accs[:, 1], label='Convergence split-point = 3')
    plt.plot(Convergence_accs[:, 0] * value_converge[6], Convergence_accs[:, 1], label='Convergence split-point = 6')
    plt.plot(Convergence_accs[:, 0] * value_converge[9], Convergence_accs[:, 1], label='Convergence split-point = 9')
    plt.plot(Convergence_accs[:, 0] * value_converge[10], Convergence_accs[:, 1], label='Convergence split-point = 10')
    plt.legend()
    plt.title('Test-Accuracy')
    plt.xlabel('Training Cost')
    plt.savefig('./Converge_Test_Accuracy_cost.png')


def draw5():
    plt.figure(dpi=1024)
    plt.plot(Convergence_loss_train[:, 0] * value_converge[3], Convergence_loss_train[:, 1],
             label='Convergence split-point = 3')
    plt.plot(Convergence_loss_train[:, 0] * value_converge[6], Convergence_loss_train[:, 1],
             label='Convergence split-point = 6')
    plt.plot(Convergence_loss_train[:, 0] * value_converge[9], Convergence_loss_train[:, 1],
             label='Convergence split-point = 9')
    plt.plot(Convergence_loss_train[:, 0] * value_converge[10], Convergence_loss_train[:, 1],
             label='Convergence split-point = 10')
    plt.legend()
    plt.title('Train-Loss')
    plt.xlabel('Training Cost')
    plt.savefig('./Converge_Train_Loss_cost.png')


def draw6():
    plt.figure(dpi=1024)
    plt.plot(Convergence_loss_test[:, 0] * value_converge[3], Convergence_loss_test[:, 1],
             label='Convergence split-point = 3')
    plt.plot(Convergence_loss_test[:, 0] * value_converge[6], Convergence_loss_test[:, 1],
             label='Convergence split-point = 6')
    plt.plot(Convergence_loss_test[:, 0] * value_converge[9], Convergence_loss_test[:, 1],
             label='Convergence split-point = 9')
    plt.plot(Convergence_loss_test[:, 0] * value_converge[10], Convergence_loss_test[:, 1],
             label='Convergence split-point = 10')
    plt.legend()
    plt.title('Test-Loss')
    plt.xlabel('Training Cost')
    plt.savefig('./Converge_Test_Loss_cost.png')


##### non-converge
def draw7():
    plt.figure(dpi=1024)
    plt.plot(NonConvergence_accs[2, :, 0] * value_non_converge[3], NonConvergence_accs[2, :, 1],
             label='Non_Converge split_point = 3')
    plt.plot(NonConvergence_accs[5, :, 0] * value_non_converge[6], NonConvergence_accs[5, :, 1],
             label='Non_Converge split_point = 6')
    plt.plot(NonConvergence_accs[8, :, 0] * value_non_converge[9], NonConvergence_accs[8, :, 1],
             label='Non_Converge split_point = 9')
    plt.plot(NonConvergence_accs[9, :, 0] * value_non_converge[10], NonConvergence_accs[9, :, 1],
             label='Non_Converge split_point = 10')
    plt.legend()
    plt.title('Test-Accuracy')
    plt.xlabel('Training Cost')
    plt.savefig('./Non_Converge_Test_Accuracy_cost.png')


def draw8():
    plt.figure(dpi=1024)
    plt.plot(NonConvergence_loss_train[2, :, 0] * value_non_converge[3], NonConvergence_loss_train[2, :, 1],
             label='Non_Converge split_point = 3')
    plt.plot(NonConvergence_loss_train[5, :, 0] * value_non_converge[6], NonConvergence_loss_train[5, :, 1],
             label='Non_Converge split_point = 6')
    plt.plot(NonConvergence_loss_train[8, :, 0] * value_non_converge[9], NonConvergence_loss_train[8, :, 1],
             label='Non_Converge split_point = 9')
    plt.plot(NonConvergence_loss_train[9, :, 0] * value_non_converge[10], NonConvergence_loss_train[9, :, 1],
             label='Non_Converge split_point = 10')
    plt.legend()
    plt.title('Train-Loss')
    plt.xlabel('Training Cost')
    plt.savefig('./Non_Converge_Train_Loss_cost.png')


def draw9():
    plt.figure(dpi=1024)
    plt.plot(NonConvergence_loss_test[2, :, 0] * value_non_converge[3], NonConvergence_loss_test[2, :, 1],
             label='Non_Converge split_point = 3')
    plt.plot(NonConvergence_loss_test[5, :, 0] * value_non_converge[6], NonConvergence_loss_test[5, :, 1],
             label='Non_Converge split_point = 6')
    plt.plot(NonConvergence_loss_test[8, :, 0] * value_non_converge[9], NonConvergence_loss_test[8, :, 1],
             label='Non_Converge split_point = 9')
    plt.plot(NonConvergence_loss_test[9, :, 0] * value_non_converge[10], NonConvergence_loss_test[9, :, 1],
             label='Non_Converge split_point = 10')
    plt.legend()
    plt.title('Test-Loss')
    plt.xlabel('Training Cost')
    plt.savefig('./Non_Converge_Test_Loss_cost.png')


def draw10():
    plt.figure(dpi=1024)
    plt.plot(Convergence_accs[:, 0] * value_converge[6], Convergence_accs[:, 1], label='Convergence split_point = 6')
    plt.plot(NonConvergence_accs[5, :, 0] * value_non_converge[6], NonConvergence_accs[5, :, 1],
             label='Non_Converge split_point = 6')
    plt.plot(Convergence_accs[:, 0] * federal, Convergence_accs[:, 1], label='federal learning')
    plt.legend()
    plt.title('Test-Accuracy')
    plt.xlabel('Training Cost')
    plt.savefig('./kinds3_Test_Accuracy_cost.png')

def draw11():
    plt.figure(dpi=1024)
    plt.plot(Convergence_loss_test[:, 0] * value_converge[6], Convergence_loss_test[:, 1], label='Convergence split_point = 6')
    plt.plot(NonConvergence_loss_test[5, :, 0] * value_non_converge[6], NonConvergence_loss_test[5, :, 1],
             label='Non_Converge split_point = 6')
    plt.plot(Convergence_loss_test[:, 0] * federal, Convergence_loss_test[:, 1], label='federal learning')
    plt.legend()
    plt.title('Test-Loss')
    plt.xlabel('Training Cost')
    plt.savefig('./kinds3_Test_Loss_cost.png')

def draw12():
    plt.figure(dpi=1024)
    plt.plot(Convergence_loss_train[:, 0] * value_converge[6], Convergence_loss_train[:, 1], label='Convergence split_point = 6')
    plt.plot(NonConvergence_loss_train[5, :, 0] * value_non_converge[6], NonConvergence_loss_train[5, :, 1],
             label='Non_Converge split_point = 6')
    plt.plot(Convergence_loss_train[:, 0] * federal, Convergence_loss_train[:, 1], label='federal learning')
    plt.legend()
    plt.title('Train-Loss')
    plt.xlabel('Training Cost')
    plt.savefig('./kinds3_Train_Loss_cost.png')

if __name__ == '__main__':
    draw1()
    draw2()
    draw3()
    draw4()
    draw5()
    draw6()
    draw7()
    draw8()
    draw9()
    draw10()
    draw11()
    draw12()
    pass
