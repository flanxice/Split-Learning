# -*- coding = utf-8 -*-
# @Time : 2022/4/23 21:43
# @Author : SBP
# @File : myClass_new.py
# @Software : PyCharm

import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
from matplotlib.figure import Figure
import os
import sys
import time
from Server import *

# Lock0 = threading.Lock()  # for test 4 params
# Lock1 = threading.Lock()  # for train 2 params

# global loss_test_every_device, acc_test_every_device
# global loss_test_all_device, acc_test_all_device
# global loss_train_every_device, loss_train_all_device
loss_test_every_device_pre = None  # 所有用户的loss
acc_test_every_device_pre = None  # 所有用户的准确率
loss_all_device_pre, acc_all_device_pre = None, None
train_loss_device_pre, train_loss_all_pre = None, None

# num_device = 0

start_flag = 0


class FrameControl:
    def __init__(self, root):
        self.right_control = tk.LabelFrame(root, text='Control Interface', font=('微软雅黑', 30, 'italic'))
        self.right_control.place(relx=0.75, rely=0, relwidth=0.25, relheight=0.41)
        self.label0 = tk.Label(self.right_control, text='Num of Device:', font=('微软雅黑', 15, 'italic')).grid(row=0,
                                                                                                            column=0,
                                                                                                            sticky=tk.W)
        self.input0 = tk.Entry(self.right_control)
        self.input0.insert(0, '2')
        self.input0.grid(row=0, column=1, sticky=tk.E)

        self.label1 = tk.Label(self.right_control, text='Split Point:', font=('微软雅黑', 15, 'italic')).grid(row=1,
                                                                                                          column=0,
                                                                                                          sticky=tk.W)

        self.input1 = tk.Entry(self.right_control)
        self.input1.insert(0, '3')
        self.input1.grid(row=1, column=1, sticky=tk.E)

        self.label9 = tk.Label(self.right_control, text='Learning Rate:', font=('微软雅黑', 15, 'italic')).grid(row=2,
                                                                                                            column=0,
                                                                                                            sticky=tk.W)

        self.input9 = tk.Entry(self.right_control)
        self.input9.insert(0, '0.001')
        self.input9.grid(row=2, column=1, sticky=tk.E)

        self.label2 = tk.Label(self.right_control, text='Train Batch Size:', font=('微软雅黑', 15, 'italic')).grid(
            row=3,
            column=0,
            sticky=tk.W)
        self.input2 = tk.Entry(self.right_control)
        self.input2.insert(0, '256')
        self.input2.grid(row=3, column=1, sticky=tk.E)

        self.label3 = tk.Label(self.right_control, text='Test Batch Size:', font=('微软雅黑', 15, 'italic')).grid(row=4,
                                                                                                              column=0,
                                                                                                              sticky=tk.W)
        self.input3 = tk.Entry(self.right_control)
        self.input3.insert(0, '256')
        self.input3.grid(row=4, column=1, sticky=tk.E)

        self.label4 = tk.Label(self.right_control, text='Num of Iteration:', font=('微软雅黑', 15, 'italic')).grid(
            row=5,
            column=0,
            sticky=tk.W)
        self.input4 = tk.Entry(self.right_control)
        self.input4.insert(0, '300')
        self.input4.grid(row=5, column=1, sticky=tk.E)

        self.label5 = tk.Label(self.right_control, text='Test Step:', font=('微软雅黑', 15, 'italic')).grid(row=6,
                                                                                                        column=0,
                                                                                                        sticky=tk.W)
        self.input5 = tk.Entry(self.right_control)
        self.input5.insert(0, '1')
        self.input5.grid(row=6, column=1, sticky=tk.E)

        self.input6 = tk.IntVar()
        tk.Checkbutton(self.right_control, text="If Converge", font=('微软雅黑', 15, 'italic'), variable=self.input6,
                       onvalue=1, offvalue=0).grid(row=7, column=0)
        self.input7 = tk.IntVar()
        tk.Checkbutton(self.right_control, text="Traditional", font=('微软雅黑', 15, 'italic'), variable=self.input7,
                       onvalue=1, offvalue=0).grid(row=7, column=1)

        self.start_button = tk.Button(self.right_control, text='Start', font=('微软雅黑', 15, 'italic'), width=15,
                                      height=1, command=self.button_on).grid(row=8, column=0, columnspan=2)
        self.right_info = Information(root)
        self.if_can_start = 0

    def button_on(self):
        p1 = threading.Thread(target=self.__button_on__)
        p1.start()

    def __button_on__(self):
        in_0 = self.input0.get()
        in_1 = self.input1.get()
        in_2 = self.input2.get()
        in_3 = self.input3.get()
        in_4 = self.input4.get()
        in_5 = self.input5.get()
        in_6 = self.input6.get()
        in_7 = self.input7.get()
        in_9 = self.input9.get()
        error_flag = 0
        if in_1.isdigit() and 0 < int(in_1) < 12:
            pass
        else:
            self.input1.delete(0, 'end')
            error_flag = 1
        for item0, item1 in [(in_0, self.input0), (in_2, self.input2), (in_3, self.input3), (in_4, self.input4),
                             (in_5, self.input5)]:
            if item0.isdigit() and int(item0) > 0:
                pass
            else:
                item1.delete(0, 'end')
                error_flag = 1
        if error_flag == 1:
            messagebox.showerror(title='Error',
                                 message='Num of Device, Split Point(1~11), Train Batch Size, Test Batch Size, Num of Iteration, Test Step should be integers(>0).')
        if self.if_can_start == 0:
            self.if_can_start = 1
            print('11111')
            left_frame = Frame_Pic(root)
            # self.right_info.get_dstrs(int(in_0))
            if int(in_7):
                print('22222')
                # 传统
                # os.system('nohup python ../testnet/Server.py ' + cmd)
            else:
                print('33333')
                global loss_test_every_device_pre
                global acc_test_every_device_pre
                global loss_all_device_pre, acc_all_device_pre
                global train_loss_device_pre, train_loss_all_pre
                global start_flag
                start_flag = 1
                # Lock0.acquire()
                loss_test_every_device_pre = torch.zeros(int(in_0))
                acc_test_every_device_pre = torch.zeros(int(in_0))
                loss_all_device_pre, acc_all_device_pre = 0, 0
                num_device_frame = int(in_0) if int(in_0) <= 3 else 3
                # Lock0.release()
                train_loss_device_pre, train_loss_all_pre = torch.zeros(int(in_0)), 0
                c = MyServer(num_device=int(in_0), LR=float(in_9), split_point=int(in_1), num_iteration_end=int(in_4),
                             step=int(in_5), batch_size_train=int(in_2), batch_size_test=int(in_3),
                             IF_testall=0, DEVICE=torch.device('cpu'))
                while True:
                    if c.iteration > c.num_iteration_end:
                        print('over')
                        break
                    if c.iteration % c.step == c.step - 1:
                        print('Test')
                        loss_test_every_device_once, acc_test_every_device_once, loss_all_device, acc_all = c.test()
                        left_frame.get_threadings_test(loss_test_every_device_once, acc_test_every_device_once,
                                                       loss_test_every_device_pre, acc_test_every_device_pre,
                                                       c.iteration, int(in_5), num_device_frame)
                        left_frame.draw_th_7(loss_all_device, loss_all_device_pre, c.iteration, int(in_5))
                        left_frame.draw_th_8(acc_all, acc_all_device_pre, c.iteration, int(in_5))
                        left_frame.threading_test_start(num_device_frame)
                        self.right_info.print_test_loss(loss_test_every_device_once, loss_all_device, c.iteration,
                                                        int(in_0))
                        self.right_info.print_test_acc(acc_test_every_device_once, acc_all, c.iteration, int(in_0))
                        loss_test_every_device_pre = deepcopy(loss_test_every_device_once)
                        acc_test_every_device_pre = deepcopy(acc_test_every_device_once)
                        loss_all_device_pre = loss_all_device
                        acc_all_device_pre = acc_all
                    print('Train')
                    loss_train_every_device, loss_train_all_device = c.train()
                    left_frame.get_threadings_train(loss_train_every_device, train_loss_device_pre, c.iteration,
                                                    num_device_frame)
                    left_frame.draw_th_9(loss_train_all_device, train_loss_all_pre, c.iteration)
                    left_frame.threading_train_start(num_device_frame)
                    self.right_info.print_train_loss(loss_train_every_device, loss_train_all_device, c.iteration,
                                                     int(in_0))
                    train_loss_device_pre = deepcopy(loss_train_every_device)
                    train_loss_all_pre = loss_train_all_device
                    # left_frame.wait_for_end_th(num_device_frame)
                    # Lock1.release()

            self.if_can_start = 0


class Frame_Pic:
    def __init__(self, root):
        self.left_Frame = tk.Frame(root, bg='white')
        self.left_Frame.place(relx=0, rely=0, relwidth=0.75, relheight=1)

        self.fig1 = plt.figure(dpi=50)
        self.ax1_0 = self.fig1.add_subplot(211)
        self.ax1_1 = self.fig1.add_subplot(212)
        self.ax1_0.set_title('test loss DEVICE 0')
        self.ax1_1.set_title('test acc DEVICE 0')
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.left_Frame)
        self.canvas1.get_tk_widget().place(relx=0, rely=0, relwidth=0.33, relheight=0.33)  # 放置位置

        self.fig2 = plt.figure(dpi=50)
        self.ax2_0 = self.fig2.add_subplot(211)
        self.ax2_1 = self.fig2.add_subplot(212)
        self.ax2_0.set_title('test loss DEVICE 1')
        self.ax2_1.set_title('test acc DEVICE 1')
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.left_Frame)
        self.canvas2.get_tk_widget().place(relx=0.33, rely=0, relwidth=0.33, relheight=0.33)  # 放置位置

        self.fig3 = plt.figure(dpi=50)
        self.ax3_0 = self.fig3.add_subplot(211)
        self.ax3_1 = self.fig3.add_subplot(212)
        self.ax3_0.set_title('test loss DEVICE 2')
        self.ax3_1.set_title('test acc DEVICE 2')
        self.canvas3 = FigureCanvasTkAgg(self.fig3, self.left_Frame)
        self.canvas3.get_tk_widget().place(relx=0.66, rely=0, relwidth=0.33, relheight=0.33)  # 放置位置

        self.fig4 = plt.figure(dpi=50)
        self.ax4_0 = self.fig4.add_subplot(111)
        self.ax4_0.set_title('train loss DEVICE 0')
        self.canvas4 = FigureCanvasTkAgg(self.fig4, self.left_Frame)
        self.canvas4.get_tk_widget().place(relx=0, rely=0.33, relwidth=0.33, relheight=0.33)  # 放置位置

        self.fig5 = plt.figure(dpi=50)
        self.ax5_0 = self.fig5.add_subplot(111)
        self.ax5_0.set_title('train loss DEVICE 1')
        self.canvas5 = FigureCanvasTkAgg(self.fig5, self.left_Frame)
        self.canvas5.get_tk_widget().place(relx=0.33, rely=0.33, relwidth=0.33, relheight=0.33)  # 放置位置

        self.fig6 = plt.figure(dpi=50)
        self.ax6_0 = self.fig6.add_subplot(111)
        self.ax6_0.set_title('train loss DEVICE 2')
        self.canvas6 = FigureCanvasTkAgg(self.fig6, self.left_Frame)
        self.canvas6.get_tk_widget().place(relx=0.66, rely=0.33, relwidth=0.33, relheight=0.33)  # 放置位置

        all_frame = tk.LabelFrame(self.left_Frame, text='DEVICE ALL:', bg='white', font=('微软雅黑', 30, 'italic'))
        all_frame.place(relx=0, rely=0.66, relwidth=1, relheight=0.33)
        self.fig7 = plt.figure(dpi=50)
        self.ax7_0 = self.fig7.add_subplot(111)
        self.ax7_0.set_title('test loss DEVICE ALL')
        self.canvas7 = FigureCanvasTkAgg(self.fig7, all_frame)
        self.canvas7.get_tk_widget().place(relx=0, rely=0, relwidth=0.33, relheight=1)  # 放置位置

        self.fig8 = plt.figure(dpi=50)
        self.ax8_0 = self.fig8.add_subplot(111)
        self.ax8_0.set_title('test acc DEVICE ALL')
        self.canvas8 = FigureCanvasTkAgg(self.fig8, all_frame)
        self.canvas8.get_tk_widget().place(relx=0.33, rely=0, relwidth=0.33, relheight=1)  # 放置位置

        self.fig9 = plt.figure(dpi=50)
        self.ax9_0 = self.fig9.add_subplot(111)
        self.ax9_0.set_title('train loss DEVICE ALL')
        self.canvas9 = FigureCanvasTkAgg(self.fig9, all_frame)
        self.canvas9.get_tk_widget().place(relx=0.66, rely=0, relwidth=0.33, relheight=1)  # 放置位置

        self.axs = [[self.ax1_0, self.ax1_1], [self.ax2_0, self.ax2_1], [self.ax3_0, self.ax3_1],
                    [self.ax4_0], [self.ax5_0], [self.ax6_0], [self.ax7_0],
                    [self.ax8_0], [self.ax9_0]]
        self.canvases = [self.canvas1, self.canvas2, self.canvas3, self.canvas4, self.canvas5, self.canvas6,
                         self.canvas7, self.canvas8, self.canvas9]
        self.threadings_test = None
        self.threadings_train = None

    def __draw1__(self, i, loss, acc, loss_pre, acc_pre, iteration, step):
        if iteration >= step:
            self.axs[i][0].plot([iteration - step, iteration], [loss_pre, loss], color='blue')
            self.axs[i][1].plot([iteration - step, iteration], [acc_pre, acc], color='red')
            self.canvases[i].draw()

    def __draw2__(self, i, loss, loss_pre, iteration):
        if iteration > 1:
            self.axs[i][0].plot([iteration - 1, iteration], [loss_pre, loss], color='orange')
            self.canvases[i].draw()

    def __draw7__(self, test_loss, test_loss_pre, iteration, step):
        if iteration >= step:
            self.axs[6][0].plot([iteration - step, iteration], [test_loss_pre, test_loss], color='blue')
            self.canvases[6].draw()

    def __draw8__(self, test_acc, test_acc_pre, iteration, step):
        if iteration >= step:
            self.axs[7][0].plot([iteration - step, iteration], [test_acc_pre, test_acc], color='red')
            self.canvases[7].draw()

    def __draw9__(self, train_loss, train_loss_pre, iteration):
        if iteration > 1:
            self.axs[8][0].plot([iteration - 1, iteration], [train_loss_pre, train_loss], color='orange')
            self.canvases[8].draw()

    def draw_th_7(self, test_loss, test_loss_pre, iteration, step):
        th7 = threading.Thread(target=self.__draw7__, args=(test_loss, test_loss_pre, iteration, step,))
        th7.start()

    def draw_th_8(self, test_acc, test_acc_pre, iteration, step):
        th8 = threading.Thread(target=self.__draw8__, args=(test_acc, test_acc_pre, iteration, step,))
        th8.start()

    def draw_th_9(self, train_loss, train_loss_pre, iteration):
        th9 = threading.Thread(target=self.__draw9__, args=(train_loss, train_loss_pre, iteration,))
        th9.start()

    def draw_th_test(self, i, loss, acc, loss_pre, acc_pre, iteration, step):
        th = threading.Thread(target=self.__draw1__, args=(i, loss, acc, loss_pre, acc_pre, iteration, step,))
        return th

    def draw_th_train(self, i, loss, loss_pre, iteration):
        th = threading.Thread(target=self.__draw2__, args=(i, loss, loss_pre, iteration,))
        return th

    def get_threadings_test(self, loss, acc, loss_pre, acc_pre, iteration, step, num_device):
        self.threadings_test = [self.draw_th_test(i, loss[i], acc[i], loss_pre[i], acc_pre[i], iteration, step) for i in
                                range(num_device)]

    def get_threadings_train(self, loss, loss_pre, iteration, num_device):
        self.threadings_train = [self.draw_th_train(i + 3, loss[i], loss_pre[i], iteration) for i in range(num_device)]

    def threading_test_start(self, num_device):
        for i in range(num_device):
            self.threadings_test[i].start()

    def threading_train_start(self, num_device):
        for i in range(num_device):
            self.threadings_train[i].start()

    def wait_for_end_th(self, num_device):
        for i in range(num_device):
            self.threadings_test[i].join()
        for i in range(num_device):
            self.threadings_train[i].join()


class Information:
    def __init__(self, root):
        self.right_control = tk.LabelFrame(root, text='Information Panel', font=('微软雅黑', 30, 'italic'))
        self.right_control.place(relx=0.75, rely=0.42, relwidth=0.25, relheight=0.55)

        tk.Label(self.right_control, text='Train Loss:', font=('微软雅黑', 15, 'italic')).place(relx=0, rely=0)
        self.frame_1 = tk.Frame(self.right_control)
        self.frame_1.place(relx=0, rely=0.07, relwidth=1, relheight=0.26)
        # 创建一个滚动条控件，默认为垂直方向
        sbar1 = tk.Scrollbar(self.frame_1)
        # 将滚动条放置在右侧，并设置当窗口大小改变时滚动条会沿着垂直方向延展
        sbar1.pack(side=tk.RIGHT, fill=tk.Y)
        # 创建水平滚动条，默认为水平方向,当拖动窗口时会沿着X轴方向填充
        sbar2 = tk.Scrollbar(self.frame_1, orient=tk.HORIZONTAL)
        sbar2.pack(side=tk.BOTTOM, fill=tk.X)
        self.mylist0 = tk.Listbox(self.frame_1, xscrollcommand=sbar2.set, yscrollcommand=sbar1.set)
        self.mylist0.place(relx=0, rely=0, relwidth=0.9, relheight=0.9)

        tk.Label(self.right_control, text='Test Loss:', font=('微软雅黑', 15, 'italic')).place(relx=0, rely=0.33)
        self.frame_2 = tk.Frame(self.right_control)
        self.frame_2.place(relx=0, rely=0.40, relwidth=1, relheight=0.26)
        sbar3 = tk.Scrollbar(self.frame_2)
        sbar3.pack(side=tk.RIGHT, fill=tk.Y)
        sbar4 = tk.Scrollbar(self.frame_2, orient=tk.HORIZONTAL)
        sbar4.pack(side=tk.BOTTOM, fill=tk.X)
        self.mylist1 = tk.Listbox(self.frame_2, xscrollcommand=sbar4.set, yscrollcommand=sbar3.set)
        self.mylist1.place(relx=0, rely=0, relwidth=0.9, relheight=0.9)

        tk.Label(self.right_control, text='Test Acc:', font=('微软雅黑', 15, 'italic')).place(relx=0, rely=0.66)
        self.frame_3 = tk.Frame(self.right_control)
        self.frame_3.place(relx=0, rely=0.73, relwidth=1, relheight=0.26)
        sbar5 = tk.Scrollbar(self.frame_3)
        sbar5.pack(side=tk.RIGHT, fill=tk.Y)
        sbar6 = tk.Scrollbar(self.frame_3, orient=tk.HORIZONTAL)
        sbar6.pack(side=tk.BOTTOM, fill=tk.X)
        self.mylist2 = tk.Listbox(self.frame_3, xscrollcommand=sbar6.set, yscrollcommand=sbar5.set)
        self.mylist2.place(relx=0, rely=0, relwidth=0.9, relheight=0.9)

    def __print_test_loss__(self, loss, loss_all, iteration, num_device):
        self.mylist1.insert(tk.END, '\niteration = {}:'.format(iteration))
        for i in range(num_device):
            self.mylist1.insert(tk.END, 'DEVICE ' + str(i) + ': {}'.format(loss[i]))
        self.mylist1.insert(tk.END, 'ALL : {}'.format(loss_all))

    def print_test_loss(self, loss, loss_all, iteration, num_device):
        th_temp = threading.Thread(target=self.__print_test_loss__, args=(loss, loss_all, iteration, num_device,))
        th_temp.start()
        return th_temp

    def __print_test_acc__(self, acc, acc_all, iteration, num_device):
        self.mylist2.insert(tk.END, '\niteration = {}:'.format(iteration))
        for i in range(num_device):
            self.mylist2.insert(tk.END, 'DEVICE ' + str(i) + ': {}'.format(acc[i]))
        self.mylist2.insert(tk.END, 'ALL : {}'.format(acc_all))

    def print_test_acc(self, acc, acc_all, iteration, num_device):
        th_temp = threading.Thread(target=self.__print_test_acc__, args=(acc, acc_all, iteration, num_device,))
        th_temp.start()
        return th_temp

    def __print_train_loss__(self, loss, loss_all, iteration, num_device):
        self.mylist0.insert(tk.END, '\niteration = {}:'.format(iteration))
        for i in range(num_device):
            self.mylist0.insert(tk.END, 'DEVICE ' + str(i) + ': {}'.format(loss[i]))
        self.mylist0.insert(tk.END, 'ALL : {}'.format(loss_all))

    def print_train_loss(self, loss, loss_all, iteration, num_device):
        th_temp = threading.Thread(target=self.__print_train_loss__, args=(loss, loss_all, iteration, num_device,))
        th_temp.start()
        return th_temp


if __name__ == '__main__':
    root = tk.Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    print(w, h)
    root.geometry(str(w) + 'x' + str(h))
    # root.resizable(width=False, height=False)
    root.title("split-learning test")
    c = FrameControl(root)
    # e = Information(root)
    # d = Frame_Pic(root)
    # d.draw_all()
    root.mainloop()
