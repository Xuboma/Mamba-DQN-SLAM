# coding=UTF-8
"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import torch

"""定义三个参数的取值"""
import numpy as np
import time
import sys
import tkinter as tk
from Utils.evaluate_episodes import get_state_and_reward

UNIT = 40   # pixels  像素
# 对照片进行重构
MAZE_H = 64 # grid height  网格高度
MAZE_W = 64  # grid width   网格宽度
# class Maze(tk.Tk, object):
height = 64
width = 64

# 动作空间的定义
class Maze(object):    #继承父类object
    def __init__(self):
        super(Maze, self).__init__()   #对继承自父类的属性进行初始化
        para1 = list(range(110, 140)) #1.1-1.4
        para2 = [2, 3, 4, 5, 6, 7, 8] #2-8
        para3 = list(range(60, 90))  # 0.6-0.9

        # 参数空间共有6300个
        self.parameter_space = []
        for i in range(len(para1)):
            for j in range(len(para2)):
                for k in range(len(para3)):
                    self.parameter_space.append((para1[i]/100.0, para2[j], para3[k]/100.0))
                    #append() 方法用于在列表末尾添加新的对象。
        #print(self.parameter_space)
        ###self.action_space = ['u', 'd', 'l', 'r']
        self.action_space = []   #三个参数的取值空间
        for i in range(len(para1)*len(para2)*len(para3)):
            self.action_space.append(i)   #[0-10500][0-8399]
        # print(len(self.parameter_space))
        self.n_actions = len(self.action_space)
        self.height = MAZE_H
        self.width = MAZE_W
        self.n_features = self.height * self.width   #特征网格的大小
        parameters = (1.2, 8, 0.89)  # 这是你要查找的参数
        self.index = self.parameter_space.index(parameters)

        # self.title('maze')
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        # self._build_maze()

    def _build_maze(self):#初始化画布
        self.canvas = tk.Canvas(self, bg='white',  
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')                  #绘制黑色矩形框

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')              #绘制红色圆形

        # pack all
        self.canvas.pack()
    # #重置环境状态，回到初始环境，方便下一次观测
    def reset(self):
        observation = np.zeros(self.n_features)  #创建一个一维数组,大小为n_features
        return observation

    # def reset(self):
    #     observation = np.zeros(self.n_features)  # 创建一个一维数组,大小为n_features
    #     observation = torch.from_numpy(observation).view(-1)  # 将numpy数组转换为形状为(4096,)的PyTorch张量
    #     return observation

    # def reset(self):
    #     observation = np.zeros(self.n_features)  # 创建一个一维数组,大小为n_features
    #     observation_tensor = torch.from_numpy(observation).float()  # 将numpy数组转换成PyTorch张量
    #     observation_tensor = observation_tensor.view(4096)  # 改变张量的形状
    #     return observation_tensor

    #推进时间步长，返回下一状态和奖励
    def step(self, action, args, gt_dict):
        varint = vars(args)
        s_name, s_ob, reward, succ = get_state_and_reward(gt_dict, args)
        # s_ = np.zeros(self.n_features)
        # s_ = np.zeros(MAZE_H)
        parameter = self.parameter_space[action]

        # print(parameter)

        done = True
        return s_name, s_ob, reward, succ

    def render(self):
        # time.sleep(0.01)
        self.update()


