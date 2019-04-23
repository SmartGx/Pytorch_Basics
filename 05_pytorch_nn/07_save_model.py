# -*- coding: utf-8 -*-
# @File  : 07_save_model.py
# @Author: SmartGx
# @Date  : 19-1-25 下午3:38
# @Desc  : 保存模型
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# model = Net()
# optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
# # 输出模型的参数字典
# for param in model.state_dict():
#     print('{}: {}'.format(param, model.state_dict()[param].size()))
#
# for var_name in optimizer.state_dict():
#     print('{}: {}'.format(var_name, optimizer.state_dict()[var_name]))


# ****************保存/加载模型参数**********************
# torch.save(model.state_dict(), './model.pth')

# model = Net()
# model.load_state_dict(torch.load('./model.pth'))


# ****************直接保存/加载模型**********************
# torch.save(model, './model.pth')

# model = torch.load('./model.pth')
