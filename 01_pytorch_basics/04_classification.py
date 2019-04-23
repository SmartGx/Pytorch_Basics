# -*- coding: utf-8 -*-
# @File  : classification.py
# @Author: SmartGx
# @Date  : 18-12-16 上午10:48
# @Desc  : 分类
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 创建分类数据
n_data = torch.ones(100, 2)
# 均值为2方差为1，shape=[100,2]的数据
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
# 均值为-2方差为1，shape=[100,2]的数据
x1 = torch.normal(-2*n_data ,1)
y1 = torch.ones(100)

# 合并正负样本数据
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)

# 显示散点图数据
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), cmap='RdYlGn')
# plt.show()

# 创建神经网络
class Net(torch.nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(in_dims, hidden_dims)
        self.activation = torch.nn.ReLU()
        self.predict = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.predict(x)
        return x
net = Net(2, 20, 2)

# ----method 2--------------
# net = torch.nn.Sequential(
#     torch.nn.Linear(2, 20),
#     torch.nn.ReLU(),
#     torch.nn.Linear(20, 2)
# )
# print(net)
optimizer = torch.optim.SGD(lr=0.2, params=net.parameters())
loss_func = torch.nn.CrossEntropyLoss()

for i in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=pred_y, cmap='RdYlGn')
        accuracy = sum(pred_y==target_y) / len(target_y)
        plt.text(-4, 4, 'Accuracy: {:.2f}%'.format(accuracy*100), fontdict={'size': 18, 'color': 'red'})
        plt.pause(0.5)

plt.ioff()
plt.show()
