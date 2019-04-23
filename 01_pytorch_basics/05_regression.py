# -*- coding: utf-8 -*-
# @File  : regression.py
# @Author: SmartGx
# @Date  : 18-12-16 上午10:01
# @Desc  : 回归
import torch
import matplotlib.pyplot as plt

# 创建数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.1*torch.rand(x.size())

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.ion()
# plt.show()

# 定义神经网络
class Net(torch.nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        # 继承Module类
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(in_dims, hidden_dims)
        self.activation = torch.nn.ReLU()
        self.predict = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.predict(x)
        return x

net = Net(in_dims=1, hidden_dims=10, out_dims=1)
# print(net)
# 定义优化器
optimizer = torch.optim.SGD(params=net.parameters(), lr=0.2)
# 定义均方差损失函数
loss_func = torch.nn.MSELoss()

# 循环100step
for i in range(200):
    # 将数据feed到net中预测输出值
    prediction = net(x)
    # 计算loss值
    loss = loss_func(prediction, y)
    # 清空上一步残余更新参数值
    optimizer.zero_grad()
    # 误差反向传播，计算参数更新值
    loss.backward()
    # 将参数更新值施加到net的参数上
    optimizer.step()

    # 动态显示收敛过程
    if i%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-')
        plt.text(0.5, 0, 'Loss={:.4f}'.format(loss.data.numpy()), fontdict={'size': 20, 'color': 'blue'})
        plt.pause(0.1)