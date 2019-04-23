# -*- coding: utf-8 -*-
# @File  : 05_pytorch_nn.py
# @Author: SmartGx
# @Date  : 19-1-23 下午7:37
# @Desc  : 使用pytorch的nn模块实现neural network
import torch

N = 64
input_dim = 1000
hidden_dim = 100
output_dim = 10

x = torch.randn(N, input_dim)
y = torch.randn(N, output_dim)

model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, output_dim)
)

learning_rate = 1e-4
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

for step in range(500):
    pre_y = model(x)
    loss = loss_fn(pre_y, y)

    print('[INFO] Loss = {:.5f}'.format(loss))

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
