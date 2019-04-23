# -*- coding: utf-8 -*-
# @File  : batch_train.py
# @Author: SmartGx
# @Date  : 18-12-16 下午12:00
# @Desc  : 生成batch数据
import torch
import torch.utils.data as Data

batch_size = 5

# 创建数据
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 创建torch数据库，用来生成batch数据
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(torch_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=2)

for epoch in range(2):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch: {}, Step: {}, batch_x: {}, batch_y: {}'.format(
            epoch, step, batch_x.numpy(), batch_y.numpy()
        ))