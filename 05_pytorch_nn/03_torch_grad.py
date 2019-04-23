# -*- coding: utf-8 -*-
# @File  : 03_torch_grad.py
# @Author: SmartGx
# @Date  : 19-1-21 下午10:32
# @Desc  : 使用torch的grad实现计算梯度
import torch

dtype = torch.float
device = torch.device('cpu')

N = 64
input_dim = 1000
output_dim = 10
hidden_dim = 100

x = torch.randn(N, input_dim, dtype=dtype, device=device)
y = torch.randn(N, output_dim, dtype=dtype, device=device)

w1 = torch.randn(input_dim, hidden_dim, dtype=dtype, device=device, requires_grad=True)
w2 = torch.randn(hidden_dim, output_dim, dtype=dtype, device=device, requires_grad=True)

learning_rate = 1e-6

for i in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print('Loss: {}'.format(loss))

    loss.backward()

    # torch.no_grad()的优先级比requires_grad=True高
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 更新权重后重置梯度
        w1.grad.zero_()
        w2.grad.zero_()