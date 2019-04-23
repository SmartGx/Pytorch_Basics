# -*- coding: utf-8 -*-
# @File  : 02_torch_nn.py
# @Author: SmartGx
# @Date  : 19-1-21 下午10:17
# @Desc  : 使用torch的Tensor实现neural network
import torch

dtype = torch.float
device = torch.device('cpu')

N = 64
input_dim = 1000
output_dim = 10
hidden_dim = 100

x = torch.randn(N, input_dim, dtype=dtype, device=device)
y = torch.randn(N, output_dim, dtype=dtype, device=device)

w1 = torch.randn(input_dim, hidden_dim, dtype=dtype, device=device)
w2 = torch.randn(hidden_dim, output_dim, dtype=dtype, device=device)

learning_rate = 1e-6

for i in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print('Loss: {}'.format(loss))

    grad_pred = 2 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_pred)
    grad_h_relu = grad_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2