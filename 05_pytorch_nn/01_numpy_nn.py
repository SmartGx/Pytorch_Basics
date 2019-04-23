# -*- coding: utf-8 -*-
# @File  : 01_numpy_nn.py
# @Author: SmartGx
# @Date  : 19-1-21 下午8:00
# @Desc  : 使用numpy实现neural network
import numpy as np

N = 64
input_dim = 1000
output_dim = 10
hidden_dim = 100

# 创建x和y样本
x = np.random.randn(N, input_dim)
y = np.random.randn(N, output_dim)

# 随机初始化权重
w1 = np.random.randn(input_dim, hidden_dim)
w2 = np.random.randn(hidden_dim, output_dim)

# b1 = np.zeros(hidden_dim)
# b2 = np.zeros(output_dim)

learning_rate = 1e-6

for i in range(500):
    h = x.dot(w1)
    h_relu = np.maximum(0, h)
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred - y).sum()
    print('Loss = {}'.format(loss))

    # 计算grad
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 反向传播
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    # print('W1: {}, W2: {}'.format(w1, w2))