# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: SmartGx
# @Date  : 18-11-29 下午9:11
# @Desc  : what is pytorch
import torch
import numpy as np
from torch.autograd import Variable


# -------------------numpy数组和torch tensor的转化----------------
# 生成2x3的numpy矩阵
np_data = np.arange(6).reshape((2, 3))
# 将numpy矩阵转化为torch的tensor
torch_tensor = torch.from_numpy(np_data)
# 将torch的tensor转化为numpy array
tensor2numpy = torch_tensor.numpy()
# tensor转化为Variable
var = Variable(torch_tensor)
# Variable转化为numpy
numpy_arr = var.data.numpy()

# print(np_data)
# print(torch_tensor)
# print(tensor2numpy)
# print(var)

# -------------------基本运算---------------------------------
data = [-1, 2, -3, 4]
# 转化为torch的32为浮点型(默认操作的是32为float)
tensor = torch.FloatTensor(data)

# 四则运算
add = torch.add(tensor/0.5, tensor*2)
print('[+-*/]: ', add)

# 计算绝对值
abs_ = torch.abs(tensor)
print('[abs]: ', abs_)

# 计算均值
mean_ = torch.mean(tensor)
print('[mean]: ', mean_)

# 最大最小值
max_ = torch.max(tensor)
min_ = torch.min(tensor)
print('[max]: ', max_)
print('[min]: ', min_)

# 矩阵乘法
mat1 = torch.Tensor([1, 2, 3, 4, 5, 6]).reshape((2, 3))
mat2 = torch.Tensor([1, 2, 3, 4, 5, 6]).reshape((3, 2))
mutmal_ = torch.mm(mat1, mat2)
print('[matrix]: ', mutmal_)
