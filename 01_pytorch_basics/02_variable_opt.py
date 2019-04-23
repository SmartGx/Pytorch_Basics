# -*- coding: utf-8 -*-
# @File  : variable_opt.py
# @Author: SmartGx
# @Date  : 18-12-16 上午9:54
# @Desc  :torch 的Variable操作
import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2], [3, 4]])
var = Variable(tensor, requires_grad=True)

print('[Tensor]: ', tensor)
print('[Variable]: ', var)