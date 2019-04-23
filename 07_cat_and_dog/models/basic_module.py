# -*- coding: utf-8 -*-
# @File  : basic_module.py
# @Author: SmartGx
# @Date  : 19-3-8 上午9:45
# @Desc  : 封装nn.Module模块，重写save和load
import torch.nn as nn
import time
import torch

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def save(self, name=None):
        if name is None:
            baseName = './checkpoints/' + self.model_name + '_'
            name = time.strftime(baseName + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))