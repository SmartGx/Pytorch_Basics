# -*- coding: utf-8 -*-
# @File  : squeezenet.py
# @Author: SmartGx
# @Date  : 19-3-8 上午9:43
# @Desc  : 搭建squeezeNet网络模型
import torch
import torch.nn as nn
import torch.optim as optim
from models.basic_module import BasicModule
from torchvision.models.squeezenet import squeezenet1_1

class SqueezeNet(BasicModule):
    def __init__(self, num_classes=2):
        super(SqueezeNet, self).__init__()
        self.model_name = 'squeezenet'
        self.num_classes = num_classes
        self.model = squeezenet1_1(pretrained=True)
        self.model.num_classes = num_classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.num_classes, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

    def forward(self, x):
        return self.model(x)

    def get_optim(self, lr, weight_decay):
        return optim.Adam(self.model.classifier.parameters(), lr=lr, weight_decay=weight_decay)

if __name__ == '__main__':
    # 测试网络模型
    squeezenet = SqueezeNet()
    print(squeezenet)

    data = torch.randn(1, 3, 224, 224)
    output = squeezenet(data)
    print(output)