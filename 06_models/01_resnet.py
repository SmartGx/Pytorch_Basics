# -*- coding: utf-8 -*-
# @File  : resnet.py
# @Author: SmartGx
# @Date  : 19-3-7 上午11:11
# @Desc  : pytorch-resnet
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, strides=1, shortcuts=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, strides, 1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_features)
        )
        self.right = shortcuts

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, 2)
        self.layer3 = self._make_layer(128, 256, 6, 2)
        self.layer4 = self._make_layer(256, 512, 3, 2)

        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


    def _make_layer(self, in_features, out_features, num_block, strides=1):
        layers = []
        shortcut = nn.Sequential(
            nn.Conv2d(in_features, out_features, 1, strides, bias=False),
            nn.BatchNorm2d(out_features)
        )
        layer = ResidualBlock(in_features, out_features, strides, shortcut)
        layers.append(layer)

        for i in range(1, num_block):
            layers.append(ResidualBlock(out_features, out_features))

        return nn.Sequential(*layers)

resnet34 = ResNet()
print(resnet34)

sample = torch.randn(1, 3, 224, 224)
output = resnet34(sample)
print(output)