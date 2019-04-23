# -*- coding: utf-8 -*-
# @File  : cifar_gpu.py
# @Author: SmartGx
# @Date  : 19-1-18 下午2:15
# @Desc  :
# -*- coding: utf-8 -*-
# @File  : cifar10.py
# @Author: SmartGx
# @Date  : 19-1-18 上午10:25
# @Desc  :
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
import torch.utils.data as Data

# 图像归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 下载Cifar10数据集，制作batch数据
trainSet = CIFAR10(root='./cifar', train=True, transform=transform, download=False)
testSet = CIFAR10(root='./cifar', train=False, transform=transform, download=False)
trainBatch = Data.DataLoader(trainSet, batch_size=32, shuffle=True, num_workers=2)
testBatch = Data.DataLoader(testSet, batch_size=32, shuffle=False, num_workers=2)

# CIFAR10包含的图像类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ----------------可视化训练图像----------------------
# import numpy as np
# import matplotlib.pyplot as plt
#
# def imshow_img(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # 创建迭代器
# dataiter = iter(trainBatch)
# # 生成一个batch的图像
# (images, labels) = dataiter.next()
#
# imshow_img(torchvision.utils.make_grid(images))
# print([classes[labels[i]] for i in range(8)])

 # -----------------------------------------------

# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2 ,2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4*4*128, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 4*4*128)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        return x

# 使用GPU驱动
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net()
net.to(device)
optimizer = optim.Adam(params=net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    train_loss = 0.0
    for i, (images, labels) in enumerate(trainBatch):
        images, labels = images.to(device), labels.to(device)
        output = net(images)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('[INFO] loss={}'.format(loss))
        train_loss += loss
        if i % 200 == 0:
            print('[INFO] Epoch={}, steps={}, loss={:.5f}'.format(epoch+1, i, train_loss/200))
            train_loss = 0.0

            # 计算测试集准确率
            accuracy = 0.0
            for j, (testImages, testLabels) in enumerate(testBatch):
                testImages, testLabels = testImages.to(device), testLabels.to(device)
                out = net(testImages)
                predictions = torch.argmax(F.softmax(out, dim=1), 1)
                correct = sum(predictions.cpu().numpy() == testLabels.cpu().numpy())
                batch_acc = correct / len(testLabels.cpu().numpy())
                accuracy += batch_acc
            print('[INFO] Accuracy={:.2f}%'.format(accuracy/j*100))


