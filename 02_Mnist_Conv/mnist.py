# -*- coding: utf-8 -*-
# @File  : mnist.py
# @Author: SmartGx
# @Date  : 18-12-16 下午1:03
# @Desc  : 使用CNN预测MNIST
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# 下载MNIST数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),   # 将数据的值从(0, 255)归一化到(0, 1)
    download=DOWNLOAD_MNIST
)

test_data = torchvision.datasets.MNIST(root='./mnist',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.show()
train_loader = Data.DataLoader(dataset=train_data, shuffle=True, batch_size=BATCH_SIZE)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.0
# print(test_data.test_data.shape)
# print(torch.unsqueeze(test_data.test_data, dim=1).shape)
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*7*7, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view((x.size()[0], -1))
        x = self.fc1(x)
        x = self.out(x)
        return x

cnn = CNN()
optimizer = optim.Adam(params=cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
# print(cnn)
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        out = cnn(batch_x)
        loss = loss_func(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_output = cnn(test_x)
            prediction = torch.argmax(F.softmax(test_output), 1)
            # 在计算前务必将Tensor数据转化为numpy类型
            correct_num = sum(prediction.numpy()==test_y.numpy())
            acc = correct_num / 2000 * 100
            print('[INFO] Loss = {:.4f}, Test accuracy = {:.4f}%'.format(loss, acc))
