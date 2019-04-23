# -*- coding: utf-8 -*-
# @File  : mnist.py
# @Author: SmartGx
# @Date  : 18-12-16 下午1:03
# @Desc  : 使用CNN预测MNIST
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

EPOCH = 1
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
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.show()

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
train_loader = Data.DataLoader(dataset=train_data, shuffle=True, batch_size=BATCH_SIZE)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda() / 255.0
test_y = test_data.test_labels[:2000].cuda()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(128*7*7, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view((x.size()[0], -1))
        x = self.fc1(x)
        x = self.out(x)
        return x

cnn = CNN()
cnn.cuda()
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()
# print(cnn)
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        train_x, train_y = batch_x.cuda(), batch_y.cuda()
        out = cnn(train_x)
        loss = loss_func(out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_output = cnn(test_x)
            prediction = torch.argmax(F.softmax(test_output), 1).cuda()
            # 在转化cuda tensor转化为numpy时，需要先转化为cpu tensor，即cuda_tensor.cpu().numpy()
            correct_num = sum(prediction.cpu().numpy()==test_y.cpu().numpy())
            acc = correct_num / 2000 * 100
            print('[INFO] Loss = {:.4f}, Test accuracy = {:.4f}%'.format(loss, acc))
