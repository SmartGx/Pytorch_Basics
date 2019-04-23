# -*- coding: utf-8 -*-
# @File  : practice.py
# @Author: SmartGx
# @Date  : 19-2-21 上午9:02
# @Desc  :
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

BATCH_SIZE = 50
EPOCHS = 10
LR = 1e-3

train_set = datasets.MNIST(root='./mnist',
                           transform=transforms.ToTensor(),
                           download=False,
                           train=True)
test_set = datasets.MNIST(root='./mnist',
                          transform=transforms.ToTensor(),
                          download=False,
                          train=False)
trainLoader = Data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# image = train_set.train_data.numpy()[0]
# plt.imshow(image, cmap='gray')
# plt.show()
test_data = torch.unsqueeze(test_set.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.0
test_labels = test_set.test_labels[:2000]

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(7*7*128, 1024),
            nn.ReLU(),
        )
        self.out = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view((x.shape[0], -1))
        x = self.fc1(x)
        x = self.out(x)

        return x

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cnn = Cnn()
optimizer = optim.SGD(params=cnn.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for step, (images, labels) in enumerate(trainLoader):
        pred = cnn(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_pred = cnn(test_data)
            label_pred = torch.argmax(F.softmax(test_pred), 1)
            correct_num = sum(label_pred.numpy() == test_labels.numpy())
            accuracy = correct_num / 2000 * 100
            print('[INFO] Loss = {:.4f}, Acc = {:.4f}%'.format(loss, accuracy))

