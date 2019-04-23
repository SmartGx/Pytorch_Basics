# -*- coding: utf-8 -*-
# @File  : 06_transfer_learning.py
# @Author: SmartGx
# @Date  : 19-1-24 下午4:43
# @Desc  : torch迁移学习
import os
import time
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

plt.ion()

# *****************制作数据集****************
# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
data_dir = './data'
# 制作数据集
image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
# 数据batch加载器
dataLoader = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
# 训练集和测试集的样本数量
dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 训练集的类别名称
class_names = image_datasets['train'].classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('[INFO] Train set: {}, Test set: {}, ClassNames: {}'.format(dataset_size['train'], dataset_size['val'], class_names))



# *****************可视化图像********************
def show_imgs(input, title):
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = input * std + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# images, classes = next(iter(dataLoader['train']))
# img_grid = torchvision.utils.make_grid(images)
# show_imgs(img_grid, title=[class_names[i] for i in classes.numpy()])


# ******************创建训练模型*****************
def train_model(model, loss_fn, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("-"*30)
        print('[INFO] Epoch {}/{}'.format(epoch+1, num_epochs))

        for parse in ['train', 'val'] :
            if parse == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for images, classes in dataLoader[parse]:
                images = images.to(device)
                classes = classes.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(parse=='train'):
                    output = model(images)
                    _, pred_y = torch.max(output, 1)
                    loss = loss_fn(output, classes)

                    if parse == 'train':
                        loss.backward()
                        optimizer.step()

                # 累计loss
                running_loss += loss * images.size(0)
                # 统计预测正确的个数
                running_corrects += torch.sum(pred_y == classes)

            epoch_loss = running_loss / dataset_size[parse]
            epoch_acc = running_corrects.double() / dataset_size[parse]

            print('{} Loss: {:.4f}, Acc: {:.4f}%'.format(parse, epoch_loss, epoch_acc))

            # 保存最优模型参数
            if parse == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Training complete in s'.format(time.time() - since))
    print('Best val Acc: {:.4f}%'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataLoader['val']):
            images = inputs.to(device)
            labels = classes.to(device)

            output = model(images)
            _, pred = torch.max(output, 1)

            # 循环batch图像
            for j in range(num_images):
                images_so_far += 1
                ax = plt.subplot(2, 2, j+1)
                ax.axis('off')
                title = class_names[pred[j]]
                show_imgs(images.cpu()[j], title)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# 预训练模型
# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)
# model_ft.to(device)
# loss_fn = nn.CrossEntropyLoss()
#
# # 设置优化器和学习率调度器
# optimizer_ft = optim.SGD(params=model_ft.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer_ft, step_size=8, gamma=0.1)
#
# # 开启训练，返回最优模型
# model_ft = train_model(model_ft, loss_fn, optimizer_ft, exp_lr_scheduler, num_epochs=25)
# # 可视化测试结果
# visualize_model(model_ft)


# ***********************冻结参数*************************
model_conv = models.resnet18(pretrained=True)
# 将所有参数冻结
for param in model_conv.parameters():
    param.requires_grad = False
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(params=model_conv.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=8, gamma=0.1)

model = train_model(model_conv, loss_fn, optimizer_conv, exp_lr_scheduler, num_epochs=25)
visualize_model(model)
plt.ioff()
plt.show()