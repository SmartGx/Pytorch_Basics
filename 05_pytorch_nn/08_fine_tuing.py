# -*- coding: utf-8 -*-
# @File  : 08_fine_tuing.py
# @Author: SmartGx
# @Date  : 19-1-25 下午6:02
# @Desc  : pytorch进行模型fine-tuing
import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import datasets, transforms, models

data_dirs = './data'
model_name = 'squeezenet'
num_classes = 2
batch_size = 32
num_epoches = 15
feature_extract = True


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    start = time.time()
    # 复制模型参数
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_history = []

    # 循环每个epoch
    for epoch in range(num_epochs):
        print('-'*20)
        print('Epoch# {}/{}'.format(epoch+1, num_epochs))

        # 循环训练/验证
        for parse in ['train', 'val']:
            if parse == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 循环加载每个batch数据
            for inputs, labels in dataloaders[parse]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 重置梯度
                optimizer.zero_grad()

                # 前向预测，计算loss
                with torch.set_grad_enabled(parse=='train'):
                    if is_inception:
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, pred = torch.max(outputs, 1)

                    # 反向传播
                    if parse == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss * inputs.shape[0]
                running_corrects += torch.sum(pred == labels.data)

            # 每个Epoch的总损失
            epoch_loss = running_loss / len(dataloaders[parse].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[parse].dataset)

            if parse == 'train':
                print('Training Loss : {:.4f}, Training Acc : {:.4f}'.format(epoch_loss, epoch_acc))
            elif parse == 'val':
                print('Evaling Loss : {:.4f}, Evaling Acc : {:.4f}'.format(epoch_loss, epoch_acc))
            else:
                print('Error!!')

            if parse == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
            if parse == 'val':
                val_history.append(epoch_acc)

    end = time.time()
    print('[INFO] Training take {:.2f}s'.format(end - start))
    print('[INFO] Best val acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_weights)
    return model, val_history


# 只用作特征提取层是，冻结参数
def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# 根据不同的模型，修改最后一层的输出维度
def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    if model_name == 'Alexnet':
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_channels = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_channels, num_classes)
        input_size = 224

    elif model_name == 'Vgg16':
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_channels = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_channels, num_classes)
        input_size = 224

    elif model_name == 'ResNet':
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_channels = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_channels, num_classes)
        input_size = 224

    elif model_name == 'SqueezeNet':
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_channels = model_ft.classifier[1].in_channels
        model_ft.classifier[1] = nn.Conv2d(num_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
        input_size = 224

    elif model_name == 'DenseNet':
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_channels = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_channels, num_classes)
        input_size = 224

    # Note: inception的输入尺寸为(299, 299), 并且有额外输出通道AuxLogits
    elif model_name == 'Inception':
        model_ft = models.inception_v3(use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_channels_1 = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_channels_1, num_classes)
        num_channels_2 = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_channels_2, num_classes)
        input_size = 299

    else:
        print('[INFO] Invalid model name, exiting...')
        exit()

    return model_ft, input_size

# 加载预训练模型
model, input_size = initialize_model('ResNet', 2)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.225], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.225], [0.229, 0.224, 0.225])
    ])}


print('[INFO] Initialing Datasets and Dataloads...')
# 创建数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dirs, x), data_transforms[x]) for x in ['train', 'val']}
# 创建数据生成器
data_loaders = {x: Data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载模型到CUDA
model = model.to(device)
# 需要更新的模型参数
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print('\t', name)

# 优化器和损失函数
optimizer = optim.SGD(params=params_to_update, lr=1e-3, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# 训练
model_ft, hist = train_model(model, data_loaders, criterion, optimizer)

