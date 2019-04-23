# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: SmartGx
# @Date  : 19-3-8 上午11:14
# @Desc  :
import torch

from config import opt
from torchnet import meter
from utils.visualize import Visualizer
from torch.utils.data import DataLoader
from data.datasets import Cat_Dog
from models.squeezenet import SqueezeNet

# 开启visdom服务
vis = Visualizer(env=opt.env, port=opt.vis_port)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 初始化模型，加载预训练权重
model = SqueezeNet()
if opt.load_pre_model:
    model.load(opt.load_pre_model)
model.to(device)

# 获取训练和验证数据
train_dataloader = DataLoader(Cat_Dog(root=opt.train_path, train=True),
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.num_worker)
val_dataloader = DataLoader(Cat_Dog(root=opt.train_path, train=False),
                      batch_size=opt.batch_size,
                      shuffle=False,
                      num_workers=opt.num_worker)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
lr = opt.lr
optimizer = model.get_optim(lr=lr, weight_decay=opt.weights_decay)

# 使用torchnet进行快速评价，平均loss和混淆矩阵
loss_meter = meter.AverageValueMeter()
confusion_matrix = meter.ConfusionMeter(2)
previous_loss = 1e10

# 开启训练
for epoch in range(opt.max_epoches):
    loss_meter.reset()
    confusion_matrix.reset()

    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # 更新统计值
        loss_meter.add(loss.item())
        confusion_matrix.add(output.detach(), labels.detach())

        # 可视化训练log
        if i % opt.print_freq == 0:
            vis.plot('loss', loss_meter.value()[0])

    # 训练epoch后，保存模型参数
    model.save()

    with torch.no_grad():
        # 开启val
        model.eval()
        val_confusion_matrix = meter.ConfusionMeter(2)
        for i, (images, labels) in enumerate(val_dataloader):
            val_images = images.to(device)
            val_labels = labels.to(device)
            val_output = model(val_images)
            val_confusion_matrix.add(val_output.detach().squeeze(), labels.detach().type(torch.LongTensor))

        # 统计val混淆矩阵，绘制准确率曲线
        cm_value = val_confusion_matrix.value()
        val_accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / cm_value.sum()

    vis.plot('val_accuracy', val_accuracy)
    # 记录log
    info = '#Epoch: {}, #Lr: {}, #Loss: {:.5f}, #Train_cm: {}, #Val_cm: {}'.format(
        epoch, lr, loss_meter.value()[0], confusion_matrix.value(), cm_value)
    vis.log(info)

    # 当前epoch的loss值高于上一次时，更新学习率
    if loss_meter.value()[0] > previous_loss:
        lr  = lr * opt.lr_decay
        # 第二种降低学习率的方法，不会造成moment等信息的丢失
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    previous_loss = loss_meter.value()[0]