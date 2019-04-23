# -*- coding: utf-8 -*-
# @File  : test_imgs.py
# @Author: SmartGx
# @Date  : 19-3-8 下午5:17
# @Desc  : 测试集
import torch
import csv
from config import opt
import torch.nn.functional as F
from data.datasets import Cat_Dog
from torch.utils.data import DataLoader
from models.squeezenet import SqueezeNet

def writer_csv(results, path):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

def main():
    checkpoints_path = './checkpoints/squeezenet_0308_17:19:16.pth'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 加载训练好的模型
    model = SqueezeNet()
    model.eval()
    model.load(checkpoints_path)
    model.to(device)

    # 加载数据集
    testSet = Cat_Dog('./data/test1', train=False, test=True)
    test_dataloader = DataLoader(testSet, batch_size=opt.batch_size, num_workers=opt.num_worker)
    results = []

    # 循环测试集数据
    for i, (images, path) in enumerate(test_dataloader):
        images = images.to(device)

        # 预测结果
        output = model(images)
        predictions = torch.argmax(F.softmax(output), 1).tolist()

        batch_results = [(path, pred) for (path, pred) in zip(path, predictions)]
        results += batch_results

    # 将结果写入CSV文件
    writer_csv(results, opt.result_file)


