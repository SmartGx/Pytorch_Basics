# -*- coding: utf-8 -*-
# @File  : datasets.py
# @Author: SmartGx
# @Date  : 19-3-8 上午10:14
# @Desc  : 创建pytorch数据集
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Cat_Dog(Dataset):

    def __init__(self, root, train=True, test=False, transform=True):
        super(Cat_Dog, self).__init__()

        self.test = test
        images = [os.path.join(root, name) for name in  os.listdir(root)]
        images_num = len(images)

        # 将数据集图像按照数字排序
        if self.test:
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        else:
            images = sorted(images, key=lambda x: int(x.split('.')[-2]))

        # 划分训练/验证/测试集
        if self.test:
            self.images = images
        elif train:
            self.images = images[: int(0.7*images_num)]
        else:
            self.images = images[int(0.7*images_num): ]

        if transform:
            normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            if train:
                self.transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        if self.test:
            label = img_path.split(os.path.sep)[-1].split('.')[0]
        else:
            label = 1 if 'dog' in img_path.split(os.path.sep)[-1] else 0
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)

        return data, label


if __name__ == '__main__':
    trainset = Cat_Dog(root='./train', train=True, test=False)
    valset = Cat_Dog(root='./train', train=False, test=False)
    testset = Cat_Dog(root='./test1', train=False, test=True)
    print('TrainSet: {}\nValSet: {}\nTestSet: {}'.format(len(trainset), len(valset), len(testset)))