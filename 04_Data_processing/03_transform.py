# -*- coding: utf-8 -*-
# @File  : 03_transform.py
# @Author: SmartGx
# @Date  : 19-1-21 下午2:13
# @Desc  : 对图像进行预处理变换
import os
import pandas as pd
import numpy as np
import warnings
import torch
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms

warnings.filterwarnings('ignore')

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.1)

# 尺度变换
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        # 判断传入的是单个size还是元组
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * h / w
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        # resize image
        image = transform.resize(image, (new_h, new_w))
        # resize landmarks
        landmarks = landmarks * [new_w/w, new_h/h]

        return {'image': image, 'landmarks': landmarks}

# 随机裁剪
class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # resize image
        image = image[top: top+new_h, left: left+new_w]
        # resize landmarks
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

# 转化为Tensor
class ToTensor(object):
    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        # 将图像转化为first_channel形式
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

# 创建数据集
class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmark_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmark_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmark_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmark_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

# 单个sample变换
def demo_1():
    face_dataset = FaceLandmarksDataset(csv_file='./faces/face_landmarks.csv', root_dir='./faces')

    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([scale, crop])

    fig = plt.figure()
    sample = face_dataset[65]

    for i, trans in enumerate([scale, crop, composed]):
        transformed_sample = trans(sample)
        ax = plt.subplot(1, 3, i+1)
        plt.tight_layout()
        ax.set_title(type(trans).__name__)
        # **表示关键字参数，接收的是一个dict
        show_landmarks(**transformed_sample)
    plt.show()

# 数据集批量转换
def demo_2():
    transformed_dataset = FaceLandmarksDataset(csv_file='./faces/face_landmarks.csv',
                                               root_dir='./faces',
                                               transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print('#{}:, {}, {}'.format(i+1, sample['image'].size(), sample['landmarks'].size()))

        if i == 3:
            break


def demo_3():
    transformed_dataset = FaceLandmarksDataset(csv_file='./faces/face_landmarks.csv',
                                               root_dir='./faces',
                                               transform=transforms.Compose([Rescale(256), RandomCrop(128), ToTensor()]))
    # batch加载数据集
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    # 循环每个batch_size
    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch['image'].size(), sample_batch['landmarks'].size())
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

def show_landmarks_batch(sample_batched):
    image_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(image_batch)
    im_size = image_batch.size(2)
    # grid的size=[3, 132, 522]
    grid = make_grid(image_batch)
    plt.imshow(grid.numpy().transpose(1, 2, 0))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i*im_size,
                    landmarks_batch[i, :, 1].numpy(), s=10, marker='.', c='r')
        plt.title('Batch from dataloader')

if __name__ == '__main__':
    # demo_1()
    # demo_2()
    demo_3()