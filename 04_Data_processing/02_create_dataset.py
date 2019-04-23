# -*- coding: utf-8 -*-
# @File  : 02_create_dataset.py
# @Author: SmartGx
# @Date  : 19-1-20 下午5:07
# @Desc  : 继承torch的Dataset类，创建自己的数据集
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.1)

class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.landmark_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmark_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmark_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmark_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        return sample

# 创建数据库实例
face_dataset = FaceLandmarksDataset(csv_file='./faces/face_landmarks.csv', root_dir='./faces')

fig = plt.figure()
for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i+1)
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break