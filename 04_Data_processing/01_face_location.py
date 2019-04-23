# -*- coding: utf-8 -*-
# @File  : 01_face_location.py
# @Author: SmartGx
# @Date  : 19-1-20 下午4:14
# @Desc  : pandas处理数据
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

import warnings
warnings.filterwarnings("ignore")
plt.ion()


def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.1)


# 加载标定点数据
lanmarks_frame = pd.read_csv('./faces/face_landmarks.csv')

n = 65
img_name = lanmarks_frame.iloc[n, 0]
landmarks = lanmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

# print('Image name:{}'.format(img_name))
# print('Landmark shape: {}'.format(landmarks.shape))
# print('Landmark: {}'.format(landmarks))

imagePath = os.path.join('./faces', img_name)
show_landmarks(io.imread(imagePath), landmarks)
plt.show()


