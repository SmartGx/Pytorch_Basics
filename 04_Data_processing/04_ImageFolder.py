# -*- coding: utf-8 -*-
# @File  : 04_ImageFolder.py
# @Author: SmartGx
# @Date  : 19-1-21 下午5:20
# @Desc  :
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 图像转换对象
data_tranforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

# 将文件夹中所有图像进行transform,文件名为标签
hymenoptera_dataset = datasets.ImageFolder(root='./root', transform=data_tranforms)
dataset_loader = DataLoader(hymenoptera_dataset, batch_size=4, shuffle=True, num_workers=4)

for i, image_batch in enumerate(dataset_loader):
    # image_batch的shape = [[4, 3, 224, 224], [4]]
    batch_size = len(image_batch[1])
    fig = plt.figure()

    for j in range(batch_size):
        ax = plt.subplot(1, 4, j+1)
        ax.set_title('{}'.format(image_batch[1][j]))
        image = image_batch[0][j].numpy()
        image = image.transpose(1, 2, 0)
        plt.imshow(image)

    plt.show()
    if i == 0:
        break




