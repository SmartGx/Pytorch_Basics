# -*- coding: utf-8 -*-
# @File  : 2_visdom.py
# @Author: SmartGx
# @Date  : 19-3-7 下午2:43
# @Desc  :
import torch
import visdom

# 创建visdom客户端
vis = visdom.Visdom(env=u'test1', use_incoming_socket=False)

# 绘制y = sinx, 窗口名称为sinx
x = torch.arange(1, 30, 0.01)
y = torch.sin(x)
vis.line(Y=y, X=x, win='sinx', opts={'title': 'y=sinx'})

# 以append形式绘制直线
for ii in range(0, 10):
    x = torch.Tensor([ii])
    y = x
    vis.line(Y=y, X=x, win='line', update='append' if ii>0 else None)

# 更新轨迹，新增一条曲线
x = torch.arange(0, 9, 0.1)
y = x ** 2 / 9
vis.line(y, x, win='line', name='this is a new line', update='new')

# 可视化图像
# vis.image(torch.randn(64, 64).numpy(), win='rand_img1')
# vis.image(torch.randn(3, 64, 64).numpy(), win='rand_img2')
# vis.images(torch.randn(8, 3, 64, 64).numpy(), nrow=4, win='randn_img3', opts={'title': 'vis_images'})

# 可视化文本
vis.text(u'''<h1>Hello Visdom</h1><br>Visdom是Facebook专门为<b>PyTorch</b>开发的一个可视化工具，
         在内部使用了很久，在2017年3月份开源了它。

         Visdom十分轻量级，但是却有十分强大的功能，支持几乎所有的科学运算可视化任务''',
         win='visdom',
         opts={'title': u'visdom简介'}
         )