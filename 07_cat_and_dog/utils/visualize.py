# -*- coding: utf-8 -*-
# @File  : visualize.py
# @Author: SmartGx
# @Date  : 19-3-8 上午11:20
# @Desc  :
import visdom
import time
import numpy as np

class Visualizer(object):
    def __init__(self, env='deafult', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # index表示画第几个点
        self.index = {}
        self.log_text = ''

    # 绘制曲线
    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append', **kwargs)
        self.index[name] = x + 1

    # log日志可视化
    def log(self, info, win='log_text'):
        self.log_text += ('[{time}] <br> {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info
        ))
        self.vis.text(self.log_text, win=win)

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    # 绘制多个图
    def plot_many(self, d):
        for k, v in d.item():
            self.plot(k, v)

    # 可视化图像
    def image(self, name, img_, **kwargs):
        self.vis.images(img_.cpu().numpy(), win=name, opts={'title': name}, **kwargs)

    def __getattr__(self, name):
        return getattr(self.vis, name)
