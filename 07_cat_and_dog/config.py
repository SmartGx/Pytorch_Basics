# -*- coding: utf-8 -*-
# @File  : config.py
# @Author: SmartGx
# @Date  : 19-3-8 上午11:08
# @Desc  :

class Default_Config():
    env = 'default'
    vis_port = 8097
    model_name = 'squeezenet'

    train_path = 'data/train'
    test_path = 'data/test1'
    checkpoints = './checkpoints'
    result_file = 'csv/results.csv'

    load_pre_model = None
    batch_size = 32
    max_epoches = 10
    print_freq = 20
    num_worker = 4

    lr = 0.001
    lr_decay = 0.5
    weights_decay = 0.0005

opt = Default_Config()