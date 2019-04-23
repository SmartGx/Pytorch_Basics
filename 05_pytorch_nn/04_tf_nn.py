# -*- coding: utf-8 -*-
# @File  : 04_tf_nn.py
# @Author: SmartGx
# @Date  : 19-1-23 下午7:14
# @Desc  : 使用tensorflow实现neural network
import tensorflow as tf
import numpy as np

N = 64
input_dim = 1000
hidden_dim = 100
output_dim = 10

x = tf.placeholder(tf.float32, shape=(None, input_dim))
y = tf.placeholder(tf.float32, shape=(None, output_dim))

w1 = tf.Variable(tf.random_normal(shape=(input_dim, hidden_dim)))
w2 = tf.Variable(tf.random_normal(shape=(hidden_dim, output_dim)))

h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
pre_y = tf.matmul(h_relu, w2)

loss = tf.reduce_sum((pre_y - y) ** 2.0)

learning_rate = 1e-6
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

new_w1 = w1.assign(w1 - learning_rate*grad_w1)
new_w2 = w2.assign(w2 - learning_rate*grad_w2)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(500):
        input = np.random.randn(N, input_dim)
        output = np.random.rand(N, output_dim)

        loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={x: input, y: output})
        print('[INFO] Loss = {:.5f}'.format(loss_value))
