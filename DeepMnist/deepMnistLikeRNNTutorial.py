# -*-coding:utf8-*-
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
#
#
'''
@version: ??
@author: xiholix
@contact: x123872842@163.com
@software: PyCharm
@file: deepMnistLikeRNNTutorial.py
@time: 17-3-27 下午9:07
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def get_weights(_shape, _name):
    return tf.Variable(tf.truncated_normal(shape=_shape, stddev=0.1, mean=0, name=_name))


def get_bias(_shape, _name):
    return tf.Variable(tf.constant(value=0.1, shape=_shape, name=_name))


def conv2d(_input, _filter):
    return tf.nn.conv2d(input=_input, filter=_filter, strides=[1,1,1,1], padding="SAME")


def max_pool(_input):
    return tf.nn.max_pool(_input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


def build_graphic():
    x_ = tf.placeholder(tf.float32, shape=(None, 784), name='input_x')
    y_ = tf.placeholder(tf.float32, shape=(None, 10), name='input_y')

    x = tf.reshape(x_, shape=[-1, 28, 28, 1])
    with tf.name_scope("conv1"):
        w1 = get_weights(_shape=[5,5,1,32], _name="weight")
        b1 = get_bias([32], _name="bias")
        layer_1_out = tf.nn.relu(conv2d(_input=x, _filter=w1)+b1)
        layer_2_in = max_pool(layer_1_out)

    with tf.name_scope("conv2"):
        w2 = get_weights(_shape=[5,5,32,64], _name="weigth")
        b2 = get_bias([64], 'bias')
        layer_2_out = tf.nn.relu(conv2d(layer_2_in, w2)+b2)
        layer_3_in = max_pool(layer_2_out)

    with tf.name_scope("fullConnect"):
        w3 = get_weights(_shape=[7*7*64, 10], _name="weigth")
        b3 = get_bias(_shape=[10], _name="bias")
        layer_3_in = tf.reshape(layer_3_in, shape=[-1, 7*7*64])
        out = tf.matmul(layer_3_in, w3)+b3

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))
    accuracy = tf.reduce_mean( tf.cast(tf.equal(tf.arg_max(out, 1), tf.arg_max(y_, dimension=1)), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(1e-4)
    train_v = tf.trainable_variables()
    train_op = optimizer.minimize(cross_entropy)

    return x_, y_, accuracy, train_op


def get_data(x, y, dataset, batch_size):
    x_, y_ = dataset.next_batch(batch_size)
    # print (x_.shape)
    feed_dic = {
        x: x_,
        y: y_
    }
    return  feed_dic


def testModel(sess, accuracy, x, y, dataset, batch_size):
    lengths = len(dataset.images)

    epoch = lengths // batch_size
    accs = 0
    # print(epoch)
    for i in xrange(epoch):
        feed_dict = get_data(x, y, dataset, batch_size)
        acc = sess.run(accuracy, feed_dict=feed_dict)
        accs += acc
        # print(i)
    print("accuracy is %f"%(accs/epoch) )

def run_graphic():
    x, y, accuracy, train_op = build_graphic()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


    for i in xrange(10000):
        feed_dict = get_data(x, y, mnist.train, 50)
        _, acc = sess.run((train_op, accuracy), feed_dict=feed_dict)
        # print (acc)
        if i%100 == 0:
            testModel(sess, accuracy, x, y, mnist.validation, 50)




if __name__ == "__main__":
    pass