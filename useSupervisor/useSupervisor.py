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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

np.random.seed(10)
input_data = np.random.randn(1000,10)
output_data = np.random.uniform(0,10, size=[1000])
flags = tf.flags
flags.DEFINE_string("batch_size", 20, "the batch size")
FLAGS = flags.FLAGS
def produce_data():
    index = tf.train.range_input_producer(1000//FLAGS.batch_size).dequeue()
    return index

def produce_batch_data(index):
    return tf.strided_slice(input_data, [index*FLAGS.batch_size, 0], [(index+1)*FLAGS.batch_size, 10]), \
           tf.strided_slice(output_data, [index*FLAGS.batch_size], [(index+1)*FLAGS.batch_size])

def train_data(input, target):
    w = tf.get_variable("weight", [10,1], dtype=tf.float64)
    b = tf.get_variable("bias", [FLAGS.batch_size], dtype=tf.float64)
    output = tf.matmul(input, w) + b
    loss = output-target
    loss = loss*loss
    loss = tf.reduce_sum(loss)
    train_var = tf.trainable_variables()
    grads = tf.gradients(loss, train_var)
    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train_op = optimizer.apply_gradients(zip(grads, train_var))
    return loss, train_op

def useSupervisorTrain():
    with tf.Graph().as_default():
        init = tf.global_variables_initializer()
        index = produce_data()
        input_data, target = produce_batch_data(index)

        initializer = tf.random_uniform_initializer(-0.1,
                                                    0.1)
        tf.set_random_seed(1)
        with tf.variable_scope("Valid", initializer=initializer):
            loss, train_op = train_data(input_data, target)
        sv = tf.train.Supervisor(logdir='mydir')
        with sv.managed_session() as sess:
            sess.run(init)
            while not sv.should_stop():
                loss_, _ = sess.run((loss,train_op))
                print(loss_)


if __name__ == "__main__":
    # with tf.Graph().as_default():
    #     sess = tf.Session()
    #     init = tf.global_variables_initializer()
    #     index = produce_data()
    #     index, output = produce_batch_data(index)
    #     sess.run(init)
    #     tf.train.start_queue_runners(sess=sess)
    #     id, out = sess.run((index, output))
    #     print(id.shape)
    #     id, out = sess.run((index, output))
    #     print(id.shape)
    #     print(out.shape)
    useSupervisorTrain()