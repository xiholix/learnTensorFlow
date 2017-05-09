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
@file: readTFRecord.py
@time: 17-3-29 下午8:12
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist

def test_readTFRecord():
    # filePath = ['data/train.tfrecords']
    filePath = ['train.tfrecorder']
    # fileQueue = 'train.tfrecord'
    fileQueue = tf.train.string_input_producer(filePath)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fileQueue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'raw_image':tf.FixedLenFeature([], tf.string),     #读取train.tfrecorder是raw_image,如果是data下为image_raw
            'label':tf.FixedLenFeature([], tf.int64),
            # 'height': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    image.set_shape(mnist.IMAGE_PIXELS)
    image = tf.cast(image, tf.float32)*(1. /255)*0.5

    label = tf.cast(features['label'], tf.int32)
    return image, label

def test_run(_test_node):
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    s = sess.run(_test_node)
    print(s)
    s = sess.run(_test_node)
    print(s)


def test():
    # node = test_string_input_producer()
    # node = test_match_filenames_once()
    # node = test_read_from_csv()
    image, label = test_readTFRecord()
    test_run(image)
    pass


if __name__ == "__main__":
    test()