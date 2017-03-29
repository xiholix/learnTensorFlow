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
@file: writeTFRecord.py
@time: 17-3-29 下午8:51
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

def test_write_record():
    path = 'data'
    data_sets = mnist.read_data_sets(path,
                                     dtype=tf.uint8,
                                     reshape=False,
                                     validation_size=5000)
    images = data_sets.train.images
    labels = data_sets.train.labels
    num_examples = data_sets.train.num_examples

    filepath = 'train.tfrecorder'
    writer = tf.python_io.TFRecordWriter(filepath)

    for index in range(num_examples):
        raw_image = images[index].tostring()
        example = tf.train.Example( features=tf.train.Features(feature={   #此处的命名参数为feature不是features
            'raw_image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_image])),
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[int(labels[index])])),
        }))
        writer.write(example.SerializeToString())   #此处一定要记得使用SerializeToString()
    writer.close()

if __name__ == "__main__":
    test_write_record()