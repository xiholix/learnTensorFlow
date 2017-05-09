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
@file: testShuffleBatch.py
@time: 17-3-29 下午10:08
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from readingData import *

def test_shuffle_batch():
    fileNames = ['file1.csv']
    fileNamesQueue = tf.train.string_input_producer(fileNames)
    default_matrix = [[0], [0], [0], [0]]
    reader = tf.TextLineReader()
    key, values = reader.read(fileNamesQueue)
    col1, col2, col3, col4 = tf.decode_csv(values, record_defaults=default_matrix)
    vec =  tf.stack([col1, col2, col3, col4])  # col1返回的仅仅是一个标量
    vec_batch = tf.train.shuffle_batch([vec], batch_size=5, capacity=20, min_after_dequeue=7)
    return vec_batch


def test():
    node = test_shuffle_batch()
    test_run(node)


if __name__ == "__main__":
    test()