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
@file: readingData.py
@time: 17-3-28 下午10:02
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf

def test_string_input_producer():
    fileNames = [('file%d'%i) for i in range(10)]
    fileNamesNode = tf.train.string_input_producer(fileNames).dequeue()
    return fileNamesNode


def test_match_filenames_once():
    '''
    测试match_filenames_once失败
    :return:
    '''
    fileNamePatterns = tf.train.match_filenames_once('file')
    fileNamesNode = tf.train.string_input_producer(fileNamePatterns).dequeue()
    return fileNamesNode


def test_read_from_csv():
    fileNames = ['file1.csv']
    fileNamesQueue = tf.train.string_input_producer(fileNames)
    default_matrix = [[0],[0],[0],[0]]
    reader = tf.TextLineReader()
    key, values = reader.read(fileNamesQueue)
    col1, col2, col3, col4 = tf.decode_csv(values, record_defaults=default_matrix)
    return tf.stack([col1,col2,col3, col4])    #col1返回的仅仅是一个标量


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
    node = test_read_from_csv()
    test_run(node)


if __name__ == "__main__":
    test()

