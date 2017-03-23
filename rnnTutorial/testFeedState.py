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
@file: testFeedState.py
@time: 17-3-19 下午7:40
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import  tensorflow as tf

class Model(object):
    def __init__(self):
        self.state = tf.Variable(1)

def test():
    m = Model()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(10):
        t = sess.run(m.state, feed_dict={m.state:i})
        print(t)
    s = sess.run(m.state)
    print(s)


if __name__ == "__main__":
    test()