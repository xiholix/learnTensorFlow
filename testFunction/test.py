#-*-coding:utf8-*-
import tensorflow as tf
import numpy as np

def testGather():
    params = np.arange(24).reshape(4,3,2)
    sess = tf.Session()
    a = tf.gather(params, [[2,1],[1,2]]) #将第二的每个元素用第一tensor的第一维对应的数据替代
    with sess.as_default():
        t = sess.run(a)
        print t
    '''
    [[[[12 13]
   [14 15]
   [16 17]]

  [[ 6  7]
   [ 8  9]
   [10 11]]]


 [[[ 6  7]
   [ 8  9]
   [10 11]]

  [[12 13]
   [14 15]
   [16 17]]]]
    '''

def testnnwithembedding_lookup():
    a = np.arange(24).reshape(4,3,2)
    b = np.arange(25,49).reshape(4,3,2)
    t = tf.nn.embedding_lookup([a,b], [0,2,4,6], partition_strategy='mod')
    #通过strategy将书ids中的id映射到前面列表中不同tensor的索引上，如本题用mod策略，则将0,2,4,6映射到
    #a的0,1,2,3索引上
    sess = tf.Session()
    with sess.as_default():
        f = sess.run(t)
        print f  #输出为a


if __name__ == "__main__":
    testnnwithembedding_lookup()