# --*--coding:utf8--*--
from __future__  import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
def testIdentity():
    initial_value = np.arange(16).reshape((4, 4))
    print( initial_value)
    a = tf.Variable(initial_value=1, trainable=False)
    result = tf.identity(a)
    return a

def testRangeInputProducer():
    m = tf.train.range_input_producer(6,num_epochs=3  ,shuffle=False).dequeue()

    return m

def testStridedSlice():
    a = tf.Variable(np.arange(16).reshape((2,2,4)))
    b = tf.strided_slice(a, begin=[1,2], end=[2,4], begin_mask=3)
    '''结果为[[0 1 2 3] [4 5 6 7] ]'''
    c = tf.strided_slice(a, begin=[0,0], end=[2,2], new_axis_mask=2)
    '''结果应该是为一的位的begin, end失效,自己暂时的理解是因为在该位置上插入了一维，所以在该维总共只有一维，
    所以它的begin，end，stride失效'''
    d = tf.strided_slice(a, begin=[0,1,0], end=[2,2,2], shrink_axis_mask=2)
    '''暂时理解为这些操作都是先在数据上变换维度，然后再在变换过后的数据上进行操作。shrink_axis_mask会将对应的维度消去，并且暂时看到
    的结果是对应维度上的数据只会保留index=begin的结果，且官网上写了要对应的slice[i]=1，才是有效的'''
    return d

def testUnstack():
    ''''unstack方法是沿着指定的axis将values分解成该维对应的数目的array，num参数必须等于axis对应的维度的长度
    '''
    a = np.arange(16).reshape(4,2,2)
    b = tf.unstack(a, num=4)
    return b

def testQueue():
    q = tf.FIFOQueue(3, "float")
    init = q.enqueue_many(([0., 0., 0.],))
    x = q.dequeue()
    y = x + 1
    q_inc = q.enqueue([y])
    return q_inc

def testSetRandom():
    a = tf.random_normal()


def testVariableScope():
    with tf.variable_scope('first') as scope:
        a = tf.get_variable('a', [1], initializer=tf.constant_initializer(1))
        # tf.get_variable_scope().reuse_variables()
        print('assert')
        assert(a.name == 'first/a:0')
        print("end assert")
    with tf.variable_scope('second') as scope2:
        with tf.variable_scope(scope):
            x = a + 2
            assert x.op.name=='second/first/add'
            b = tf.get_variable('b', [1])
            assert b.name=='first/b:0'

    with tf.variable_scope('second'):
        with tf.variable_scope('first'):
            x = a + 2
            # assert x.op.name == 'second/add'
            print(x.op.name) # second_1/first/add
            b = tf.get_variable('b', [1])
            assert b.name == 'second/first/b:0'
    with tf.variable_scope('second'):
        with tf.variable_scope('first', reuse=True):
            x = b + 2
            # assert x.op.name == 'second/add'
            print(x.op.name) # second_2/first/add
            # b = tf.get_variable('b', [1])
            assert b.name == 'second/first/b:0'

    return a

def testSquenceLoss():
    np.random.seed(6)
    a = np.random.random((8,5))
    b = np.random.random_integers(0,4, 8)
    a = tf.Variable(a, tf.float32)
    b = tf.Variable(b)
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([a],[b], [3*tf.ones(8, dtype=tf.float64)],
                                                              average_across_timesteps=False)
    # print(b)
    # print (a)
    return loss


def run_function():
    result = testRangeInputProducer()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    init2 = tf.initialize_local_variables()

    sess.run(init)
    sess.run(init2)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in xrange(100):
        res = sess.run(result)
        print(res)

    # res = sess.run(result)


def run_functions():
    result = testSquenceLoss()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    r = sess.run(result)
    print(r)

if __name__ == "__main__":
   # run_function()
   # testSquenceLoss()
   run_functions()