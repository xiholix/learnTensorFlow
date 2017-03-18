# --*--coding:utf8--*--
import tensorflow as tf
import numpy as np


def testBasicLstmCellOutput():
    size = 5
    cell = tf.contrib.rnn.BasicLSTMCell(
        10, forget_bias=0.0, state_is_tuple=True)
    input_data_ = np.arange(10*size).reshape(10,size)  #只能使用二维的数据
    print input_data_
    input_data = tf.Variable(input_data_, dtype=tf.float32)
    init_state = cell.zero_state(10, tf.float32)
    cell_output, state = cell(tf.identity(input_data), init_state) #不加identity就会报错

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    output_, state_ = sess.run((cell_output, state))
    print output_.shape
    # print state_
    print output_



if __name__ == "__main__":
    testBasicLstmCellOutput()