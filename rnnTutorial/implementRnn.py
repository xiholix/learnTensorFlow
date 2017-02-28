# --*--coding:utf8--*--
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import reader

class DataInput():
    def __init__(self, raw_data, batch_size, num_step):
        self.input, self.target = reader.ptb_producer(raw_data, batch_size, num_step)
        self.batch_size =batch_size
        self.num_step = num_step

class Config():
    def __init__(self, batch_size, num_step, hidden_size):
        self.batch_size = batch_size
        self.num_steps = num_step
        self.hidden_size = hidden_size

class PTBModel():
    def __init__(self, input, config, is_trainig):
        self.input = input
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.hidden_size = config.hidden_size
        def rnn_layer():
            return tf.contrib.rnn.BasicLSTM(self.hidden_size,
                                            forget_bias=0.0,
                                            state_is_tuple=True)

def testReadData():
    path = 'data/'
    datas = reader.ptb_raw_data(path)
    train_data, valid_data, test_data, _ = datas
    data_input = DataInput(train_data, 5, 2)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    x, y = sess.run([data_input.input, data_input.target])
    print(x)
    x, y = sess.run([data_input.input, data_input.target])
    print(x)

if __name__ == "__main__":
    testReadData()