# --*--coding:utf8--*--
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import reader

class DataInput():
    def __init__(self, raw_data, batch_size, num_step):
        self.input, self.target = reader.ptb_producer(raw_data, batch_size, num_step)
        self.batchSize =batch_size
        self.numStep = num_step
        self.epcheSize = ((len(raw_data)//batch_size)-1) // num_step

class Config(object):
    batchSize = 20
    learingRate = 1.0
    maxGradNorm = 1
    numLayers = 2
    numSteps = 20
    hiddenSize = 200
    maxEpoch = 4
    initScale = 0.1
    maxMaxEpoch = 13
    keepProb = 1.0
    lrDecay = 0.5
    vocabSize=10000


class PTBModel():
    def __init__(self, input, config, is_trainig):
        self.input = input
        self.batchSize = config.batchSize
        self.numSteps = config.numSteps
        self.hiddenSize = config.hiddenSize
        def rnn_layer():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hiddenSize, forget_bias=0.0, state_is_tuple=True)

        cell = rnn_layer   #这里不能出现括号调用
        if is_trainig and config.keepProb<1.0:
            def cell():
                return tf.contrib.rnn.DropoutWrapper(
                    rnn_layer(), output_keep_prob=config.keepProb)

        multiCellLayers = tf.contrib.rnn.MultiRNNCell(
            [cell() for _ in range(config.numLayers)], state_is_tuple=True)

        self.initStates = multiCellLayers.zero_state(self.batchSize, tf.float32)

        with tf.device('/cpu:0'):
            embedding = tf.get_variable(
                "embedding", [config.vocabSize, config.hiddenSize], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, input.input)

        if is_trainig and config.keepProb<1.0:
            inputs = tf.nn.dropout(inputs, keep_prob=config.keepProb)

        outputs = []
        state = self.initStates
        with tf.variable_scope("RNN"):
            for time_step in range(config.numSteps):
                if time_step > 0 : tf.get_variable_scope().reuse_variables()
                (cell_output, state) = multiCellLayers(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape( tf.concat(outputs, 1), [-1, config.hiddenSize])

        softmaxW = tf.get_variable(
            "softmaxW", [config.hiddenSize, config.vocabSize], dtype=tf.float32)
        softmaxB = tf.get_variable(
            "softmaxB", [config.vocabSize], dtype=tf.float32)
        logits = tf.matmul(output, softmaxW) + softmaxB
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input.target, [-1])],
            [tf.ones([config.batchSize*config.numSteps], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / config.batchSize
        self.final_state = state

        if not is_trainig:
            return



def testReadData():
    path = 'data/'
    datas = reader.ptb_raw_data(path)
    train_data, valid_data, test_data, _ = datas
    config = Config()
    data_input = DataInput(train_data, config.batchSize, config.numSteps)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    x, y = sess.run([data_input.input, data_input.target])
    print(x)
    x, y = sess.run([data_input.input, data_input.target])
    print(x)


def runModel():
    path = 'data/'
    datas = reader.ptb_raw_data(path)
    train_data, valid_data, test_data, _ = datas
    config = Config()
    data_input = DataInput(train_data, config.batchSize, config.numSteps)
    model = PTBModel(data_input, config, False)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    cost = sess.run(model.cost)
    print (cost)
    cost = sess.run(model.cost)
    print(cost)

if __name__ == "__main__":
    # testReadData()
    runModel()