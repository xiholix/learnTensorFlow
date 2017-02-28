import tensorflow as tf
from newDataHelper import DataHelper
import pickle
import numpy as np
from processData import *
def getWordVecMap(word_vec, word_indices):
    rows = len(word_indices.values())+1
    columns = len(word_vec.values()[0])
    word_vec_map = np.random.uniform(-0.5, 0.5, size=(rows, columns))
    word_vec_map[0] = 0
    for word, vec in word_vec.items():
        if word in word_indices.keys():
           indices = word_indices[word]
           word_vec_map[indices] = vec
    return word_vec_map

class CNN:
    def __init__(self, word_vec_map, squence_length, num_classes, num_filters, filter_step,
                 voc_dim):
        self.word_vec_map = tf.get_variable('word_vec_map', shape=word_vec_map.shape,
                                            initializer=tf.constant_initializer(word_vec_map), trainable=False)
        self.init_word_vec_map = word_vec_map
        self.num_filters = num_filters
        self.filter_step = filter_step
        self.voc_dim = voc_dim
        self.x = tf.placeholder(tf.int32, shape=[None, squence_length], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes], name='b')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.device('/cpu:0'):
            self.net_input = tf.nn.embedding_lookup(self.word_vec_map, self.x)
            self.net_input_tensor = tf.expand_dims(self.net_input, dim=-1)
        pool_out = []
        # convolution layer
        for step in self.filter_step:
            conv_w = tf.truncated_normal(shape=[step, self.voc_dim, 1, self.num_filters], stddev=0.1, name='conv_w')
            conv_w = tf.Variable(conv_w)
            conv_b = tf.constant(0.1, shape=[self.num_filters], dtype=tf.float32, name='conv_b')
            conv_b = tf.Variable(conv_b)
            conv_out = tf.nn.conv2d(self.net_input_tensor, conv_w, strides=[1,1,1,1], padding='VALID') + conv_b
            conv_out = tf.nn.relu(conv_out)
            # the output dimension is (batch_size, squence_length-step+1, 1, num_filters)
            conv_out = tf.nn.max_pool(conv_out, ksize=[1, squence_length-step+1, 1, 1], strides=[1,1,1,1], padding='VALID')
            # the conv_out dimension is (batch_size, 1, 1, 1)
            pool_out.append(conv_out)
        full_input = tf.concat(3, pool_out)
        full_input = tf.reshape(full_input, shape=[-1, len(self.filter_step)*self.num_filters])
        full_input = tf.nn.dropout(full_input, self.keep_prob)

        full_w = tf.truncated_normal(shape=[len(self.filter_step)*self.num_filters, num_classes], stddev=0.1, name='full_w')
        full_b = tf.constant(0.1, shape=[num_classes], dtype=tf.float32, name='full_b')
        full_w = tf.Variable(full_w)
        full_b = tf.Variable(full_b)
        output = tf.matmul(full_input, full_w) + full_b

        self.loss = tf.nn.softmax_cross_entropy_with_logits(output, self.y)
        acc = tf.cast(tf.equal(tf.argmax(output, dimension=1), tf.argmax(self.y, dimension=1)), tf.float32)
        self.acc = tf.reduce_mean(acc)
        optimizer = tf.train.AdamOptimizer(1e-2)
        self.train = optimizer.minimize(self.loss)
        self.equals = tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(self.init_word_vec_map, tf.float32), self.word_vec_map) ,tf.int32))





def test():
    word_vec = pickle.load(open('data/word_vecs.d'))
    word_indice = pickle.load(open('data/wordIndiceMap.d'))
    word_vec_map = getWordVecMap(word_vec, word_indice)
    cnn = CNN(word_vec_map, 56, 2, 4, [3,4,5], 300)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    pos, neg = getPaddingData()
    pos_len = len(pos)
    neg_len = len(neg)
    length = pos_len + neg_len
    labels = [[1, 0]] * pos_len
    labels += [[0, 1]] * neg_len
    labels = np.array(labels)
    data = np.concatenate((pos, neg), axis=0)
    # shuffle data
    indice = np.random.permutation(length)
    data = data[indice]
    labels = labels[indice]
    test_data = data[0.9 * length:]
    test_labels = labels[0.9 * length:]
    data = data[:0.9 * length]
    labels = labels[:0.9 * length]
    d = DataHelper(50, data.shape[0], data, word_indice, labels)
    t_d = DataHelper(50, test_data.shape[0], test_data, word_indice, test_labels)

    optm = tf.train.AdamOptimizer(1e-3)
    train_ops = optm.apply_gradients(optm.compute_gradients(cnn.loss))
    for i in range(10000):
         x, y = d.next()
         x = np.reshape(x, [-1, 56])
         feed_dict = {cnn.x:x, cnn.y:y, cnn.keep_prob:0.9}

         acc,_, equals = sess.run([cnn.acc, cnn.train, cnn.equals], feed_dict=feed_dict)
         if i%100 == 10:
             print acc
             print equals
             # break
    word_vec = pickle.load(open('data/word_vecs.d'))
    word_indice = pickle.load(open('data/wordIndiceMap.d'))
    word_vec_map = getWordVecMap(word_vec, word_indice)
    # t = tf.reduce_sum( tf.cast(tf.not_equal(tf.cast(word_vec_map, tf.float32), cnn.word_vec_map), tf.float32))

    # print t


test()