#-*-coding:utf8-*-
import tensorflow as tf

class CNNNet():
    def __init__(self, num_filters, filter_steps, batch_size, wordvec_size, sentence_len, output_size):
        self.num_filters = num_filters
        self.filter_steps = filter_steps
        self.batch_size = batch_size
        self.wordvec_size = wordvec_size
        self.sentence_len = sentence_len
        self.output_size = output_size

        # define input data placeholder
        self.input_x = tf.placeholder(tf.float32, shape=(None, self.sentence_len, self.wordvec_size, 1), name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=(None, self.output_size), name='input_y')
        self.drop_pro = tf.placeholder(tf.float32, shape=(1,), name='drop_pro')
        self.conv_w = []

        # define convolution layer and pool
        with tf.name_scope('conv_layer') as scope:
            conv_result = []
            for step in self.filter_steps:
              con_w = tf.Variable( initial_value=tf.truncated_normal(shape=(step, self.wordvec_size, 1, self.num_filters),
                                                                          stddev=0.1),
                                   name='conv_w')
              self.conv_w.append(con_w)
              conv_out = tf.nn.conv2d(self.input_x, con_w, strides=[1,1,1,1], padding='VALID', name='conv_out')
              # in our code, conv_out has shape(batch_size, sentenc_len-step+1, 1, num_filters）
              conv_result.append( tf.nn.max_pool(conv_out,
                                                 ksize=(1, (self.sentence_len-step+1), 1, 1),
                                                 strides=[1,1,1,1],
                                                 padding='VALID'))
            conv_feature_size = self.num_filters*len(self.filter_steps)
            conv_feature = tf.concat(3, conv_result)
            self.conv_feature = tf.reshape(conv_feature, shape=(batch_size, conv_feature_size), name='conv_feature')

        #define fully connected layer
        with tf.name_scope('fullyconnected') as scope:
            self.full_w = tf.Variable(initial_value=tf.truncated_normal(shape=(conv_feature_size, self.output_size),
                                                                   stddev=0.1),
                                 name='full_w')
            self.full_b = tf.Variable(tf.constant(0.1, shape=[2,]), name='full_b' )
            output = tf.nn.xw_plus_b(self.conv_feature, self.full_w, self.full_b, name='full_out')
        self.output = output
        accuracy = tf.equal( tf.arg_max(output, dimension=1), tf.arg_max(self.input_y, dimension=1))
        accuracy = tf.cast(accuracy, tf.float32)
        self.accuracy = tf.reduce_mean(accuracy, name='accuracy')

        softmax = tf.nn.softmax(output)
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(softmax, self.input_y, dim=1), name='loss')
        self.loss = loss
