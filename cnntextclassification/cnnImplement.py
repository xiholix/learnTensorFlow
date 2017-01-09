# -*-coding:utf8-*-
import tensorflow as tf
from processData import *
from DataHelper import *
tf.flags.DEFINE_integer('embedding_size', 100, 'the embedding size')
tf.flags.DEFINE_integer('word_size', 300, 'the word size')
tf.flags.DEFINE_integer('batch_size', 50, 'the batch size')
tf.flags.DEFINE_integer('max_length', 56, 'max sentence length')
tf.flags.DEFINE_integer('filter_channels', 4, 'the channel of each filter')
tf.flags.DEFINE_integer('output_size', 2, 'the output size')


flags = tf.flags.FLAGS
flags._parse_flags()

x_placeholder = tf.placeholder(tf.float32, (None, flags.max_length, flags.word_size, 1), name='x_placeholder')
y_placeholder = tf.placeholder(tf.int32, (None,2), name='y_placeholder')
prob = tf.placeholder(tf.float32, name='prob')
filter_steps = [3,4,5]
l2_loss = tf.constant(0.0)
pool_result = []
for filter_size in filter_steps:
    con_weights = tf.Variable(    #该变量定义在for循环中是否不会更新
        tf.truncated_normal(
            [filter_size, flags.word_size, 1, flags.filter_channels], stddev=0.1))
    conv_output = tf.nn.conv2d(x_placeholder, con_weights, [1,1,1,1], padding='VALID')
    conv_output = tf.nn.max_pool(conv_output, [1, (flags.max_length-filter_size+1), 1, 1], [1,1,1,1], padding='VALID')
    pool_result.append(conv_output)
conv_out = tf.concat(3, pool_result)
conv_out = tf.reshape(conv_out, (flags.batch_size, -1) )
feature_size = flags.filter_channels*len(filter_steps)
full_weights = tf.Variable(tf.truncated_normal([feature_size, flags.output_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=(2,)))

l2_loss += tf.nn.l2_loss(full_weights)
l2_loss += tf.nn.l2_loss(b)

out_put = tf.matmul(conv_out, full_weights) + b
out_put = tf.nn.dropout(out_put, prob)
predict = tf.nn.softmax(out_put)
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(predict, y_placeholder) ) + 0.1*l2_loss
y_predict = tf.cast( tf.equal(tf.argmax(predict, 1), tf.argmax(y_placeholder, 1)), tf.float32)
accuracy = tf.reduce_mean(y_predict)


pos, neg = getPaddingData()
word_vec = pickle.load(open('data/word_vecs.d'))
word_indice = pickle.load(open('data/wordIndiceMap.d'))
pos_len = len(pos)
neg_len = len(neg)
length = pos_len + neg_len
labels = [[1,0]]*pos_len
labels += [[0,1]]*neg_len
labels = np.array(labels)
data = np.concatenate((pos,neg), axis=0)
    # shuffle data
indice = np.random.permutation(length)
data = data[indice]
labels = labels[indice]
test_data = data[0.9*length:]
test_labels = labels[0.9*length:]
data = data[:0.9*length]
labels = labels[:0.9*length]
d = DataHelper(50, data.shape[0], data, word_vec, word_indice, labels)
t_d = DataHelper(50, test_data.shape[0], test_data, word_vec, word_indice, test_labels)
optm = tf.train.AdamOptimizer(1e-3)
train_ops = optm.apply_gradients(optm.compute_gradients(cross_entropy))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

def train_step(x, y):
    feed_dicts = {x_placeholder:x, y_placeholder:y, prob:0.5}
    loss = sess.run([train_ops], feed_dict=feed_dicts)

with sess.as_default():
  for i in xrange(10000):
    if i%100 == 0:
        a = 0
        for i in range(20):
            x, y = t_d.next()
            x = np.reshape(x, [-1, flags.max_length, flags.word_size, 1])
            feed_dicts = {x_placeholder:x, y_placeholder:y, prob:1}
            accuracys = sess.run(accuracy, feed_dicts)
            a += accuracys
        print 'test acc is ' + str(a/20)
    else:
        x, y = d.next()
        x = np.reshape(x, [-1, flags.max_length, flags.word_size, 1])
        train_step(x, y)
        if i%100 == 1:
             x = np.reshape(x, [-1, flags.max_length, flags.word_size, 1])
             feed_dicts = {x_placeholder: x, y_placeholder: y, prob:1}
             accuracys = sess.run(accuracy, feed_dicts)
             print 'train acc is ' + str(accuracys)
             print '***************************************\n\n'


