#-*-coding:utf8-*-
import tensorflow as tf
from processData import *
from dataHelper import *
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
filter_steps = [3,4,5]

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

out_put = tf.matmul(conv_out, full_weights) + b

predict = tf.nn.softmax(out_put)
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(predict, y_placeholder) )
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
d = dataHelper(50, 10662, data, word_vec, word_indice, labels)

optm = tf.train.AdamOptimizer(1e-3)
train_ops = optm.apply_gradients(optm.compute_gradients(cross_entropy))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

def train_step(x, y):
    feed_dicts = {x_placeholder:x, y_placeholder:y}
    loss = sess.run([train_ops], feed_dict=feed_dicts)

with sess.as_default():
  for i in xrange(10000):
    x, y = d.next()
    x = np.reshape(x, [-1,flags.max_length, flags.word_size, 1])
    if i%10 == 0:
        feed_dicts = {x_placeholder:x, y_placeholder:y}
        loss, accuracys = sess.run([train_ops, accuracy], feed_dicts)
        print accuracys
    else:
        train_step(x, y)


