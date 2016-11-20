from cnnDefine import CNNNet
from dataHelper import *
import tensorflow as tf


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
data_length = len(data)

d = dataHelper(50, 10662, data, word_vec, word_indice, labels)

cnn = CNNNet(4, [3,4,5], 50, 300, 56, 2)
optm = tf.train.AdamOptimizer(1e-3)
train_ops = optm.apply_gradients(optm.compute_gradients(cnn.loss))
tf.summary.scalar('accuracy', cnn.accuracy)
tf.summary.scalar('loss', cnn.loss)
tf.summary.histogram('full_w', cnn.full_w)
summary = tf.merge_all_summaries()
sess = tf.Session()
summaryWriter = tf.train.SummaryWriter('./log/summary/', sess.graph)


sess.run(tf.initialize_all_variables())

def train_step(x, y):
    feed_dicts = {cnn.input_x:x, cnn.input_y:y}
    loss = sess.run([train_ops], feed_dict=feed_dicts)

with sess.as_default():
  print 'he'
  for i in xrange(10000):
    x, y = d.next()
    x = np.reshape(x, [-1,56, 300, 1])
    sign =0
    arr = 0
    if i%10 == 0:
        if sign==0:
            sign = 1
            arr = cnn.conv_w[0].eval()
        feed_dicts = {cnn.input_x:x, cnn.input_y:y}
        loss, accuracys, data = sess.run([train_ops, cnn.accuracy, summary], feed_dicts)
        # summaryWriter.add_summary(data)
        print accuracys
        s = cnn.conv_w[0].eval()
        t = np.sum( s-arr)
        print t
    else:
        train_step(x, y)