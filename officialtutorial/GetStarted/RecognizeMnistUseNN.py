import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weightVariable(shape, name):
    values = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=values, name=name)

def biasVariable(shape, name):
    bias = tf.constant(0.1, dtype=tf.float32)
    return tf.Variable(initial_value=bias, name=name)

def conv2dStride2mul2(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# define input placeholder
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')


# define first layer

W1 = weightVariable([5, 5, 1, 32], 'w1')
b1 = biasVariable([32], 'b1')
hconv1 = conv2dStride2mul2(x, W1)
hconv1 = tf.nn.relu(hconv1+b1)
hconv1_out = maxPool(hconv1)

# define second layer
W2 = weightVariable([5, 5, 32, 64], 'w2')
b2 = biasVariable([64], 'b2')
hconv2 = conv2dStride2mul2(hconv1_out, W2)
hconv2 = tf.nn.relu(hconv2+b2)
hconv2_out = maxPool(hconv2)

# full connect
wf = weightVariable([7*7*64, 1024], 'wf')
bf = biasVariable([1024], 'bf')
f_input = tf.reshape(hconv2_out, shape=[-1, 7*7*64])
#
f_out = tf.nn.relu( tf.matmul(f_input, wf) + bf )

# dropout
prob = tf.placeholder(dtype=tf.float32)
d_out = tf.nn.dropout(f_out, prob)


#output layer
wo = weightVariable([1024, 10], name='wo')
bo = biasVariable([10], 'bo')
out = tf.matmul(d_out, wo) + bo

loss = tf.nn.softmax_cross_entropy_with_logits(out, y)
loss = tf.reduce_mean(loss)

acc = tf.equal(tf.argmax(out, dimension=1), tf.argmax(y, dimension=1))
acc = tf.reduce_mean( tf.cast(acc, tf.float32) )

optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
begin = datetime.datetime.now()
for i in range(10000):
    batch = mnist.train.next_batch(100)
    d = sess.run(train, feed_dict={x: batch[0].reshape(-1, 28, 28, 1), y: batch[1], prob: 0.5})
    if i % 100 == 1:
        accs = sess.run(acc, feed_dict={x: batch[0].reshape(-1, 28, 28, 1), y: batch[1], prob: 1.0})
        print accs
        end = datetime.datetime.now()
        print (end-begin).seconds
        begin = end
print 'begin test'
accs = 0
for i in range(100):
    batch = mnist.train.next_batch(100)
    accs += sess.run(acc, feed_dict={x: batch[0].reshape(-1, 28, 28, 1), y: batch[1], prob: 1})
print accs/100