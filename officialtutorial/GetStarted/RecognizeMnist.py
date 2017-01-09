import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print mnist.validation.images[0].shape
print mnist.validation.labels[0].shape

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.int32, shape=[None, 10])

W = tf.Variable( tf.truncated_normal(shape=[784, 10]), name='W')
b = tf.Variable( tf.zeros(shape=[10]), name='b')

predict_y = tf.nn.xw_plus_b(x, W, b)
cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_y, y, dim=1))
accuracy = tf.equal( tf.argmax(predict_y, dimension=1), tf.argmax(y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entroy)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for i in range(1000):
    batch = mnist.train.next_batch(100)
    d = sess.run(train, feed_dict={x:batch[0], y:batch[1]})
    if i%100==1:
        acc = sess.run(accuracy, feed_dict={x:batch[0], y:batch[1]})
        print acc

test_x = mnist.test.images
test_y = mnist.test.labels
acc = sess.run(accuracy, feed_dict={x:test_x, y:test_y})
print acc