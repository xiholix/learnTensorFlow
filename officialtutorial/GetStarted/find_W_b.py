import tensorflow as tf
import numpy as np

x = np.random.rand(100).astype(np.float32)
y = 0.4*x + 0.3

W = tf.Variable(tf.truncated_normal([1], mean=0.0, stddev=0.1), name='W')
b = tf.Variable(tf.zeros(shape=[1]), name='b')

y_predict = x*W + b
loss = tf.reduce_mean(tf.square(y_predict - y))

optm = tf.train.GradientDescentOptimizer(0.5)
t = optm.apply_gradients(optm.compute_gradients(loss))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(10000):
    sess.run(t)
    if step%20==0:
        w, b2 = sess.run([W,b])
        print w, b2
