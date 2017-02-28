import tensorflow as tf

a = tf.truncated_normal(shape=[3,3], stddev=0.1)
b = tf.Variable(a)
b += 1

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

get_b, get_a = sess.run([b,a])
# get_a = sess.run(a)

print get_a
print get_b
print get_a==get_b