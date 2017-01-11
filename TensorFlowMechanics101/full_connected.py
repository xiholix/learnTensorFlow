import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist import *
import time
def place_holder(batch_size, pixels):
    x_input = tf.placeholder(tf.float32, shape=[batch_size, pixels])
    y_labels = tf.placeholder(tf.int32, shape=[batch_size])

    return x_input, y_labels


def fill_feed_dict(data_set, images_pl, labels_pl, batch_size):
    images, labels = data_set.next_batch(batch_size)
    feed_dict = {
        images_pl:images,
        labels_pl:labels
    }

    return feed_dict

def do_eval(sess, eval_correct, image_placeholder, labels_placeholder,
            data_set, batch_size):
    true_count = 0
    steps = data_set.num_examples // batch_size
    num_examples = steps * batch_size
    for i in range(steps):
        feed_dict = fill_feed_dict(data_set, image_placeholder, labels_placeholder,
                                   batch_size)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print 'num examples: %d, num corrects: %d, precisions: %.4f'%(
        num_examples, true_count, precision
    )

def run_training():
    data_sets = input_data.read_data_sets('data')

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = place_holder(50, 28*28)
        logits = interfence(images_placeholder, 28*28, 10, 128, 32)
        losses = loss(logits, labels_placeholder)
        train_operator = train_op(losses, 1e-2)
        eval_correct = evaluate(logits, labels_placeholder)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for step in xrange(10000):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder, 50)
            _, loss_value = sess.run([train_operator, losses], feed_dict=feed_dict)
            duration = time.time()-start_time

            if (step+1)%100 == 0:
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test, 50)
                print 'hello '

if __name__ == "__main__":
    run_training()