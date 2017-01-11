import tensorflow as tf
import numpy as np
import math

def interfence(images, pixel_size, num_classes, hidden1_size, hidden2_size):
    with tf.name_scope("hidden1"):
        weight = tf.Variable(initial_value=tf.truncated_normal(
            shape=[pixel_size, hidden1_size], stddev=1.0/math.sqrt(pixel_size)),
        name='weight')
        bias = tf.Variable(initial_value=tf.zeros(hidden1_size), name='bias')
        output = tf.nn.relu(tf.matmul(images, weight)+bias)

    with tf.name_scope("hidden2"):
        weight = tf.Variable(initial_value=tf.truncated_normal(
            shape=[hidden1_size, hidden2_size], stddev=1.0/math.sqrt(hidden1_size)),
        name='weight')
        bias = tf.Variable(initial_value=tf.zeros(hidden2_size), name='bias')
        output = tf.nn.relu(tf.matmul(output, weight)+bias)

    with tf.name_scope("output"):
        weight = tf.Variable(initial_value=tf.truncated_normal(
            shape=[hidden2_size, num_classes], stddev=1.0/math.sqrt(hidden2_size)),
                             name='weight')
        bias = tf.Variable(initial_value=tf.zeros(num_classes), name='bias')
        output = tf.matmul(output, weight) + bias

    return output




def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def train_op(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    train_operator = optimizer.minimize(loss, global_step=global_step)
    return train_operator


def evaluate(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))