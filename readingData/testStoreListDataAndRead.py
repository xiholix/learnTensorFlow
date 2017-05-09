# --*--coding:utf8--*--

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



def produce_recorder_data_file(_path):
    writer = tf.python_io.TFRecordWriter(_path)

    documents = np.arange(20).reshape((5,4))
    querys = np.arange(15).reshape((5,3))
    answers = np.arange(5).reshape((5,1))
    for i in xrange(len(documents)):
        documentsFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=documents[i]))  #如果是值本来就是list此处就不应该有[]
        #不同与writeTFRecord.py中
        querysFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=querys[i]))
        answersFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=answers[i]))

        features = tf.train.Features(feature={'documents':documentsFeature, 'querys':querysFeature, 'answers':answersFeature})
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord():
    path = ["train.tf"]
    reader = tf.TFRecordReader()
    inputs = tf.train.string_input_producer(path)
    _, example = reader.read(inputs)
    features = tf.parse_single_example(example, features={
        'documents':tf.FixedLenFeature([4], tf.int64),  #此处的[]中必须填写正确的维度
        'querys':tf.FixedLenFeature([3], tf.int64),
        'answers':tf.FixedLenFeature([1], tf.int64),
    })

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # tf.train.start_queue_runners(sess=sess)
    tf.train.start_queue_runners(sess=sess)
    f = sess.run(features)
    f = sess.run(features)  #运行多少次，就读取第几个example

    print(f['documents'])


if __name__ == "__main__":
    # produce_recorder_data_file("train.tf")
    read_tfrecord()