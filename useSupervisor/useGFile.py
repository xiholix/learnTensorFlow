# --*--coding:utf8--*--

import tensorflow as tf

def readFromFile():
    f = tf.gfile.GFile("hello", "a")
    f.write("add to file")











if __name__ == "__main__":
    readFromFile()