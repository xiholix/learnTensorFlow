# -*-coding:utf8-*-
import numpy as np
import pickle

class DataHelper:
    def __init__(self, batch_size, max_length, data,  word_indices, labels):
        self.batch_size = batch_size
        self.index = 0
        self.max_length = max_length
        self.data = data
        self.word_indices = word_indices
        self.labels = labels
        rows = len(word_indices.values()) + 1  # what's the type of word_indices.  it maybe is a dict



    def next(self):
        if self.index + self.batch_size < self.max_length:
            data = self.data[self.index: self.batch_size + self.index]
            labels = self.labels[self.index: self.batch_size + self.index]
            self.index += self.batch_size
        else:
            data = self.data[self.index:]
            labels = self.labels[self.index:]
            read_nums = self.max_length - self.index
            remain_nums = self.batch_size - read_nums
            remain_data = self.data[:remain_nums]
            remain_labels = self.labels[:remain_nums]
            self.index = remain_nums
            data = np.concatenate((data, remain_data), axis=0)
            labels = np.concatenate((labels, remain_labels), axis=0)
            # data = data.concatenate(remain_data, axis=0) #没有concatenate方法

        return data, labels

def test():
    data = np.array( np.arange(15).reshape(5,3) )
    labels = np.array( np.arange(10).reshape(5,2))

    word_indices = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12,
                    13:13, 14:14, 15:15}
    d = DataHelper(3, 5, data, word_indices, labels)
    for i in xrange(4):
        data, labels = d.next()
        print data
        print labels

        print '***************************'

