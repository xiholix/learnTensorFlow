# -*-coding:utf8-*-
from processData import *
import numpy as np
class  DataHelper:
    def __init__(self, batch_size, max_length, data, word_vec, word_indices, labels):
        self.batch_size = batch_size
        self.index = 0
        self.max_length = max_length
        self.data = data
        self.word_vec = word_vec
        self.word_indices = word_indices
        self.labels = labels
        rows = len(word_indices.values()) + 1    # what's the type of word_indices.  it maybe is a dict
        self.vecMap = np.random.uniform(-0.25, 0.25, (rows, len(word_vec.values()[0])))
        self.vecMap[0] = 0
        for word, vec in word_vec.items():
            # by this method, we can use the index get the responding vector from this map
            self.vecMap[word_indices[word]] = vec

    def next(self):
        if self.index+self.batch_size < self.max_length:
            data = self.data[self.index: self.batch_size+self.index]
            labels = self.labels[self.index: self.batch_size+self.index]
            self.index += self.batch_size
        else:
            data = self.data[self.index:]
            labels = self.labels[self.index:]
            read_nums = self.max_length - self.index
            remain_nums = self.batch_size - read_nums
            remain_data = self.data[:remain_nums]
            remain_labels = self.labels[:remain_nums]
            self.index = remain_nums
            data = np.concatenate((data,remain_data), axis=0)
            labels = np.concatenate((labels, remain_labels), axis=0)
            # data = data.concatenate(remain_data, axis=0) #没有concatenate方法
        result_3d_data = []
        for i in xrange(data.shape[0]):
            onedata = data[i]
            result_3d_data.append(self.vecMap[onedata])
        result_3d_data = np.array(result_3d_data)
        return result_3d_data, labels

def test():
    data = np.array( np.arange(15).reshape(5,3) )
    labels = np.array( np.arange(10).reshape(5,2))
    word_vec = {0:[0,0,0,0], 1:[1,1,1,1], 6:[6,6,6,6], 9:[9,9,9,9]}
    word_indices = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12,
                    13:13, 14:14, 15:15}
    d = DataHelper(3, 5, data, word_vec, word_indices, labels)
    for i in xrange(4):
        data, labels = d.next()
        print data
        print labels

        print '***************************'

def testListArray():
    a = np.array([[0,1], [2,3]])
    b = np.array([[4,5], [6,7]])
    c = [a,b]
    c = np.array(c)
    print c.shape
    print c

def usedataHelperOnTrueData():
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
    print data.shape
    print labels.shape
    d = DataHelper(5, 10662, data, word_vec, word_indice, labels)
    data, label = d.next()
    print data.shape
    print label.shape
    print label
    print data

if __name__ == "__main__":
    usedataHelperOnTrueData()
    # testListArray()
    # test()