# -*-coding:utf8-*-
import re
import numpy as np
import pickle
def loadData(): #从文件中读取文字，将其处理好返回
    path = 'data/rt-polarity.'
    polarity = ['neg', 'pos']
    pos = []
    neg = []
    max_len = 0
    sign = 0
    for p in polarity:
        f = open(path+p)
        line = f.readline()
        if sign == 0:
            datas = neg
        else:
            datas = pos
        while line:
            line = clean_str(line)
            datas.append(line)
            if max_len < len(line.split()):
                max_len = len(line.split())
            line = f.readline()
        sign = 1
    return pos, neg, max_len

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        count = 0
        for line in xrange(vocab_size):
            if count%10000==0:
                print count
            count += 1
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def getVocabulary(pos, neg):
    vocabulary = set()
    for p in pos:
        vocabulary |= set(p.split())
    for n in neg:
        vocabulary |= set(n.split())
    return list(vocabulary)

def buildWordIndiceMap(pos, neg):
    # 0号的indice不是已出现的word
    wordInice = {}
    indice = 1
    pos.extend(neg)
    for p in pos:
        words = p.strip().split()
        for word in words:
            if word not in wordInice:
                wordInice[word] = indice
                indice += 1
    pickle.dump(wordInice, open('data/wordIndiceMap.d', 'wb') )
    print indice

def MapSentenceToIndicesAndPadding(pos, neg, max_len, wordIndices, padding=0):
    #  padding用来指定左右填充的位数
    positive = []
    negative = []
    for p in pos:
        words = p.strip().split()
        t = []
        for word in words:
            t.append(wordIndices[word])   # bug 此处如果出现不在wordIndice中的word会出现错误
        if len(words) < max_len+padding:
            t.extend([0]*(max_len+padding-len(words)))
        if padding!= 0:
            t = [0]*padding+t
        positive.append(t)

    for n in neg:
        words = n.strip().split()
        t = []
        for word in words:
            t.append(wordIndices[word])
        if len(words) < max_len + padding:
            t.extend([0] * (max_len + padding - len(words)))
        if padding != 0:
            t = [0] * padding + t
        negative.append(t)
    return np.array(positive), np.array(negative)


def getPaddingData():
    pos, neg, maxlen = loadData()
    word_indice = pickle.load(open('data/wordIndiceMap.d'))
    pos, neg = MapSentenceToIndicesAndPadding(pos, neg, maxlen, word_indice)
    return pos, neg

def test():
    pos, neg = getPaddingData()
    word_vec = pickle.load(open('data/word_vecs.d'))
    word_indice = pickle.load(open('data/wordIndiceMap.d'))
    buildDataMatrix(pos, neg, word_vec, word_indice)


def buildDataMatrix(pos, neg, word_vec, word_indice):
    rows = len(word_indice.values())+1
    vecMap = np.random.uniform(-0.25, 0.25, (rows, len(word_vec.values()[0])) )
    vecMap[0] = 0
    for word, vec in word_vec.items():
        vecMap[word_indice[word]] = vec
    print vecMap.shape
    for i in xrange(pos.shape[0]):
        onedata = pos[i]
        dataMatrix = vecMap[onedata]
        print dataMatrix
        print dataMatrix.shape
        break


def processFromBegin():
    pos, neg, maxlen = loadData()
    vocabulary = getVocabulary(pos, neg)
    # word_vecs = load_bin_vec('data/GoogleNews-vectors-negative300.bin', vocabulary)
    buildWordIndiceMap(pos, neg)
    getPaddingData()


if __name__ == "__main__":
    # pos, neg, maxlen = loadData()
    # vocabulary = getVocabulary(pos, neg)
    # print 'begin'
    # word_vecs = load_bin_vec('data/GoogleNews-vectors-negative300.bin', vocabulary)
    # pickle.dump(word_vecs, open('data/word_vecs.d', 'wb'))
    # p, neg, max = loadData()
    # print p[0]
    #
    # test()

    processFromBegin()
    pass
