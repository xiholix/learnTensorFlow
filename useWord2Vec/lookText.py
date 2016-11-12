#-*-coding:utf8-*-
def exploreData():
    path = 'text8'
    f = open(path)
    line = f.readline()
    print len(line)
    print len(line.split())



if __name__ == "__main__":
    exploreData()