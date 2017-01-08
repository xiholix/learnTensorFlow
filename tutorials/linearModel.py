import tempfile
import urllib
f = 'data/adult.data'
f2 = 'data/adult.test'
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", f)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", f2)

