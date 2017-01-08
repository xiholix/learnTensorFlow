import tensorflow as tf
import numpy as np
import pandas as pd

IRIS_TRAINING = 'data/iris_training.csv'
IRIS_TEST = 'data/iris_test.csv'

# training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
# print training_set

data = pd.read_csv(IRIS_TRAINING)
training_data = data.values[:,:4]
training_target = np.asarray(data.values[:, 4], dtype=np.int32)
test = pd.read_csv(IRIS_TEST)
test_data = data.values[:, :4]
test_target = np.asarray(data.values[:,4], dtype=np.int32)

a = tf.contrib.layers.real_valued_column('a')
b = tf.contrib.layers.real_valued_column('b')
c = tf.contrib.layers.real_valued_column('c')
d = tf.contrib.layers.real_valued_column('d')


classifier = tf.contrib.learn.DNNClassifier(feature_columns=[a,b,c,d],
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir='/iris_model')
training = { lambda x:str(x):training_data[:,x] for x in range(4)}
classifier.fit(x=training, y=training_target, steps=2000)
accuracy_score = classifier.evaluate(x=test_data, y=test_target)['accuracy']
print accuracy_score
