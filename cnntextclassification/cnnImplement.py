import tensorflow as tf

tf.flags.DEFINE_integer('embedding_size', 100, 'the embedding size')
tf.flags.DEFINE_integer('word_size', 100, 'the word size')
tf.flags.DEFINE_integer('batch_size', 50, 'the batch size')
tf.flags.DEFINE_integer('max_length', 56, 'max sentence length')
tf.flags.DEFINE_integer('filter_channels', 4, 'the channel of each filter')
tf.flags.DEFINE_integer('output_size', 2, 'the output size')

flags = tf.flags.FLAGS
flags._parse_flags()

x_placeholder = tf.placeholder(tf.dtypes.float32, (None, flags.max_length, flags.embedding_size))
y_placeholder = tf.placeholder(tf.dtypes.int32, (None,2))
filter_steps = [3,4,5]

pool_result = []
for filter_size in filter_steps:
    con_weights = tf.variables(
        tf.truncated_normal(
            [filter_size, flags.word_size, 1, flags.filter_channels], stddev=0.1))
    b = tf.variables(tf.constant(0, shape=[flags.filter_channels]) )
    conv_output = tf.nn.conv2d(x_placeholder, con_weights, [1,tf.filter_size,1,1], padding='VALID')