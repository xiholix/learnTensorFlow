import tensorflow as tf

def test_tf__train__match_filenames_once():
    path_list = ['file1.csv', 'file2.csv', 'file3.csv']
    p = tf.train.string_input_producer(path_list, 3)
    sess = tf.Session()
    with tf.Graph().as_default():
        m = sess.run(p)
        # print m
def read_csv():
    path_list = ['file1.csv', 'file2.csv', 'file3.csv']
    new_path_list = []
    for path in path_list:
        path = 'data/' + path
        new_path_list.append(path)
    print new_path_list
    file_queue = tf.train.string_input_producer(new_path_list)
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)

    record_default = [[0],[0],[0],[0]]
    col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_default)
    feature = tf.pack([col1,col2,col3])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(2):
            examples, labels = sess.run([feature, col4])
            print examples
            # print key
    coord.request_stop()
    coord.join(threads)
    

if __name__ == "__main__":
    read_csv()