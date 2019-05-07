import pickle
import os
import datetime
import time

import tensorflow as tf
import numpy as np

from model import Model 

os.environ["CUDA_VISIBLE_DEVICES"]="0"

data_directory = "../Data"
backup_directory = "../Backup/"

dataset_file_path = data_directory+"/test_dataset"

with open(dataset_file_path, 'rb') as f:
    dataset = pickle.load(f)

test_X = np.array(dataset[0])

n_class = 3
embedding_dim = 100
max_sen_len = 30

hidden_size = 100

x1 = []
x2 = []
for i in range(200):
    x1.append(test_X[i][3])
    x2.append(test_X[i][4])

timestamp = "1557212168"

checkpoint_dir = os.path.abspath(backup_directory+timestamp)
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

graph = tf.Graph()
with graph.as_default():

    session_config = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=False
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    with sess.as_default():

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        model_x1 = graph.get_operation_by_name("input/x1").outputs[0]
        model_x2 = graph.get_operation_by_name("input/x2").outputs[0]
        model_y = graph.get_operation_by_name("input/y").outputs[0]

        model_predictions = graph.get_operation_by_name("predictions").outputs[0]
        model_accuracy = graph.get_operation_by_name("metrics/accuracy").outputs[0]
        #model_c_matrix = graph.get_operation_by_name("metrics/c_matrix").outputs[0]

        feed_dict = {
            model_x1: x1,
            model_x2: x2,
            model_y: np.array([[0, 0, 1]])
        }

        accuracy = sess.run(model_accuracy, feed_dict=feed_dict)
        print("Accurency", accuracy)
