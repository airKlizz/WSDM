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

x1 = test_X[0][3]
x2 = test_X[0][4]

timestamp = "1557135948"

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

        feed_dict = {
            model.x1: x1,
            model.x2: x2,
            model.y: np.array([0, 0, 1])
        }

        accuracy, c_matrix = sess.run([model.accuracy, model.c_matrix], feed_dict=feed_dict)
        print(c_matrix)
