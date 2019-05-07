import pickle
import os
import datetime
import time
import csv

import tensorflow as tf
import numpy as np

from model import Model 

os.environ["CUDA_VISIBLE_DEVICES"]="0"

data_directory = "../Data"
backup_directory = "../Backup/"

sample_submission_file_path = data_directory+"/sample_submission.csv"
submission_file_path = data_directory+"/submission.csv"
dataset_file_path = data_directory+"/test_dataset"

submission = []

with open(sample_submission_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        submission.append([row[0], row[1]])

with open(dataset_file_path, 'rb') as f:
    dataset = pickle.load(f)

test_X = np.array(dataset[0])

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

        for i in range(len(test_X)):

            feed_dict = {
                model_x1: [test_X[i][3]],
                model_x2: [test_X[i][4]],
                model_y: np.array([[0, 0, 0]])
            }

            predictions = sess.run(model_predictions, feed_dict=feed_dict)

            print(predictions)

            if predictions == 0:
                submission[i][1] = "agreed"
            elif predictions == 1:
                submission[i][1] = "disagreed"
            else :
                submission[i][1] = "unrelated"

with open(submission_file_path, newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in submission:
        writer.writerow(row)
