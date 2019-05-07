import pickle
import os
import datetime
import time
import csv

import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"

data_directory = "../Data"
backup_directory = "../Backup/"

sample_submission_file_path = data_directory+"/sample_submission.csv"
submission_file_path = data_directory+"/submission.csv"
dataset_file_path = data_directory+"/test_dataset"

batch_size = 200

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

        batch = 0
        idx_max = 0

        while idx_max < len(test_X)-1:

            idx_min = batch * batch_size
            idx_max = min((batch+1) * batch_size, len(test_X)-1)
            x1 = np.array([test_X[idx_min][3]])
            x2 = np.array([test_X[idx_min][4]])

            for i in range(idx_min+1, idx_max):
                x1 = np.append(x1, np.array([test_X[i][3]]), axis=0)
                x2 = np.append(x2, np.array([test_X[i][4]]), axis=0)

            feed_dict = {
                model_x1: x1,
                model_x2: x2,
                model_y: np.array([[0, 0, 0]])
            }

            predictions = sess.run(model_predictions, feed_dict=feed_dict)

            for i in range(len(predictions)):
                if predictions[i] == 0:
                    submission[idx_min+i+1][1] = "agreed"
                elif predictions[i] == 1:
                    submission[idx_min+i+1][1] = "disagreed"
                elif predictions[i] == 2:
                    submission[idx_min+i+1][1] = "unrelated"
                else :
                    print("Error prediction")

            batch += 1

with open(submission_file_path, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(submission)
