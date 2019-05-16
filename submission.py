import pickle
import os
import datetime
import time
import csv

import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="3"

data_directory = "../Data"
backup_directory = "../Models/"

sample_submission_file_path = data_directory+"/sample_submission.csv"
submission_file_path = data_directory+"/submission_15_2.csv"
dataset_file_path = data_directory+"/test_dataset_2"

batch_size = 200

submission = []

with open(sample_submission_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        submission.append([row[0], row[1]])

with open(dataset_file_path, 'rb') as f:
    dataset = pickle.load(f)

test_id = dataset[0]
test_X = np.array(dataset[1])

print("Shape:")
print(len(test_id))
print(np.shape(test_X))

timestamp = '1557926061' 

checkpoint_dir = os.path.abspath(backup_directory+timestamp)
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

results_dict = {}

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
        #model_class_weights = graph.get_operation_by_name("input/class_weights").outputs[0]

        model_predictions = graph.get_operation_by_name("predictions").outputs[0]

        batch = 0
        idx_max = 0

        while idx_max < len(test_X):

            idx_min = batch * batch_size
            idx_max = min((batch+1) * batch_size, len(test_X))
            
            x1 = test_X[idx_min:idx_max, 0]
            x2 = test_X[idx_min:idx_max, 1]
            batch_id = test_id[idx_min:idx_max]

            feed_dict = {
                model_x1: x1,
                model_x2: x2,
                model_y: np.zeros((len(x1), 3)),
                #model_class_weights: np.ones(len(x1)),
            }

            predictions = sess.run(model_predictions, feed_dict=feed_dict)

            for i in range(len(predictions)):
                if predictions[i] == 0:
                    results_dict[batch_id[i]] = "agreed"
                elif predictions[i] == 1:
                    results_dict[batch_id[i]] = "disagreed"
                elif predictions[i] == 2:
                    results_dict[batch_id[i]] = "unrelated"
                else :
                    print("Error prediction")

            batch += 1

for i in range(1, len(submission)):
    submission[i][1] = results_dict[submission[i][0]]

with open(submission_file_path, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in range(len(submission)):
        wr.writerow(submission[i])
