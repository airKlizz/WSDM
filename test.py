import pickle
import os
import datetime
import time
import csv

import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"

data_directory = "../Data"
backup_directory = "../Models/"

dataset_file_path = data_directory+"/train_dataset_2"
#dataset_file_path = data_directory+"/train_dataset"

print("Restore Data")

with open(dataset_file_path, 'rb') as f:
    dataset = pickle.load(f)

print("DATASET :", np.shape(dataset))
print("train_X :", np.shape(dataset[0]))
print("train_y :", np.shape(dataset[1]))
print("test_X :", np.shape(dataset[2]))
print("test_y :", np.shape(dataset[3]))

train_X = np.array(dataset[0])
train_y = np.array(dataset[1])
test_X = np.array(dataset[2])
test_y = np.array(dataset[3])

batch_size = 200
class_weights = [1/15, 1/5, 1/16]

#timestamp = "1557409770" # DDD
#timestamp = '1557322767' # SDD
#timestamp = '1557478792' #SSSc
#timestamp = '1557596444' #SScv2
#timestamp = '1557654254' #SScv3
#timestamp = '1557655718' #SScv4
#timestamp = '1557663468' # SSS dropout
timestamp = '1557917420' 

specifications = 'SSS combine sampling dataset'


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
        #model_class_weights = graph.get_operation_by_name("input/class_weights").outputs[0]

        model_predictions = graph.get_operation_by_name("predictions").outputs[0]

        batch = 0
        idx_max = 0

        accuracy_test = 0
        accuracy_test_weights = 0
        sum_accuracy = 0
        sum_weights = 0

        while batch < 50:
            print(batch, "/", len(test_X)/batch_size)
            idx_min = batch * batch_size
            idx_max = min((batch+1) * batch_size, len(test_X))
            x1 = np.array([test_X[idx_min][0]])
            x2 = np.array([test_X[idx_min][1]])
            y = np.array([test_y[idx_min]])

            for i in range(idx_min, idx_max):
                x1 = np.append(x1, np.array([test_X[i][0]]), axis=0)
                x2 = np.append(x2, np.array([test_X[i][1]]), axis=0)
                y = np.append(y, np.array([test_y[i]]), axis=0)

            feed_dict = {
                model_x1: x1,
                model_x2: x2,
                model_y: y,
                #model_class_weights: np.ones(len(x1))
            }

            predictions = sess.run(model_predictions, feed_dict=feed_dict)

            for i in range(len(predictions)):
                if predictions[i] == np.argmax(y[i]):
                    accuracy_test +=1
                    accuracy_test_weights += class_weights[np.argmax(y[i])]
                
                sum_accuracy += 1
                sum_weights += class_weights[np.argmax(y[i])]

            batch += 1
        
        accuracy_test = accuracy_test/sum_accuracy
        accuracy_test_weights = accuracy_test_weights/sum_weights

        batch = 0
        idx_max = 0

        accuracy_train = 0
        accuracy_train_weights = 0
        sum_accuracy = 0
        sum_weights = 0

        while batch < 50:
            print(batch, "/", len(train_X)/batch_size)
            idx_min = batch * batch_size
            idx_max = min((batch+1) * batch_size, len(train_X))
            x1 = np.array([train_X[idx_min][0]])
            x2 = np.array([train_X[idx_min][1]])
            y = np.array([train_y[idx_min]])

            for i in range(idx_min, idx_max):
                x1 = np.append(x1, np.array([train_X[i][0]]), axis=0)
                x2 = np.append(x2, np.array([train_X[i][1]]), axis=0)
                y = np.append(y, np.array([train_y[i]]), axis=0)

            feed_dict = {
                model_x1: x1,
                model_x2: x2,
                model_y: y,
                #model_class_weights: np.ones(len(x1))
            }

            predictions = sess.run(model_predictions, feed_dict=feed_dict)

            for i in range(len(predictions)):
                if predictions[i] == np.argmax(y[i]):
                    accuracy_train +=1
                    accuracy_train_weights += class_weights[np.argmax(y[i])]
                
                sum_accuracy += 1
                sum_weights += class_weights[np.argmax(y[i])]

            batch += 1
        
        accuracy_train = accuracy_train/sum_accuracy
        accuracy_train_weights = accuracy_train_weights/sum_weights



file = open("results.txt","a") 
line = timestamp+" - "+specifications+" : test "+str(accuracy_test)+" train "+str(accuracy_train)+" test weights "+str(accuracy_test_weights)+" train weights "+str(accuracy_train_weights)+"\n"
file.write(line) 
file.close() 
