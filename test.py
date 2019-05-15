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

#timestamp = "1557409770" # DDD
#timestamp = '1557322767' # SDD
#timestamp = '1557478792' #SSSc
#timestamp = '1557596444' #SScv2
#timestamp = '1557654254' #SScv3
#timestamp = '1557655718' #SScv4
#timestamp = '1557663468' # SSS dropout
timestamp = '1557322767' #SDD

specifications = 'model SDD'


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
        try:
            model_class_weights = graph.get_operation_by_name("input/class_weights").outputs[0]
            class_weights_bool = True
        except:
            class_weights_bool = False


        model_predictions = graph.get_operation_by_name("predictions").outputs[0]

        batch = 0
        idx_max = 0

        accuracy_test = 0

        while idx_max < len(test_X)-1:
            print(batch, "/", len(test_X)/batch_size)
            idx_min = batch * batch_size
            idx_max = min((batch+1) * batch_size, len(test_X))
            x1 = np.array([test_X[idx_min][0]])
            x2 = np.array([test_X[idx_min][1]])
            y = np.array([test_y[idx_min]])

            for i in range(idx_min, idx_max):
                x1 = np.append(x1, np.array([test_X[i][0]]), axis=0)
                x2 = np.append(x2, np.array([test_X[i][1]]), axis=0)
                y = np.array([test_y[i]])

            feed_dict = {
                model_x1: x1,
                model_x2: x2,
                model_y: y,
            }

            if class_weights_bool:
                feed_dict["model_class_weights"] = np.ones(len(x1))

            predictions = sess.run(model_predictions, feed_dict=feed_dict)

            print(predictions)
            print(y)

            for i in range(len(predictions)):
                if predictions[i] == np.argmax(y[idx_min+i]):
                    accuracy_test +=1

            batch += 1
        
        accuracy_test = accuracy_test/len(test_X)

        batch = 0
        idx_max = 0

        accuracy_train = 0

        while idx_max < len(train_X)-1:
            print(batch, "/", len(train_X)/batch_size)
            idx_min = batch * batch_size
            idx_max = min((batch+1) * batch_size, len(train_X))
            x1 = np.array([train_X[idx_min][0]])
            x2 = np.array([train_X[idx_min][1]])
            y = np.array([train_y[idx_min]])

            for i in range(idx_min, idx_max):
                x1 = np.append(x1, np.array([train_X[i][0]]), axis=0)
                x2 = np.append(x2, np.array([train_X[i][1]]), axis=0)
                y = np.array([train_y[i]])

            feed_dict = {
                model_x1: x1,
                model_x2: x2,
                model_y: y,
            }

            if class_weights_bool:
                feed_dict["model_class_weights"] = np.ones(len(x1))

            predictions = sess.run(model_predictions, feed_dict=feed_dict)

            print(predictions)
            print(y)

            for i in range(len(predictions)):
                print(predictions[i])
                if predictions[i] == np.argmax(y[idx_min+i]):
                    accuracy_train +=1

            batch += 1
        
        accuracy_train = accuracy_train/len(train_X)



file = open("results.txt","a") 
 
file.write(timestamp, " - ", specifications, " : ", "test ",accuracy_test, " train ", accuracy_train, "\n") 
 
file.close() 
