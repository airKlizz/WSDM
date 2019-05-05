#!/usr/bin/env python
# coding: utf-8

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

dataset_file_path = data_directory+"/dataset"

with open(dataset_file_path, 'rb') as f:
    dataset = pickle.load(f)

train_X = dataset[0]
train_y = dataset[1]
test_X = dataset[2]
test_y = dataset[3]

n_class = 3
embedding_dim = 100
max_sen_len = 30

hidden_size = 100

learning_rate = 0.001
batch_size = 100
test_batch_size = 500
num_epochs = 10
evaluate_every = 50

nb_batch_per_epoch = int(len(train_X)/batch_size+1)

allow_soft_placement = True
log_device_placement = False

with tf.Graph().as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement
    )
    session_config.gpu_options.allow_growth = False
    session_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=session_config)

    with sess.as_default():
        model = Model(
        max_sen_len = max_sen_len,
        embedding_dim = embedding_dim,
        class_num = n_class,
        hidden_size = hidden_size
        )

        model.build_model()

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath(backup_directory+timestamp)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        best_accuracy = 0.
        predict_round = 0

        for epoch in range(num_epochs):
            indices = np.arange(len(train_X))
            np.random.shuffle(indices)
            train_y = np.array(train_y)[indices]
            train_X = np.array(train_X)[indices]
            
            for batch in range(nb_batch_per_epoch):
                idx_min = batch * batch_size
                idx_max = min((batch+1) * batch_size, len(train_X)-1)
                print(idx_min, idx_max)
                x1 = train_X[idx_min:idx_max, 3]
                print(type(x1))
                print(np.shape(x1))
                x2 = train_X[idx_min:idx_max, 4]
                y = train_y[idx_min:idx_max]

                feed_dict = {
                    model.x1: x1,
                    model.x2: x2,
                    model.y: y
                }

                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, model.loss, model.accuracy], 
                    feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}/{}, loss {:g}, acc {:g}".format(time_str, step, num_epochs*nb_batch_per_epoch, loss, accuracy))

                current_step = tf.train.global_step(sess, global_step)

                if current_step % evaluate_every == 0:
                    predict_round += 1
                    print("\nEvaluation round %d:" % (predict_round))
                    
                    indices = np.arange(len(test_X))
                    np.random.shuffle(indices)
                    test_X = np.array(test_X)[indices]
                    test_y = np.array(test_y)[indices]

                    x1 = test_X[:test_batch_size, 3]
                    print(type(x1))
                    print(np.shape(x1))
                    x2 = test_X[:test_batch_size, 4]
                    y = test_y[:test_batch_size]

                    feed_dict = {
                        model.x1: x1,
                        model.x2: x2,
                        model.y: y
                    }

                    accuracy = sess.run(model.accuracy, feed_dict=feed_dict)

                    if accuracy >= best_accuracy:
                        best_accuracy = accuracy
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))