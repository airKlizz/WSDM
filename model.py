import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self, max_sen_len, class_num, embedding_dim, hidden_size):

        self.max_sen_len = max_sen_len
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.hidden_size = hidden_size

        with tf.name_scope('input'):
            self.x1 = tf.placeholder(tf.float32, [None, self.max_sen_len, self.embedding_dim], name="x1")
            self.x2 = tf.placeholder(tf.float32, [None, self.max_sen_len, self.embedding_dim], name="x2")
            self.y = tf.placeholder(tf.float32, [None, self.class_num], name="y")

        with tf.name_scope('weights'):
            self.weights = {
                'q_1_to_2': tf.Variable(tf.random_uniform([2 * embedding_dim, self.hidden_size], -0.01, 0.01)),
                'q_2_to_1': tf.Variable(tf.random_uniform([2 * embedding_dim, self.hidden_size], -0.01, 0.01)),

                'p_1_to_2': tf.Variable(tf.random_uniform([self.hidden_size, 1], -0.01, 0.01)),
                'p_2_to_1': tf.Variable(tf.random_uniform([self.hidden_size, 1], -0.01, 0.01)),

                'z_1': tf.Variable(tf.random_uniform([self.hidden_size+self.max_sen_len, self.hidden_size], -0.01, 0.01)),
                'z_2': tf.Variable(tf.random_uniform([self.hidden_size+self.max_sen_len, self.hidden_size], -0.01, 0.01)),

                'f': tf.Variable(tf.random_uniform([2*self.hidden_size, self.class_num], -0.01, 0.01)),
            }

        with tf.name_scope('biases'):
            self.biases = {
                'q_1_to_2': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),
                'q_2_to_1': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),

                'p_1_to_2': tf.Variable(tf.random_uniform([1], -0.01, 0.01)),
                'p_2_to_1': tf.Variable(tf.random_uniform([1], -0.01, 0.01)),

                'z_1': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),
                'z_2': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),

                'f': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
            }

    def inter_attention_1_to_2(self):
        
        x1_shape = tf.shape(self.x1)
        x2_shape = tf.shape(self.x2)

        x1_reshape = tf.reshape(self.x1, [-1, self.embedding_dim, 1])
        ones = tf.ones([x1_shape[0]*self.max_sen_len, 1, self.max_sen_len])
        x1_increase = tf.matmul(x1_reshape, ones)
        x1_increase = tf.transpose(x1_increase, perm=[0, 2, 1])
        x1_increase = tf.reshape(x1_increase, [-1, self.max_sen_len*self.max_sen_len, self.embedding_dim])

        x2_reshape = tf.reshape(self.x2, [-1, self.embedding_dim, 1])
        ones = tf.ones([x2_shape[0]*self.max_sen_len, 1, self.max_sen_len])
        x2_increase = tf.matmul(x2_reshape, ones)
        x2_increase = tf.transpose(x2_increase, perm=[0, 2, 1])
        x2_increase = tf.reshape(x2_increase, [-1, self.max_sen_len, self.max_sen_len, self.embedding_dim])
        x2_increase = tf.transpose(x2_increase, perm=[0, 2, 1, 3])
        x2_increase = tf.reshape(x2_increase, [-1, self.max_sen_len*self.max_sen_len, self.embedding_dim])

        concat = tf.concat([x1_increase, x2_increase], axis=-1)
        concat = tf.reshape(concat, [-1, 2*self.embedding_dim])

        s_1_to_2 = tf.matmul(concat, self.weights['q_1_to_2']) + self.biases['q_1_to_2']
        s_1_to_2 = tf.matmul(s_1_to_2, self.weights['p_1_to_2']) + self.biases['p_1_to_2']
        s_1_to_2 = tf.reshape(s_1_to_2, [-1, self.max_sen_len, self.max_sen_len])

        a = tf.reshape(tf.nn.softmax(tf.reduce_max(s_1_to_2, axis=-1), axis=-1), [-1, 1, self.max_sen_len])

        self.v_a_1_to_2 = tf.reshape(tf.matmul(a, self.x1), [-1, self.embedding_dim])

    def inter_attention_2_to_1(self):
            
        x1_shape = tf.shape(self.x1)
        x2_shape = tf.shape(self.x2)

        x2_reshape = tf.reshape(self.x2, [-1, self.embedding_dim, 1])
        ones = tf.ones([x2_shape[0]*self.max_sen_len, 1, self.max_sen_len])
        x2_increase = tf.matmul(x2_reshape, ones)
        x2_increase = tf.transpose(x2_increase, perm=[0, 2, 1])
        x2_increase = tf.reshape(x2_increase, [-1, self.max_sen_len*self.max_sen_len, self.embedding_dim])

        x1_reshape = tf.reshape(self.x1, [-1, self.embedding_dim, 1])
        ones = tf.ones([x1_shape[0]*self.max_sen_len, 1, self.max_sen_len])
        x1_increase = tf.matmul(x1_reshape, ones)
        x1_increase = tf.transpose(x1_increase, perm=[0, 2, 1])
        x1_increase = tf.reshape(x1_increase, [-1, self.max_sen_len, self.max_sen_len, self.embedding_dim])
        x1_increase = tf.transpose(x1_increase, perm=[0, 2, 1, 3])
        x1_increase = tf.reshape(x1_increase, [-1, self.max_sen_len*self.max_sen_len, self.embedding_dim])

        concat = tf.concat([x2_increase, x1_increase], axis=-1)
        concat = tf.reshape(concat, [-1, 2*self.embedding_dim])

        s_2_to_1 = tf.matmul(concat, self.weights['q_2_to_1']) + self.biases['q_2_to_1']
        s_2_to_1 = tf.matmul(s_2_to_1, self.weights['p_2_to_1']) + self.biases['p_2_to_1']
        s_2_to_1 = tf.reshape(s_2_to_1, [-1, self.max_sen_len, self.max_sen_len])

        a = tf.reshape(tf.nn.softmax(tf.reduce_max(s_2_to_1, axis=-1), axis=-1), [-1, 1, self.max_sen_len])

        self.v_a_2_to_1 = tf.reshape(tf.matmul(a, self.x2), [-1, self.embedding_dim])

    def long_short_memory_encoder_1(self):

        x1_reshape = tf.reshape(self.x1, [-1, self.max_sen_len, self.embedding_dim])

        lstm_cell = tf.keras.layers.LSTMCell(self.hidden_size)
        LSTM_layer = tf.keras.layers.RNN(lstm_cell)
        self.v_c_1 = LSTM_layer(x1_reshape)

    def long_short_memory_encoder_2(self):
        '''
        x2_shape = tf.shape(self.x2)

        x2_reshape = tf.reshape(self.x2, [-1, self.max_sen_len, self.embedding_dim])

        LSTM_layer = tf.keras.layers.LSTMCell(self.hidden_size)
        h = tf.transpose(LSTM_layer.apply(x2_reshape), perm=[0, 2, 1])

        zeros = np.zeros([x2_shape[0], self.max_sen_len, self.embedding_dim])
        for i in range(x2_shape[0]):
            zeros[i, -1, -1] = 1

        self.v_c_2 = tf.transpose(tf.reduce_sum(tf.matmul(h, zeros), axis=-1), perm=[1, 0])
        '''

        x2_reshape = tf.reshape(self.x2, [-1, self.max_sen_len, self.embedding_dim])

        lstm_cell = tf.keras.layers.LSTMCell(self.hidden_size)
        LSTM_layer = tf.keras.layers.RNN(lstm_cell)
        self.v_c_2 = LSTM_layer(x2_reshape)

    def prediction(self):

        v1 = tf.concat([self.v_a_1_to_2, self.v_c_1], -1)
        v1 = tf.nn.relu(tf.matmul(v1, self.weights['z_1']) + self.biases['z_1'])

        v2 = tf.concat([self.v_a_2_to_1, self.v_c_2], -1)
        v2 = tf.nn.relu(tf.matmul(v2, self.weights['z_2']) + self.biases['z_2'])

        v = tf.concat([v1, v2], -1)
        self.scores = tf.nn.softmax((tf.matmul(v, self.weights['f']) + self.biases['f']), axis=-1)

        self.predictions = tf.arg_max(self.y, -1)

    def build_model(self):

        self.inter_attention_1_to_2()
        self.inter_attention_2_to_1()
        self.long_short_memory_encoder_1()
        self.long_short_memory_encoder_2()
        self.prediction()
        
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits = self.scores,
                labels = self.y
            )
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("metrics"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")