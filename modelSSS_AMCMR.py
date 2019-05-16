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
            self.class_weights = tf.placeholder(tf.float32, [None], name="class_weights")

        with tf.name_scope('weights'):
            self.weights = {
                'q_1_to_2': tf.Variable(tf.random_uniform([2 * embedding_dim, self.hidden_size], -0.01, 0.01)),

                'p_1_to_2': tf.Variable(tf.random_uniform([self.hidden_size, 1], -0.01, 0.01)),

                'inter_attention': tf.Variable(tf.random_uniform([4*self.embedding_dim, 2*self.embedding_dim], -0.01, 0.01)),
                
                'lstm': tf.Variable(tf.random_uniform([3*self.hidden_size, self.hidden_size], -0.01, 0.01)),

                'attention_concat' : tf.Variable(tf.random_uniform([self.hidden_size, 1], -0.01, 0.01)),
                'attention_dot' : tf.Variable(tf.random_uniform([self.hidden_size, 1], -0.01, 0.01)),
                'attention_minus' : tf.Variable(tf.random_uniform([self.hidden_size, 1], -0.01, 0.01)),


                'z': tf.Variable(tf.random_uniform([2*self.embedding_dim+self.hidden_size, self.hidden_size], -0.01, 0.01)),

                'f': tf.Variable(tf.random_uniform([self.hidden_size, self.class_num], -0.01, 0.01)),
            }

        with tf.name_scope('biases'):
            self.biases = {
                'q_1_to_2': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),

                'inter_attention': tf.Variable(tf.random_uniform([2*self.embedding_dim], -0.01, 0.01)),

                'p_1_to_2': tf.Variable(tf.random_uniform([], -0.01, 0.01)),
                
                'lstm': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),

                'attention_concat': tf.Variable(tf.random_uniform([1], -0.01, 0.01)),
                'attention_dot': tf.Variable(tf.random_uniform([1], -0.01, 0.01)),
                'attention_minus': tf.Variable(tf.random_uniform([1], -0.01, 0.01)),

                'z': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),

                'f': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
            }

    def inter_attention(self):
        
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

        s_1_to_2 = tf.nn.relu(tf.matmul(concat, self.weights['q_1_to_2']) + self.biases['q_1_to_2'])
        s_1_to_2 = tf.matmul(s_1_to_2, self.weights['p_1_to_2']) + self.biases['p_1_to_2']
        s_1_to_2 = tf.reshape(s_1_to_2, [-1, self.max_sen_len, self.max_sen_len])

        a_1 = tf.reshape(tf.nn.softmax(tf.reduce_max(s_1_to_2, axis=-1), axis=-1), [-1, 1, self.max_sen_len])

        self.v_a_1_to_2 = tf.reshape(tf.matmul(a_1, self.x1), [-1, self.embedding_dim])

        a_2 = tf.reshape(tf.nn.softmax(tf.reduce_max(tf.transpose(s_1_to_2, perm=[0, 2, 1]), axis=-1), axis=-1), [-1, 1, self.max_sen_len])

        self.v_a_2_to_1 = tf.reshape(tf.matmul(a_2, self.x2), [-1, self.embedding_dim])

        self.v_a_concat = tf.concat([self.v_a_1_to_2, self.v_a_2_to_1], axis=-1) # (-1, 2*embedding_dim)
        self.v_a_dot = tf.multiply(self.v_a_1_to_2, self.v_a_2_to_1) # (-1, embedding_dim)
        self.v_a_minus = tf.math.subtract(self.v_a_1_to_2, self.v_a_2_to_1) # (-1, embedding_dim)

        self.v_a = tf.nn.relu(tf.matmul(tf.concat([self.v_a_concat, self.v_a_dot, self.v_a_minus], axis=-1), self.weights['inter_attention'])) + self.biases['inter_attention']

    def attention_LSTM_concat(self):

        lstm_cell = tf.keras.layers.LSTMCell(self.hidden_size)
        LSTM_layer = tf.keras.layers.RNN(lstm_cell, return_sequences=True) # -1, 30, 100
        self.v_c_concat = LSTM_layer(tf.concat([self.x1, self.x2], axis=1))

        v_c_reshape = tf.reshape(self.v_c_concat, [-1, self.hidden_size])
        alpha = tf.matmul(v_c_reshape, self.weights['attention_concat']) + self.biases['attention_concat'] # (-1*2*30, 100) ** (-1, 100, 1) -> (-1, 2*30)
        alpha = tf.nn.softmax(alpha, axis=-1)
        alpha = tf.reshape(alpha, [-1, 1, 2*self.max_sen_len]) 

        self.h_concat = tf.tanh(tf.matmul(alpha, self.v_c)) # (-1, 1, 2*30) ** (-1, 2*30, 100) -> (-1, 100)
        self.h_concat = tf.reshape(self.h_concat, [-1, self.hidden_size])

    def attention_LSTM_dot(self):

        lstm_cell = tf.keras.layers.LSTMCell(self.hidden_size)
        LSTM_layer = tf.keras.layers.RNN(lstm_cell, return_sequences=True) # -1, 30, 100
        self.v_c_dot = LSTM_layer(tf.multiply(self.x1, self.x2))

        v_c_reshape = tf.reshape(self.v_c_dot, [-1, self.hidden_size])
        alpha = tf.matmul(v_c_reshape, self.weights['attention_dot']) + self.biases['attention_dot'] # (-1*2*30, 100) ** (-1, 100, 1) -> (-1, 2*30)
        alpha = tf.nn.softmax(alpha, axis=-1)
        alpha = tf.reshape(alpha, [-1, 1, 2*self.max_sen_len]) 

        self.h_dot = tf.tanh(tf.matmul(alpha, self.v_c)) # (-1, 1, 2*30) ** (-1, 2*30, 100) -> (-1, 100)
        self.h_dot = tf.reshape(self.h_dot, [-1, self.hidden_size])

    def attention_LSTM_minus(self):

        lstm_cell = tf.keras.layers.LSTMCell(self.hidden_size)
        LSTM_layer = tf.keras.layers.RNN(lstm_cell, return_sequences=True) # -1, 30, 100
        self.v_c_minus = LSTM_layer(tf.math.subtract(self.x1, self.x2))

        v_c_reshape = tf.reshape(self.v_c_minus, [-1, self.hidden_size])
        alpha = tf.matmul(v_c_reshape, self.weights['attention_minus']) + self.biases['attention_minus'] # (-1*2*30, 100) ** (-1, 100, 1) -> (-1, 2*30)
        alpha = tf.nn.softmax(alpha, axis=-1)
        alpha = tf.reshape(alpha, [-1, 1, 2*self.max_sen_len]) 

        self.h_minus = tf.tanh(tf.matmul(alpha, self.v_c)) # (-1, 1, 2*30) ** (-1, 2*30, 100) -> (-1, 100)
        self.h_minus = tf.reshape(self.h_minus, [-1, self.hidden_size])

    def attention_multi_LSTM(self):

        self.v_c = tf.nn.relu(tf.matmul(tf.concat([self.h_concat, self.h_dot, self.h_minus], axis=-1), self.weights['lstm'])) + self.biases['lstm']

    def prediction(self):

        v = tf.concat([self.v_a, self.v_c], -1)
        v = tf.nn.relu(tf.matmul(v, self.weights['z']) + self.biases['z'])

        self.scores = tf.nn.softmax((tf.matmul(v, self.weights['f']) + self.biases['f']), axis=-1)

        self.predictions = tf.argmax(self.scores, -1, name="predictions")

    def build_model(self):

        self.inter_attention()
        self.attention_LSTM_concat()
        self.attention_LSTM_dot()
        self.attention_LSTM_minus()
        self.attention_multi_LSTM()
        self.prediction()
        
        with tf.name_scope("loss"):

            losses = tf.losses.sparse_softmax_cross_entropy(
                labels=tf.argmax(self.y, -1),
                logits=self.scores,
                weights=self.class_weights)

            '''losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits = self.scores,
                labels = self.y
            )'''

            self.loss = tf.reduce_mean(losses)
            
        with tf.name_scope("metrics"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            #self.accuracy = (tf.reduce_sum(self.class_weights*tf.cast(correct_predictions, "float")) / tf.reduce_sum(self.class_weights))
            self.c_matrix = tf.confusion_matrix(labels = tf.argmax(self.y, -1), predictions = self.predictions, name="c_matrix")