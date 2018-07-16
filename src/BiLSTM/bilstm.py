import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import time
import os
import parameter

class BiLSTM():
    def __init__(self):
        #basic parameters

        self.embedding_size = parameter.EMBEDDING_SIZE
        self.class_num = parameter.CLASS_NUM
        self.hidden_units_num = parameter.HIDDEN_UNITS_NUM
        self.hidden_units_num2=parameter.HIDDEN_UNITS_NUM2
        self.layer_num = parameter.LAYER_NUM

        self.vocab_size=parameter.VOCAB_SIZE


    #新建一个权重,initializer这里可以根据需求修改
    def get_weights_variable(self,shape,regularizer):
        weights=tf.get_variable(
            name="weights",
            shape=shape,
            dtype=tf.float32,
            initializer=tf.initializers.truncated_normal(stddev=0.1)
        )
        if regularizer!=None:
            tf.add_to_collection(name="regularized",value=regularizer(weights))
        return weights

    # 全连接层,封装为pytorch API风格,简洁
    def linear(self, variable_scope, X, weights_shape, regularizer):
        with tf.variable_scope(variable_scope):
            weights = self.get_weights_variable(shape=weights_shape,regularizer=regularizer)

            biases = tf.get_variable(
                name="biases",
                shape=(weights_shape[1],),
                dtype=tf.float32,
                initializer=tf.initializers.constant()
            )
            logits = tf.matmul(X, weights) + biases
            return logits

    #前向运算
    #这里的输入X其实是train.py传入的一个placeholder,形状为[None, self.max_sentence_size]
    def forward(self,X,regularizer):
        #字embeddings
        embeddings = tf.Variable(
            initial_value=tf.zeros(shape=(self.vocab_size, self.embedding_size), dtype=tf.float32),
            name="embeddings"
        )

        #把输入扩展为字embedding之后的形式.作为LSTM的真正输入
        inputs = tf.nn.embedding_lookup(params=embeddings, ids=X, name="embeded_input")

        # bisltm
        # forward part
        lstm_forward1 = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
        lstm_forward2 = rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
        lstm_forward = rnn.MultiRNNCell(cells=[lstm_forward1, lstm_forward2])

        # backward part
        lstm_backward1 = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
        lstm_backward2 = rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
        lstm_backward = rnn.MultiRNNCell(cells=[lstm_backward1, lstm_backward2])

        # result
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_forward,cell_bw=lstm_backward,
            inputs=inputs,dtype=tf.float32
        )

        outputs_forward = outputs[0];  # shape of h is [batch_size, max_time, cell_fw.output_size]
        outputs_backward = outputs[1]  # shape of h is [batch_size, max_time, cell_bw.output_size]

        # concat togeter(forward information and backward information)
        # shape of h is [batch_size, max_time, cell_fw.output_size*2]
        h = tf.concat(values=[outputs_forward, outputs_backward], axis=2, name="h")

        # reshape,new shape is [batch_size*max_time, cell_fw.output_size*2]
        h = tf.reshape(tensor=h, shape=[-1, 2 * self.hidden_units_num2], name="h_reshaped")

        # fully connect layer
        logits = self.linear(
            variable_scope="FC1",
            X=h,
            weights_shape=(2 * self.hidden_units_num2, self.class_num),
            regularizer=regularizer
        )
        return logits               # shape of logits:[batch_size*max_time, 5]