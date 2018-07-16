import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf

import bilstm
import parameter


sys.path.append("../..")
#不提示调试信息和警告信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

learning_rate = parameter.LEARNING_RATE
max_epoch = parameter.MAX_EPOCH
batch_size=parameter.BATCH_SIZE
max_sentence_size=parameter.MAX_SENTENCE_SIZE

def train(X_train, y_train):
    # data placeholder
    X_p = tf.placeholder(dtype=tf.int32,shape=(None, max_sentence_size),name="input_p")
    y_p = tf.placeholder(dtype=tf.int32,shape=(None, max_sentence_size),name="label_p")

    # 使用regularizer控制权重
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    model=bilstm.BiLSTM()
    logits=model.forward(X_p,regularizer)

    # prediction
    # shape of pred[batch_size*max_time, 1]
    pred = tf.cast(tf.argmax(logits, 1), tf.int32, name="pred")
    # pred in an normal way,shape is [batch_size, max_time]
    pred_normal = tf.reshape(tensor=pred,shape=(-1, max_sentence_size),name="pred_normal")

    # correct_prediction
    correct_prediction = tf.equal(pred, tf.reshape(y_p, [-1]))
    # accracy
    accuracy = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32),name="accuracy")
    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y_p, shape=[-1]),logits=logits)
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init_op = tf.global_variables_initializer()

    # Saver class
    saver = tf.train.Saver()

    # ------------------------------------Session-----------------------------------------
    with tf.Session() as sess:
        sess.run(init_op)  # initialize all variables
        train_size = X_train.shape[0];
        for epoch in range(1, max_epoch + 1):
            print("Epoch:", epoch)
            # time evaluation
            start_time = time.time()
            train_losses = [];
            train_accus = []  # training loss/accuracy in every mini-batch
            # mini batch
            for i in range(0, (train_size // batch_size)):
                _, train_loss, train_accuracy = sess.run(
                    fetches=[optimizer, loss, accuracy],
                    feed_dict={
                        X_p: X_train[i * batch_size:(i + 1) * batch_size],
                        y_p: y_train[i * batch_size:(i + 1) * batch_size]
                    }
                )
                # add to list
                train_losses.append(train_loss);
                train_accus.append(train_accuracy)
            end_time = time.time()
            print("spend: ", (end_time - start_time) / 60, " mins")
            print("average train loss:",sum(train_losses)/len(train_losses))
            print("average train accuracy:",sum(train_accus)/len(train_accus))


if __name__=="__main__":
    # 读数据
    print("Loading Data....")
    df_train = pd.read_pickle(path="../../dataset/msr/summary_train.pkl")
    df_validation = pd.read_pickle(path="../../dataset/msr/summary_validation.pkl")

    X_train = np.asarray(list(df_train['X'].values))
    y_train = np.asarray(list(df_train['y'].values))
    print("Loading Done!")

    print("Training Start")
    train(X_train,y_train)




