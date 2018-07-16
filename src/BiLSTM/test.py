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


MODEL_SAVING_PATH="./saved_models/lstm.ckpt-2"


def test(X_valid, y_valid):
    # data placeholder
    X_p = tf.placeholder(dtype=tf.int32,shape=(None, max_sentence_size),name="input_p")
    y_p = tf.placeholder(dtype=tf.int32,shape=(None, max_sentence_size),name="label_p")

    #测试阶段就不使用regularizer了
    regularizer = None
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

    init_op = tf.global_variables_initializer()

    # Saver class
    saver = tf.train.Saver()

    # ------------------------------------Session-----------------------------------------
    with tf.Session() as sess:
        sess.run(init_op)  # initialize all variables
        valid_size = X_valid.shape[0];

        # restore
        saver.restore(sess=sess, save_path=MODEL_SAVING_PATH)

        #prediction
        start_time = time.time()
        valid_loss, valid_accuracy = sess.run(fetches=[loss, accuracy],feed_dict={X_p: X_valid,y_p: y_valid})
        end_time = time.time()
        print("spend: ", (end_time - start_time) / 60, " mins")
        print("valid loss:", valid_loss)
        print("vlaid accuracy:", valid_accuracy)


if __name__=="__main__":
    # 读数据
    print("Loading Data....")
    df_validation = pd.read_pickle(path="../../dataset/msr/summary_validation.pkl")
    X_validation = np.asarray(list(df_validation['X'].values))
    y_validation = np.asarray(list(df_validation['y'].values))
    print("Loading Done!")

    print("Test Start")
    test(X_validation,y_validation)