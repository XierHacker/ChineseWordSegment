import os
import sys
import numpy as np
import tensorflow as tf

import bilstm

sys.path.append("..")
#不提示调试信息和警告信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

learning_rate = parameter.LEARNING_RATE
max_epoch = parameter.MAX_EPOCH
batch_size=parameter.BATCH_SIZE
max_sentence_size=parameter.MAX_SENTENCE_SIZE

def train():
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
    with self.session as sess:
        print("Training Start")
        sess.run(self.init_op)  # initialize all variables

        train_Size = X_train.shape[0];
        validation_Size = X_validation.shape[0]
        best_validation_accuracy = 0  # best validation accuracy in training process

        for epoch in range(1, self.max_epoch + 1):
            print("Epoch:", epoch)
            start_time = time.time()  # time evaluation
            train_losses = [];
            train_accus = []  # training loss/accuracy in every mini-batch
            # mini batch
            for i in range(0, (train_Size // self.batch_size)):
                _, train_loss, train_accuracy = sess.run(
                    fetches=[self.optimizer, self.loss, self.accuracy],
                    feed_dict={
                        self.X_p: X_train[i * self.batch_size:(i + 1) * self.batch_size],
                        self.y_p: y_train[i * self.batch_size:(i + 1) * self.batch_size]
                    }
                )
                # print training infomation
                if (print_log):
                    self.showInfo(print_log, train_loss, train_accuracy)
                # add to list
                train_losses.append(train_loss);
                train_accus.append(train_accuracy)

            validation_loss, validation_accuracy = sess.run(
                fetches=[self.loss, self.accuracy],
                feed_dict={self.X_p: X_validation, self.y_p: y_validation}
            )
            print("Epoch ", epoch, " finished.", "spend ", round((time.time() - start_time) / 60, 2), " mins")
            self.showInfo(False, validation_loss, validation_accuracy, train_losses, train_accus)
            # when we get a new best validation accuracy,we store the model
            if best_validation_accuracy < validation_accuracy:
                print("New Best Accuracy ", validation_accuracy, " On Validation set! ")
                print("Saving Models......")
                # exist ./models folder?
                if not os.path.exists("./models/"):
                    os.mkdir(path="./models/")
                if not os.path.exists("./models/" + name):
                    os.mkdir(path="./models/" + name)
                if not os.path.exists("./models/" + name + "/bilstm"):
                    os.mkdir(path="./models/" + name + "/bilstm")
                # create saver
                saver = tf.train.Saver()
                saver.save(sess, "./models/" + name + "/bilstm/my-model-10000")
                # Generates MetaGraphDef.
                saver.export_meta_graph("./models/" + name + "/bilstm/my-model-10000.meta")




if __name__=="__main__":
    pass


# forward process and training process
def fit(self, X_train, y_train, X_validation, y_validation, name, print_log=True):
    # ---------------------------------------forward computation--------------------------------------------#
    # ---------------------------------------define graph---------------------------------------------#
    with self.graph.as_default():


# 返回预测的结果或者准确率,y not None的时候返回准确率,y ==None的时候返回预测值
def pred(self, name, X, y=None, ):
    start_time = time.time()  # compute time
    if y is None:
        with self.session as sess:
            # restore model
            new_saver = tf.train.import_meta_graph(
                meta_graph_or_file="./models/" + name + "/bilstm/my-model-10000.meta",
                clear_devices=True
            )
            new_saver.restore(sess, "./models/" + name + "/bilstm/my-model-10000")
            # get default graph
            graph = tf.get_default_graph()
            # get opration from the graph
            pred_normal = graph.get_operation_by_name("pred_normal").outputs[0]
            X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
            pred = sess.run(fetches=pred_normal, feed_dict={X_p: X})
            print("this operation spends ", round((time.time() - start_time) / 60, 2), " mins")
            return pred
    else:
        with self.session as sess:
            # restore model
            new_saver = tf.train.import_meta_graph(
                meta_graph_or_file="./models/" + name + "/bilstm/my-model-10000.meta",
                clear_devices=True
            )
            new_saver.restore(sess, "./models/" + name + "/bilstm/my-model-10000")
            graph = tf.get_default_graph()
            # get opration from the graph
            accuracy = graph.get_operation_by_name("accuracy").outputs[0]
            X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
            y_p = graph.get_operation_by_name("label_placeholder").outputs[0]
            # forward and get the results
            accu = sess.run(fetches=accuracy, feed_dict={X_p: X, y_p: y})
            print("this operation spends ", round((time.time() - start_time) / 60, 2), " mins")
            return accu


# 把一个句子转成一个分词后的结构
def infer(self, sentence, name):
    pass
