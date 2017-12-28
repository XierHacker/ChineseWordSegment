import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import time
import parameter

class BiLSTM():
    def __init__(self):
        # basic environment
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        #basic parameters
        self.learning_rate = parameter.LEARNING_RATE
        self.max_epoch = parameter.MAX_EPOCH
        self.embedding_size = parameter.EMBEDDING_SIZE
        self.class_num = parameter.CLASS_NUM
        self.hidden_units_num = parameter.HIDDEN_UNITS_NUM
        self.hidden_units_num2=parameter.HIDDEN_UNITS_NUM2
        self.layer_num = parameter.LAYER_NUM
        self.max_sentence_size=parameter.MAX_SENTENCE_SIZE
        self.vocab_size=parameter.VOCAB_SIZE
        self.batch_size=parameter.BATCH_SIZE

    def fit(self,X_train,y_train,X_validation,y_validation,name,print_log=True):
        #---------------------------------------forward computation--------------------------------------------#
        #---------------------------------------define graph---------------------------------------------#
        with self.graph.as_default():
            # data place holder
            self.X_p = tf.placeholder(
                    dtype=tf.int32,
                    shape=(None, self.max_sentence_size),
                    name="input_placeholder"
            )
            self.y_p = tf.placeholder(
                    dtype=tf.int32,
                    shape=(None,self.max_sentence_size),
                    name="label_placeholder"
            )

            #embeddings
            embeddings=tf.Variable(
                initial_value=tf.zeros(shape=(self.vocab_size,self.embedding_size),dtype=tf.float32),
                name="embeddings"
            )
            #embeded inputs
            inputs=tf.nn.embedding_lookup(params=embeddings,ids=self.X_p,name="embeded_input")

            #bisltm
            #forward part
            lstm_forward1=rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            lstm_forward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            lstm_forward=rnn.MultiRNNCell(cells=[lstm_forward1,lstm_forward2])

            #backward part
            lstm_backward1=rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            lstm_backward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            lstm_backward=rnn.MultiRNNCell(cells=[lstm_backward1,lstm_backward2])

            #result
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_forward,
                cell_bw=lstm_backward,
                inputs=inputs,
                dtype=tf.float32
            )
            outputs_forward = outputs[0];   #shape of h is [batch_size, max_time, cell_fw.output_size]
            outputs_backward = outputs[1]   #shape of h is [batch_size, max_time, cell_bw.output_size]

            #concat togeter(forward information and backward information)
            # shape of h is [batch_size, max_time, cell_fw.output_size*2]
            h=tf.concat(values=[outputs_forward,outputs_backward],axis=2,name="h")
            print("h.shape:",h.shape)

            #reshape,new shape is [batch_size*max_time, cell_fw.output_size*2]
            h=tf.reshape(tensor=h,shape=[-1,2*self.hidden_units_num2],name="h_reshaped")
            print("h.shape",h.shape)

            #fully connect layer
            w=tf.Variable(
                initial_value=tf.random_normal(shape=(2*self.hidden_units_num2,self.class_num)),
                name="weights"
            )
            b=tf.Variable(
                initial_value=tf.random_normal(shape=(self.class_num,)),
                name="bias"
            )
            logits=tf.matmul(h,w)+b     #shape of logits:
            print("logit.shape",logits.shape)[batch_size*max_time, 5]

            #pred  shape of pred[batch_size*max_time, 1]
            pred=tf.cast(tf.argmax(logits, 1), tf.int32,name="pred")
            print("pred.shape",pred.shape)

            #pred in an normal way,shape is [batch_size, max_time]
            pred_normal=tf.reshape(tensor=pred,shape=(-1,self.max_sentence_size),name="pred_normal")
            print("pred_normal.shape",pred_normal.shape)

            #correct_prediction
            correct_prediction = tf.equal(pred, tf.reshape(self.y_p, [-1]))
            print("correct_prediction.shape:",correct_prediction.shape)

            #accracy
            self.accuracy=tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction,dtype=tf.float32),name="accuracy")

            #loss
            self.loss=tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(self.y_p,shape=[-1]),logits=logits)

            #optimizer
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            self.init_op=tf.global_variables_initializer()


        #------------------------------------Session-----------------------------------------
        with self.session as sess:
            sess.run(self.init_op)          #initialize all variables
            best_validation_accuracy=0

            train_Size = X_train.shape[0]
            validation_Size = X_validation.shape[0]

            print("Training Start")
            for epoch in range(1,self.max_epoch+1):
                start_time=time.time()      #performance evaluation
                print("Epoch:", epoch)
                train_losses = [];          # training loss in every mini-batch
                train_accus = [];           # training accuracy in every mini-batch

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
                        print("Mini-Batch: ", i * self.batch_size, "~", (i + 1) * self.batch_size, "of epoch:", epoch)
                        print("     training loss   :  ", train_loss)
                        print("     train accuracy  :  ", train_accuracy)
                        print()

                    # add to list
                    train_losses.append(train_loss)
                    train_accus.append(train_accuracy)

                duration=round((time.time()-start_time)/60,2)   #spend time in an epoch

                validation_loss, validation_accuracy = sess.run(
                    fetches=[self.loss, self.accuracy],
                    feed_dict={
                                self.X_p: X_validation,
                                self.y_p: y_validation
                    }
                )

                print("Epoch ",epoch," finished.","spend ",duration," mins")
                print("----average training loss        : ", sum(train_losses) / len(train_losses))
                print("----average training accuracy    : ", sum(train_accus) / len(train_accus))
                print("----average validation loss      : ", validation_loss)
                print("----average validation accuracy  : ", validation_accuracy)

                # when we get a new best validation accuracy,we store the model
                if best_validation_accuracy < validation_accuracy:
                    print("we got a new best accuracy on validation set! saving model!")
                    saver = tf.train.Saver()
                    saver.save(sess, "./models/"+name+"/my-model-10000")
                    # Generates MetaGraphDef.
                    saver.export_meta_graph("./models/"+name+"/my-model-10000.meta")


    #返回预测的结果或者准确率,y not None的时候返回准确率,y ==None的时候返回预测值
    def pred(self,name,X,y=None,):
        start_time = time.time()
        if y is None:

            with self.session as sess:
                # restore model
                new_saver = tf.train.import_meta_graph("./models/"+name+"/my-model-10000.meta", clear_devices=True)
                new_saver.restore(sess, "./models/"+name+"/my-model-10000")

                graph = tf.get_default_graph()
                # get opration from the graph
                pred = graph.get_operation_by_name("pred").outputs[0]
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                pred = sess.run(fetches=pred, feed_dict={X_p: X})
                #compute time
                duration=round((time.time()-start_time)/60,2)
                print("this operation spends ",duration," mins")
                return pred
        else:
            with self.session as sess:
                # restore model
                new_saver = tf.train.import_meta_graph("./models/"+name+"/my-model-10000.meta", clear_devices=True)
                new_saver.restore(sess, "./models/"+name+"/my-model-10000")
                graph = tf.get_default_graph()
                # get opration from the graph
                accuracy=graph.get_operation_by_name("accuracy").outputs[0]
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                y_p=graph.get_operation_by_name("label_placeholder").outputs[0]

                accu = sess.run(fetches=accuracy,feed_dict={X_p: X,y_p: y})
                #compute time
                duration = round((time.time() - start_time) / 60, 2)
                print("this operation spends ", duration, " mins")
                return accu


if __name__ =="__main__":
    bilstm=BiLSTM()
    x=[1,2,3,4,5]
    y=[1,2,3]
    bilstm.fit()
