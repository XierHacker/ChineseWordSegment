import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf
import time
import parameter
from sklearn.metrics import accuracy_score

class BiLSTM_CRF():
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
            #crf 格式要求
            self.sequence_lengths=tf.placeholder(
                    dtype=tf.int32,
                    shape=(None,),
                    name="sequence_lengths"
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
            # shape of outputs_forward is [batch_size, max_time, cell_fw.output_size]
            outputs_forward = outputs[0]; outputs_backward = outputs[1]

            #concat togeter(forward information and backward information)
            # shape of h is [batch_size, max_time, cell_fw.output_size*2]
            h=tf.concat(values=[outputs_forward,outputs_backward],axis=2,name="h")

            #reshape,new shape is [batch_size*max_time, cell_fw.output_size*2]
            h=tf.reshape(tensor=h,shape=[-1,2*self.hidden_units_num2],name="h_reshaped")

            #fully connect layer
            w=tf.Variable(
                initial_value=tf.random_normal(shape=(2*self.hidden_units_num2,self.class_num)),
                name="weights"
            )
            b=tf.Variable(
                initial_value=tf.random_normal(shape=(self.class_num,)),
                name="bias"
            )
            logits=tf.matmul(h,w)+b     #shape of logits:[batch_size*max_time, 5]
            logits_normal=tf.reshape(
                    tensor=logits,
                    shape=(-1,self.max_sentence_size,self.class_num),
                    name="logits_normal"
            )                           #change logits to the shape of [batch_size,max_sentence_size,5]

            #loss
            log_likelihood,transition_params=crf.crf_log_likelihood(
                inputs=logits_normal,
                tag_indices=self.y_p,
                sequence_lengths=self.sequence_lengths
            )
            self.trans_matrix=tf.Variable(initial_value=transition_params,name="trans_matrix")
            self.loss = tf.reduce_mean(-log_likelihood,name="loss")

            # decode
            # shape of potentials=[batch_size, max_seq_len, num_tags]
            # shape of decode_tags is [batch_size, max_seq_len]
            decode_tags,best_score=crf.crf_decode(
                potentials=logits_normal,
                transition_params=transition_params,
                sequence_length=self.sequence_lengths
            )

            # correct_prediction
            correct_prediction = tf.equal(tf.reshape(decode_tags,[-1]),tf.reshape(self.y_p, [-1]))

            # accuracy
            self.accuracy = tf.reduce_mean(
                input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32),
                name="accuracy"
            )

            #optimizer
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            self.init_op=tf.global_variables_initializer()


        #------------------------------------Session-----------------------------------------
        with self.session as sess:
            print("Training Start")
            sess.run(self.init_op)                      #initialize all variables

            train_Size = X_train.shape[0]
            validation_Size = X_validation.shape[0]
            best_validation_accuracy = 0

            #crf函数用到
            sequence_length_train =np.full(
                shape=(self.batch_size,),
                fill_value=self.max_sentence_size
            )
            sequence_length_valid = np.full(
                shape=(X_validation.shape[0],),
                fill_value=self.max_sentence_size
            )

            for epoch in range(1,self.max_epoch+1):
                print("Epoch:", epoch)
                start_time=time.time()                  #time evaluation
                train_losses = [];  train_accus = []    # training loss/accuracy in every mini-batch

                # mini batch
                for i in range(0, (train_Size // self.batch_size)):
                    _, train_loss, train_accuracy = sess.run(
                        fetches=[self.optimizer, self.loss, self.accuracy],
                        feed_dict={
                            self.X_p: X_train[i * self.batch_size:(i + 1) * self.batch_size],
                            self.y_p: y_train[i * self.batch_size:(i + 1) * self.batch_size],
                            self.sequence_lengths: sequence_length_train
                        }
                    )
                    # add to list
                    train_losses.append(train_loss)
                    train_accus.append(train_accuracy)

                    # print training infomation
                    if (print_log):
                        self.showInfo(print_log,loss=train_loss,accuracy=train_accuracy)

                duration=round((time.time()-start_time)/60,2)   #spend time in an epoch
                validation_loss, validation_accuracy = sess.run(
                    fetches=[self.loss, self.accuracy],
                    feed_dict={
                            self.X_p: X_validation,
                            self.y_p: y_validation,
                            self.sequence_lengths:sequence_length_valid
                    }
                )
                #show train and validation info
                print("Epoch ",epoch," finished.","spend ",duration," mins")
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
                    if not os.path.exists("./models/" + name + "/bilstm_crf"):
                        os.mkdir(path="./models/" + name + "/bilstm_crf")
                    # create saver
                    saver = tf.train.Saver()
                    saver.save(sess, "./models/" + name + "/bilstm_crf/my-model-10000")
                    # Generates MetaGraphDef.
                    saver.export_meta_graph("./models/" + name + "/bilstm_crf/my-model-10000.meta")


    #display loss and accuracy information
    def showInfo(self,print_log=False,loss=None,accuracy=None,train_losses=None,train_accus=None):
        if print_log:
            print("----training loss  : ", loss)
            print("----train accuracy : ", accuracy)
            print()
        else:
            print("----average training loss        : ", sum(train_losses) / len(train_losses))
            print("----average training accuracy    : ", sum(train_accus) / len(train_accus))
            print("----average validation loss      : ", loss)
            print("----average validation accuracy  : ", accuracy)

    #返回预测的结果或者准确率,y not None的时候返回准确率,y ==None的时候返回预测值
    def pred(self,name,X,y=None,):
        start_time = time.time()
        if y is None:
            with self.session as sess:
                viterbi_sequences=[]
                #viterbi_scores=[]
                #crf need
                sequence_lengths = np.full(shape=(X.shape[0],), fill_value=self.max_sentence_size)

                # restore model
                new_saver = tf.train.import_meta_graph(
                    meta_graph_or_file="./models/" + name + "/bilstm_crf/my-model-10000.meta",
                    clear_devices=True
                )
                new_saver.restore(sess, "./models/" + name + "/bilstm_crf/my-model-10000")
                # get default graph
                graph = tf.get_default_graph()
                # get opration from the graph
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                ligits_normal=graph.get_operation_by_name("logits_normal").outputs[0]
                trans_matrix=graph.get_operation_by_name("trans_matrix").outputs[0]

                logits,trans = sess.run(fetches=[logits_normal,trans_matrix],feed_dict={X_p: X})

                for i in range(X.shape[0]):
                    viterbi_sequence,viterbi_score=crf.viterbi_decode(score=logits[i],transition_params=trans)
                    viterbi_sequences.append(viterbi_sequence)
                    #viterbi_scores.append(viterbi_score)
                print("this operation spends ",round((time.time()-start_time)/60,2)," mins")
                return viterbi_sequences
        else:
            with self.session as sess:
                viterbi_sequences = []
                # viterbi_scores=[]
                # crf need
                sequence_lengths = np.full(shape=(X.shape[0],), fill_value=self.max_sentence_size)

                # restore model
                new_saver = tf.train.import_meta_graph(
                    meta_graph_or_file="./models/" + name + "/bilstm_crf/my-model-10000.meta",
                    clear_devices=True
                )
                new_saver.restore(sess, "./models/" + name + "/bilstm_crf/my-model-10000")
                # get default graph
                graph = tf.get_default_graph()
                # get opration from the graph
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                ligits_normal = graph.get_operation_by_name("logits_normal").outputs[0]
                trans_matrix = graph.get_operation_by_name("trans_matrix").outputs[0]

                logits, trans = sess.run(fetches=[logits_normal, trans_matrix], feed_dict={X_p: X})
                for i in range(X.shape[0]):
                    viterbi_sequence, viterbi_score = crf.viterbi_decode(score=logits[i], transition_params=trans)
                    viterbi_sequences.append(viterbi_sequence)
                    # viterbi_scores.append(viterbi_score)
                viterbi_sequences=np.array(viterbi_sequences)
                accuracy=accuracy_score(y_true=np.reshape(y,[-1]),y_pred=np.reshape(viterbi_sequences,[-1]))
                print("this operation spends ", round((time.time() - start_time) / 60, 2), " mins")
                return accuracy

