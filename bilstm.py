import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf
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
            lstm_forward2=rnn.BasicLSTMCell(num_units=self.class_num)
            lstm_forward=rnn.MultiRNNCell(cells=[lstm_forward1,lstm_forward2])

            #backward part
            lstm_backward1=rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            lstm_backward2=rnn.BasicLSTMCell(num_units=self.class_num)
            lstm_backward=rnn.MultiRNNCell(cells=[lstm_backward1,lstm_backward2])

            #result
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_forward,
                cell_bw=lstm_backward,
                inputs=inputs,
                dtype=tf.float32
            )
            outputs_forward = outputs[0]
            outputs_backward = outputs[1]
            #shape of h is [batch_size, max_time, cell_fw.output_size]
            pred = outputs_forward + outputs_backward
            print("pred.shape:",pred.shape)

            pred_2=tf.reshape(tensor=pred,shape=[-1,5],name="pred")
            print("pred_2.shape:",pred_2.shape)

            #correct_prediction
            correct_prediction = tf.equal(tf.cast(tf.argmax(pred_2, 1), tf.int32), tf.reshape(self.y_p, [-1]))
            print("correct_prediction.shape:",correct_prediction.shape)

            #accracy
            self.accuracy=tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction,dtype=tf.float32),name="accuracy")

            #loss
            self.loss=tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(self.y_p,shape=[-1]),logits=pred_2)

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
                    print("we got a new best accuracy on validation set!")
                    saver = tf.train.Saver()
                    saver.save(sess, "./models/"+name+"/my-model-10000")
                    # Generates MetaGraphDef.
                    saver.export_meta_graph("./models/"+name+"/my-model-10000.meta")


    def pred(self,name,X,y=None,):
        '''
        返回预测的结果或者准确率
            y not None的时候返回准确率
            y ==None的时候返回预测值
        :param X:
        :param y:
        :param name:
        :return:
        '''
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
            return accu



if __name__ =="__main__":
    bilstm=BiLSTM()
    x=[1,2,3,4,5]
    y=[1,2,3]
    bilstm.fit()

'''
For Chinese word segmentation.
#import tensorflow as tf

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#from tensorflow.contrib import rnn
#import numpy as np

# ##################### config ######################
decay = 0.85
max_epoch = 5
max_max_epoch = 10
timestep_size = max_len = 32  # 句子长度
vocab_size = 5159  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64  # 字向量长度
class_num = 5
hidden_size = 128  # 隐含层节点数
layer_num = 2  # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置

with tf.variable_scope('embedding'):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def lstm_cell():
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def bi_lstm(X_inputs):
    """build the bi-LSTMs network. Return the y_pred"""
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)

    # ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)

    # ** 2.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    # 下面两部分是等价的
    # **************************************************************
    # ** 把 inputs 处理成 rnn.static_bidirectional_rnn 的要求形式
    # ** 文档说明
    # inputs: A length T list of inputs, each a tensor of shape
    # [batch_size, input_size], or a nested tuple of such elements.
    # *************************************************************
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # inputs.shape = [batchsize, timestep_size, embedding_size]  ->  timestep_size tensor, each_tensor.shape = [batchsize, embedding_size]
    # inputs = tf.unstack(inputs, timestep_size, 1)
    # ** 3.bi-lstm 计算（tf封装）  一般采用下面 static_bidirectional_rnn 函数调用。
    #   但是为了理解计算的细节，所以把后面的这段代码进行展开自己实现了一遍。
    #     try:
    #         outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
    #                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
    #     except Exception: # Old TensorFlow version only returns outputs not states
    #         outputs = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
    #                         initial_state_fw = initial_state_fw, initial_stat output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2])
    # ****************************************e_bw = initial_state_bw, dtype=tf.float32)
    #    *******************

    # ***********************************************************
    # ** 3. bi-lstm 计算（展开）
    with tf.variable_scope('bidirectional_rnn'):
        # *** 下面，两个网络是分别计算 output 和 state
        # Forward direction
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)

        # backward direction
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        # *** 然后把 output_bw 在 timestep 维度进行翻转
        # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
        outputs_bw = tf.reverse(outputs_bw, [0])
        # 把两个oupputs 拼成 [timestep_size, batch_size, hidden_size*2]
        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.reshape(output, [-1, hidden_size * 2])
    # ***********************************************************
    return output  # [-1, hidden_size*2]


with tf.variable_scope('Inputs'):
    X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
    y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')

bilstm_output = bi_lstm(X_inputs)

with tf.variable_scope('outputs'):
    softmax_w = weight_variable([hidden_size * 2, class_num])
    softmax_b = bias_variable([class_num])
    y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b

# adding extra statistics to monitor
# y_inputs.shape = [batch_size, timestep_size]
correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_inputs, [-1]), logits=y_pred))

# ***** 优化求解 *******
tvars = tf.trainable_variables()  # 获取模型的所有参数
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # 优化器

# 梯度下降计算
train_op = optimizer.apply_gradients(zip(grads, tvars),
                                     global_step=tf.contrib.framework.get_or_create_global_step())
print
'Finished creating the bi-lstm model.'


def test_epoch(dataset):
    """Testing or valid."""
    _batch_size = 500
    fetches = [accuracy, cost]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    for i in xrange(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:1e-5, batch_size:_batch_size, keep_prob:1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
    mean_acc= _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost


sess.run(tf.global_variables_initializer())
tr_batch_size = 128
max_max_epoch = 6
display_num = 5  # 每个 epoch 显示是个结果
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
saver = tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量
for epoch in xrange(max_max_epoch):
    _lr = 1e-4
    if epoch > max_epoch:
        _lr = _lr * ((decay) ** (epoch - max_epoch))
    print 'EPOCH %d， lr=%g' % (epoch+1, _lr)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0
    for batch in xrange(tr_batch_num):
        fetches = [accuracy, cost, train_op]
        X_batch, y_batch = data_train.next_batch(tr_batch_size)
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:_lr, batch_size:tr_batch_size, keep_prob:0.5}
        _acc, _cost, _ = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
        _accs += _acc
        _costs += _cost
        show_accs += _acc
        show_costs += _cost
        if (batch + 1) % display_batch == 0:
            valid_acc, valid_cost = test_epoch(data_valid)  # valid
            print '\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                show_costs / display_batch, valid_acc, valid_cost)
            show_accs = 0.0
            show_costs = 0.0
    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num
    if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        print 'the save path is ', save_path
    print '\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost)
    print 'Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time)
# testing
print '**TEST RESULT:'
test_acc, test_cost = test_epoch(data_test)
print '**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost)

'''









'''
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

FLAGS=tf.app.flags.FLAGS


class Perceptron():
    def __init__(self):
        #basic environment
        self.graph=tf.Graph()
        self.session=tf.Session(graph=self.graph)

    #contains forward and training
    def fit(self,X,y,epochs=5,batch_size=100,learning_rate=0.001,print_log=False):
        #num of samples,features and category
        n_samples=X.shape[0]
        n_features=X.shape[1]

        #one hot-encoding,num of category
        y_dummy=pd.get_dummies(data=y).values
        n_category=y_dummy.shape[1]

        # shuffle for random sampling
        sp = ShuffleSplit(n_splits=epochs, train_size=0.8)
        indices = sp.split(X=X)

        # best accuracy on validation test
        best_validation_accus = 0

        #epoch record
        epoch = 1

        ##########################define graph(forward computation)#####################
        with self.graph.as_default():
            #data place holder
            self.X_p = tf.placeholder(dtype=tf.float32,
                                      shape=(None, n_features),
                                      name="input_placeholder")

            self.y_dummy_p = tf.placeholder(dtype=tf.float32,
                                            shape=(None, n_category),
                                            name="label_dummy_placeholder")

            self.y_p=tf.placeholder(dtype=tf.int64,
                                    shape=(None,),
                                    name="label_placeholder")

            #--------------------------fully connected layer-----------------------------------#
            #weights(initialized to 0)
            self.weights=tf.Variable(initial_value=tf.zeros(shape=(n_features,n_category)),
                                     name="weights")

            #biases(initialized to 0)
            self.biases=tf.Variable(initial_value=tf.zeros(shape=(n_category,)),
                                    name="biases")

            #shape of logits is (None,num_of_category)
            logits = tf.matmul(self.X_p, self.weights) + self.biases

            #----------------------------------------------------------------------------------#

            #probability
            self.prob=tf.nn.softmax(logits=logits,name="prob")

            #prediction
            self.pred=tf.argmax(input=self.prob,axis=1,name="pred")

            #accuracy
            self.accuracy=tf.reduce_mean(
                            input_tensor=tf.cast(x=tf.equal(x=self.pred,y=self.y_p),dtype=tf.float32),
                            name="accuracy"
                )

            #loss
            self.cross_entropy = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y_dummy_p)
                )

            #optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)

            self.init = tf.global_variables_initializer()

        #SGD training
        with self.session as sess:
            sess.run(self.init)

            #restore
            #new_saver = tf.train.import_meta_graph('./model/my-model-10000.meta')
            #new_saver.restore(sess, './model/my-model-10000')

            print("------------traing start-------------")

            for train_index,validation_index in indices:
                print("epoch:", epoch)

                trainDataSize=train_index.shape[0]
                validationDataSize=validation_index.shape[0]

                #average train loss and validation loss
                train_losses=[]; validation_losses=[]
                #average taing accuracy and validation accuracy
                train_accus=[]; validation_accus=[]

                #mini batch
                for i in range(0,(trainDataSize//batch_size)):
                    _,train_loss,train_accuracy=self.session.run(
                                    fetches=[self.optimizer,self.cross_entropy,self.accuracy],

                                    feed_dict={self.X_p:X[train_index[i*batch_size:(i+1)*batch_size]],
                                                self.y_dummy_p:y_dummy[train_index[i*batch_size:(i+1)*batch_size]],
                                                self.y_p:y[train_index[i*batch_size:(i+1)*batch_size]]
                                            }
                                    )

                    validation_loss,validation_accuracy=self.session.run(
                                    fetches=[self.cross_entropy,self.accuracy],

                                    feed_dict={self.X_p:X[validation_index],
                                                self.y_dummy_p:y_dummy[validation_index],
                                                self.y_p:y[validation_index]
                                            }
                                    )

                    #add to list to compute average value
                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    train_accus.append(train_accuracy)
                    validation_accus.append(validation_accuracy)


                    #weather print training infomation
                    if(print_log):
                        print("#############################################################")
                        print("batch: ",i*batch_size,"~",(i+1)*batch_size,"of epoch:",epoch)
                        print("training loss:",train_loss)
                        print("validation loss:",validation_loss)
                        print("train accuracy:", train_accuracy)
                        print("validation accuracy:", validation_accuracy)
                        print("#############################################################\n")

               # print("train_losses:",train_losses)
                ave_train_loss=sum(train_losses)/len(train_losses)
                ave_validation_loss=sum(validation_losses)/len(validation_losses)
                ave_train_accuracy=sum(train_accus)/len(train_accus)
                ave_validation_accuracy=sum(validation_accus)/len(validation_accus)
                print("average training loss:",ave_train_loss)
                print("average validation loss:",ave_validation_loss)
                print("average training accuracy:", ave_train_accuracy)
                print("average validation accuracy:", ave_validation_accuracy)
                epoch+=1

                #when we get a new best validation accuracy,we store the model
                if best_validation_accus<ave_validation_accuracy:
                    print("we got a new best accuracy on validation set!")

                    # Creates a saver. and we only keep the best model
                    saver = tf.train.Saver()
                    saver.save(sess, './model/my-model-10000')
                    # Generates MetaGraphDef.
                    saver.export_meta_graph('./model/my-model-10000.meta')


    def predict(self,X):
        with self.session as sess:
            #restore model
            new_saver = tf.train.import_meta_graph('./model/my-model-10000.meta',clear_devices=True)
            new_saver.restore(sess, './model/my-model-10000')

            graph=tf.get_default_graph()

            #get opration from the graph
            pred=graph.get_operation_by_name("pred").outputs[0]
            X_p=graph.get_operation_by_name("input_placeholder").outputs[0]
            pred = sess.run(fetches=pred, feed_dict={X_p: X})
        return pred

    def predict_prob(self,X):
        with self.session.as_default():
            prob=self.session.run(fetches=self.prob,feed_dict={self.X_p:X})
        return prob

'''