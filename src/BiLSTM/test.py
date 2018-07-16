
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