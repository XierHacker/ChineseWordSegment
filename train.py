import tensorflow as tf
import numpy as np
import pandas as pd
from bilstm import BiLSTM

#读数据
df_train=pd.read_pickle(path="./dataset/msr/summary_train.pkl")
df_validation=pd.read_pickle(path="./dataset/msr/summary_validation.pkl")
df_test=pd.read_pickle(path="./dataset/msr/summary_test.pkl")

X_train=np.asarray(list(df_train['X'].values))
y_train=np.asarray(list(df_train['y'].values))

X_validation=np.asarray(list(df_validation['X'].values))
y_validation=np.asarray(list(df_validation['y'].values))

X_test=np.asarray(list(df_test['X'].values))
y_test=np.asarray(list(df_test['y'].values))


#train model
bilstm=BiLSTM()
bilstm.fit(X_train,y_train,X_validation,y_validation,"msr")

#testing model
#accuracy=bilstm.pred(name="msr",X=X_test,y=y_test)
#print(accuracy)
