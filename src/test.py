import tensorflow as tf
import numpy as np
import pandas as pd
from bilstm import BiLSTM
from bilstm_crf import BiLSTM_CRF

#读数据
df_test=pd.read_pickle(path="./dataset/msr/summary_test.pkl")
X_test=np.asarray(list(df_test['X'].values))
y_test=np.asarray(list(df_test['y'].values))

#testing model
bilstm=BiLSTM()
accuracy=bilstm.pred(name="msr",X=X_test,y=y_test)
print(accuracy)
