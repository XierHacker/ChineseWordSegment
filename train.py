import tensorflow as tf
import numpy as np
import pandas as pd
from bilstm_crf import BiLSTM_CRF


#读数据
df_train=pd.read_pickle(path="./dataset/msr/summary_train.pkl")
df_validation=pd.read_pickle(path="./dataset/msr/summary_validation.pkl")

X_train=np.asarray(list(df_train['X'].values))
y_train=np.asarray(list(df_train['y'].values))
X_validation=np.asarray(list(df_validation['X'].values))
y_validation=np.asarray(list(df_validation['y'].values))

#train model
bilstm=BiLSTM_CRF()
bilstm.fit(X_train,y_train,X_validation,y_validation,"msr")
