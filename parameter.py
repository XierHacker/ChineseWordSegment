'''
    #file that contains parameters of this model
'''

MAX_SENTENCE_SIZE=32    #固定句子长度为32
TIMESTEP_SIZE=MAX_SENTENCE_SIZE #LSTM的time_step应该和句子长度一致
INPUT_SIZE=EMBEDDING_SIZE=64
DECAY=0.85
MAX_EPOCH=5
LAYER_NUM=2     #lstm层数
HIDDEN_UNITS_NUM=128
CLASS_NUM=5     #类别数量

max_max_epoch = 10
vocab_size = 5159  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）



