'''
    BMES taged dataset

'''

import numpy as np
import pandas as pd
import re
import time


def file2str(filename):
    with open(filename, 'rb') as inp:
        texts = inp.read().decode('gbk')
    # print(texts)
    sentences = texts.split('\r\n')  # 根据换行切分
    print(len(sentences))

    # 将不规范的内容（如每行的开头）去掉
    def clean(s):
        if u'“/s' not in s:  # 句子中间的引号不应去掉
            return s.replace(u' ”/s', '')
        elif u'”/s' not in s:
            return s.replace(u'“/s ', '')
        elif u'‘/s' not in s:
            return s.replace(u' ’/s', '')
        elif u'’/s' not in s:
            return s.replace(u'‘/s ', '')
        else:
            return s
    texts = u''.join(map(clean, sentences))  # 把所有的词拼接起来
    print('Length of texts is %d' % len(texts))
    print('Example of texts: \n', texts[:300])
    return texts

#use re to split whole str to sentence
def split2sentence(str):






if __name__ =="__main__":
    text=file2str(filename='data/msr_train.txt')
