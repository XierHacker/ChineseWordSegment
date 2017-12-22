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
    #print(len(sentences))

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
    texts = u''.join(map(clean, sentences))  # 把所有的词连接起来,这里中间连接不用其他字符

    #print('Length of texts is %d' % len(texts))
    #print('Example of texts: \n', texts[:300])
    return texts


#以标点符号为分割,把整句划分为一个"句子"列表
def split2sentences(str):
    sentences = re.split(u'[，。！？、‘’“”]/[bems]', str)
    return sentences

#做成标签和字符对应元组,传入所有的sentences,得到每句sentence对应的word和labels
#顺便去除了一些空样本
def make_dataset(sentences):
    print("there has ",len(sentences)," sentence")
    datas=[]
    labels=[]
    for sentence in sentences:
        #这个时候word_tags应该是这种形式:[('人', 'b'), ('们', 'e'), ('常', 's'), ('说', 's')]
        words_tags = re.findall('(.)/(.)', sentence)
        if words_tags:
            words_tags=np.array(words_tags)
            words=words_tags[:,0];  datas.append(words)
            tags=words_tags[:,1]; labels.append(tags)
    return datas,labels



if __name__ =="__main__":
    text=file2str(filename='data/msr_train.txt')
    sentences=split2sentences(text)
    #print('Sentences number:', len(sentences))
    #print('Sentence Example:\n', sentences[0])
    datas,labels=make_dataset(sentences)
    print(len(datas),len(labels))
    print(datas[0],labels[0])