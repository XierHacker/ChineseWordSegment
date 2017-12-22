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
    texts = u''.join(map(clean, sentences))  # 把所有的词连接起来,这里中间连接不用其他字符

    #print('Length of texts is %d' % len(texts))
    #print('Example of texts: \n', texts[:300])
    return texts


#以标点符号为分割,把整句划分为一个"句子"列表
def split2sentences(str):
    sentences = re.split(u'[，。！？、‘’“”]/[bems]', str)
    return sentences

#做成标签和字符对应元组,传入所有的sentence就行.
def make_dataset(sentences):
    for sentence in sentences:
        words_tags = re.findall('(.)/(.)', sentences[0])



if __name__ =="__main__":
    text=file2str(filename='data/msr_train.txt')
    sentences=split2sentences(text)
    #print('Sentences number:', len(sentences))
    print('Sentence Example:\n', sentences[0])
    words_tags = re.findall('(.)/(.)', sentences[0])
    print(words_tags)