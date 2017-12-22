'''
#   BMES taged dataset
#   author:xiekun 2017.12.22
'''

import numpy as np
import pandas as pd
import re
import time
from itertools import chain

#固定句子长度为32
MAX_SENTENCE_SIZE=32


def clean(s):  # 清洗.（如每行的开头）去掉
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

#语料文件文件转换为一个原始语料句子的list
def file2corpus(filename):
    with open(filename, 'rb') as inp:
        corpus = inp.read().decode('gbk')   #原始语料 str对象

    corpus = corpus.split('\r\n')           #换行切分,得到一个简陋列表
    corpus = u''.join(map(clean, corpus))   # 把所有处理的句子连接起来,这里中间连接不用其他字符 str对象

    corpus = re.split(u'[，。！？、‘’“”]/[bems]', corpus)    # 以标点符号为分割,把语料划分为一个"句子"列表
    return corpus              #[人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e ,xxx,....]


# 传入原始语料句子corpus列表,
#得到的字数据datas和对应的labels数据都放到dataframe里面存储,方便后面的处理
#返回的是df_data,word2id,id2word,tag2id,id2tag
def make_component(corpus):
    # 从原始语料corpus列表
    # 得到每句corpus想应的sentence以及对应的labels
    print("there has ", len(corpus)," sentence")
    sentences= []
    tags = []
    for s in corpus:
        sentence_tags = re.findall('(.)/(.)', s)     # sentence_tags:[('人', 'b'), ('们', 'e'), ('常', 's'), ('说', 's')]
        if sentence_tags:                            # 顺便去除了一些空样本
            sentence_tags = np.array(sentence_tags)
            sentences.append(sentence_tags[:, 0])
            tags.append(sentence_tags[:, 1])
    '''
    # 注意:这里sentences每一个元素表示一个sentence['人' '们' '常' '说' '生' '活' '是' '一' '部' '教' '科' '书']
    # tags每一个元素表示的是一个句子对应的标签['b' 'e' 's' 's' 'b' 'e' 's' 's' 's' 'b' 'm' 'e']
    '''

    #使用pandas处理,简化流程
    df_data = pd.DataFrame({'sentences': sentences, 'tags': tags}, index=range(len(sentences)))
    # 每句话长度
    df_data['sentence_len'] = df_data['sentences'].apply(lambda sentences: len(sentences))

    # 得到所有的字,这里的all_words是一个列表,存放了这个语料中所有的字
    all_words = list(chain(*df_data['words'].values))

    # 2.统计所有 word
    sr_allwords = pd.Series(data=all_words)
    # print(sr_allwords)

    # 统计每个字出现的频率,同时相当于去重复,得到字的集合(这里还是Serieas的index对象)
    words_set = (sr_allwords.value_counts()).index
    # print(words_set)

    # 字的id列表
    # 从1开始，因为准备把0作为填充值
    word_ids = range(1, len(words_set) + 1)

    # tag列表
    tags = ['x', 's', 'b', 'm', 'e']
    tag_ids = range(len(tags))

    # 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
    word2id = pd.Series(word_ids, index=words_set)
    # print("word2id:\n",word2id.head(3))
    id2word = pd.Series(words_set, index=word_ids)
    # print("id2word:\n",id2word.head(3))
    # print(word2id["的"])
    # print(id2word[1])
    tag2id = pd.Series(tag_ids, index=tags)
    # print("tag2id:\n",tag2id.head)
    id2tag = pd.Series(tags, index=tag_ids)
    # print("id2tag:\n",id2tag.head)

    return df_data, word2id, id2word, tag2id, id2tag



#得到一句话的padding后的 id
def X_padding(sentence,word2id):
    pass

#得到一个label的padding后的id
def y_padding(labels,tags2id):
    pass


if __name__ =="__main__":
    corpus=file2sentences(filename='data/msr_train.txt')
    print('Sentences number:', len(corpus))
    print('Sentence Example:\n', corpus[0])

    #orpus,labels=make_corpus(sentences)
    #print(len(corpus),len(labels))
    #print(corpus[0],labels[0])
    #print("elements of datas:",type(corpus[0]))

    #merge2frame(datas,labels)
    #all_words=chain(datas)
    #print(list(all_words))
    #print(word2id.head(5))