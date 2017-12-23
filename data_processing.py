'''
#   BMES taged dataset
#   author:xiekun 2017.12.22
'''
import numpy as np
import pandas as pd
import re
import time
from itertools import chain
from parameter import MAX_SENTENCE_SIZE

def clean(s):
    '''
     # 清洗.（如每行的开头）去掉
    :param s:
    :return:
    '''
    if u'“/s' not in s:                 # 句子中间的引号不应去掉
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

def file2corpus(filename):
    '''
    :param filename:
    :return: 语料文件文件转换为一个原始语料句子的list
    '''
    with open(filename, 'rb') as inp:
        corpus = inp.read().decode('gbk')   #原始语料 str对象
    corpus = corpus.split('\r\n')           #换行切分,得到一个简陋列表
    corpus = u''.join(map(clean, corpus))   # 把所有处理的句子连接起来,这里中间连接不用其他字符 str对象
    corpus = re.split(u'[，。！？、‘’“”]/[bems]', corpus)    # 以标点符号为分割,把语料划分为一个"句子"列表
    return corpus              #[人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e ,xxx,....]

def make_component(corpus,save=False):
    '''
    :param corpus: 传入原始语料句子corpus列表得到的字数据datas和对应的labels数据都放到dataframe里面存储,方便后面的处理
    :return: df_data,word2id,id2word,tag2id,id2tag
    '''
    sentences= []
    tags = []
    for s in corpus:           #corpus列表得到每句corpus想应的sentence以及对应的labels
        sentence_tags = re.findall('(.)/(.)', s)     # sentence_tags:[('人', 'b'), ('们', 'e'), ('常', 's'), ('说', 's')]
        if sentence_tags:                            # 顺便去除了一些空样本
            sentence_tags = np.array(sentence_tags)
            sentences.append(sentence_tags[:, 0])    #sentences每一个元素表示一个sentence['人' '们' '常' '说' '生' '活' '是' '一' '部' '教' '科' '书']
            tags.append(sentence_tags[:, 1])         #tags每一个元素表示的是一个句子对应的标签['b' 'e' 's' 's' 'b' 'e' 's' 's' 's' 'b' 'm' 'e']

    #使用pandas处理,简化流程
    df_data = pd.DataFrame({'sentences': sentences, 'tags': tags}, index=range(len(sentences)))
    df_data['sentence_len'] = df_data['sentences'].apply(lambda sentences: len(sentences))  # 每句话长度

    # 得到所有的字,这里的all_words是一个列表,存放了这个语料中所有的字
    all_words = list(chain(*df_data['sentences'].values))
    sr_allwords = pd.Series(data=all_words)         # 2.列表做成pandas的Series

    words = (sr_allwords.value_counts()).index  #字列表.统计每个字出现的频率,同时相当于去重复,得到字的集合(这里还是Serieas的index对象)
    word_ids = range(1, len(words) + 1)         #字的id列表,从1开始，因为准备把0作为填充值
    tags = ['x', 's', 'b', 'm', 'e']            #tag列表
    tag_ids = range(len(tags))                  #tag的id列表

    # 映射表
    word2id = pd.Series(word_ids, index=words_set)      #字到id
    id2word = pd.Series(words_set, index=word_ids)      #id到字
    tag2id = pd.Series(tag_ids, index=tags)             #tag到id
    id2tag = pd.Series(tags, index=tag_ids)             #id到tag
    return df_data, word2id, id2word, tag2id, id2tag
    # need to do:save this component to .csv files


def X_padding(sentence,word2id):
    '''
    返回一句话padding之后的id列表,使用的时候,把一个字符串转为list传进来就行
    :param sentence: 一个句子的列表
    :param word2id: word2id映射
    :return:   一句话的padding后的 ids
    '''
    ids=list(word2id[sentence])
    if len(ids) > MAX_SENTENCE_SIZE:        #超过就截断
        return ids[:MAX_SENTENCE_SIZE]
    if len(ids) < MAX_SENTENCE_SIZE:        #短了就补齐
        ids.extend([0]*(MAX_SENTENCE_SIZE-len(ids)))
    return ids


def y_padding(tags,tags2id):
    '''
    #得到一个label的padding后的id
    :param tags:
    :param tags2id:
    :return:
    '''
    ids = list(tags2id[tags])
    if len(ids) > MAX_SENTENCE_SIZE:  # 超过就截断
        return ids[:MAX_SENTENCE_SIZE]
    if len(ids) < MAX_SENTENCE_SIZE:  # 短了就补齐
        ids.extend([0] * (MAX_SENTENCE_SIZE - len(ids)))
    return ids

def make_dataset(filename):
    corpus = file2corpus(filename)
    print("corpus contains", len(corpus), " sentence")
    df_data, word2id, id2word, tag2id, id2tag = make_component(corpus,save=False)
    #把数据转换为ids的数据



if __name__ =="__main__":
    ids_sentence=X_padding(list("的一国"),word2id)
    #print(ids)
    #print(len(ids))
    ids_tags=y_padding(['x', 's', 'b', 'm', 'e'],tag2id)
    #print("ids_tags:",ids_tags)
    #orpus,labels=make_corpus(sentences)
    #print(len(corpus),len(labels))
    #print(corpus[0],labels[0])
    #print("elements of datas:",type(corpus[0]))

    #merge2frame(datas,labels)
    #all_words=chain(datas)
    #print(list(all_words))
    #print(word2id.head(5))