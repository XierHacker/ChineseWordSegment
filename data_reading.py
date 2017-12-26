import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def loadComponenet(folderPath):
    '''
    读入组件id2tags,id2words,tags2id,words2id
    :param folderPath: path that store dataset
    :return:
    '''
    id2tages=pd.read_csv(filepath_or_buffer=folderPath+"id2tags.csv",encoding="utf-8",usecols=(0,1))
    return id2tages


def loadDataset(folderPath):
    pass

def loadAll(folderPath):
    pass

if __name__=="__main__":
    id2tages=loadComponenet("./dataset/msr/")
    print(type(id2tages))
    print(id2tages)