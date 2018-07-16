import numpy as np

def words_cut(tags,sentence):
    pass




# display accuracy and loss information
def showInfo(print_log=False, loss=None, accuracy=None, train_losses=None, train_accus=None):
    if print_log:
        print("----training loss  : ", loss)
        print("----train accuracy : ", accuracy)
        print()
    else:
        print("----average training loss       : ", sum(train_losses) / len(train_losses))
        print("----average training accuracy   : ", sum(train_accus) / len(train_accus))
        print("----average validation loss     : ", loss)
        print("----average validation accuracy : ", accuracy)