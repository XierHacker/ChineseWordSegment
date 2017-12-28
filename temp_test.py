import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'''
value=[[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]]
dataset=np.array(value)

print(dataset)


'''

frame=pd.DataFrame(
    data={"X":[1,2,3,4,5,6],"y":[0,1,0,1,1,1]}
)
print("frame\n",frame)

X_train,X_test=train_test_split(frame,test_size=0.2)
print("X_train:\n",X_train)
print("X_test:\n",X_test)

frame.to_pickle(path="./fream.pkl")

frame2=pd.read_pickle(path="./fream.pkl")
print(frame2)


















