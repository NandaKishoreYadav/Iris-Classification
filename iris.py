import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\NandaKishore\OneDrive\Desktop\AIML Projects\Iris Classification\iris.csv")

data.fillna(1,inplace=True)

x=np.array(data.iloc[ : , :-1]) 
y=data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1) 

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)

import pickle

pickle.dump(model,open('model.pkl','wb'))


