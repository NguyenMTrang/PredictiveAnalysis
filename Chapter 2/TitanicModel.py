# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:55:31 2018
Model Building for Titanic Dataset
@author: nguyentrang
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#fullpath = '/Users/Sony/Desktop/GitHub/PredictiveAnalysis/Chapter 2/'
fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 2/'
filename = 'titanic3.csv'
data = pd.read_csv(fullpath + filename)
wanted_columns = ['pclass','sex','age','sibsp','parch','fare','survived']
data = data[wanted_columns]

data['sex'] = data['sex'].map({'male': 0, 'female' : 1}) # replacing the data for sex 1,0
data = data.dropna()

data.isna()

X = data.drop('survived',axis =1)
y = data['survived']


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1)

model = SVC(kernel = 'linear', probability = True, random_state = 0)

# DecisionTree model
#from sklearn import tree
#model = tree.DecisionTreeClassifier()

#from sklearn import tree
#model = tree.DecisionTreeClassifier(min_samples_split = 15) 


model.fit(X_train, y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)





