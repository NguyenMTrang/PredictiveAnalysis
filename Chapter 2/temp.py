# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Import and evaluate the data set.
import pandas as pd
fullpath = '/Users/Sony/Desktop/GitHub/PredictiveAnalysis/Chapter 2'
filename = '/titanic3.csv'
data = pd.read_csv(fullpath + filename)

print(data.head(10))
print(data.shape)
data_columns = data.columns.values
data_stats = data.describe()
data.dtypes


#%% Handling Missing Data.
data.isnull()
pd.isnull(data['age'])
pd.notnull(data['age'])
pd.notnull(data['age']).sum()

# By deleting null values:
data.dropna(axis = 0, how = 'all') # 0 for row, 1 for columns.
data.dropna(axis = 0, how = 'any')

# By replacing null values with meaningful values:
data.fillna(0)
data.fillna('missing')

data['age'].fillna(0)
data['age'].fillna(data['age'].mean())

data['age'].fillna(method = 'ffill')
data['age'].fillna(method = 'backfill')

#%% Creating dummy variables.
dummy_sex = pd.get_dummies(data['sex'])
data.drop('sex',axis = 1) # data.drop('age',1,inplace = true) drop columns without reassign data.
data.drop(data.columns[[0,1]],axis = 1)
data.drop(['sex','body'],axis = 1)


data_dummy = data.drop('sex',axis = 1)
data_dummy.join(dummy_sex)

#%%

