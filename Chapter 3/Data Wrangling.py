# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:42:53 2018

@author: nguyentrang
Data Wrangling
"""
#%% Subsetting a dataset:
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 3/'
#fullpath = '/Users/Sony/Desktop/GitHub/PredictiveAnalysis/Chapter 3/'

filename = 'Customer Churn Model.txt'
data = pd.read_csv(fullpath + filename)


# Selecting columns:
account_length = data['Account Length']
account_length.head()

subdata = data[['Account Length','VMail Message','Day Calls']]
subdata.head()

wanted_columns = ['Account Length','VMail Message','Day Calls']
subdata = data[wanted_columns]
subdata.head()



# Selecting rows:
data[0:50]
data[25:75]
data[:50]


CA_data = data[(data['State']== 'CA') & (data['Day Calls'] > 100)]
CA_data = data[(data['State']== 'CA') | (data['Day Calls'] > 100)]
CA_data.shape

# Selecting both rows and columns - loc and iloc are recommended to use.
    
subdata = data[['Account Length','VMail Message','Day Calls']][:50] #first 50 records 

data.ix[1:100,1:5]
data.ix[1:100,[0,1,2]]

data.iloc[:5] # return the first 5 rows
data.iloc[:,1:3]
data.iloc[:, [1,2,5]] #gets rows (or columns) at particular positions in the index (so it only takes integers).
#data.iloc[:data.index.get_loc('Row_Index_label') + 1, :4]
data.loc[:5] # return the first 6 rows - treat 5 as the row index.
data.loc[:,['Account Length' ,'Day Calls']] # gets rows (or columns) with particular labels from the index
data.loc[:,'Account Length' :'Day Calls']
data.loc[data['State'] == 'CA']



# Creating a new column:
data['Total Mins'] = data['Day Mins'] + data['Night Mins'] + data['Eve Mins'] + data['Intl Mins'] 
data['Total Mins'].head()

#%% Generating random numbers and their usage
np.random.randint(1,100)
np.random.random() # range 0 - 1
random.randrange(0,100,5)

a = range(100)
np.random.shuffle(a)

# Generating random number with seed
np.random.seed(1)
for i in range(5):
    print(np.random.random())

# Generating random number with probability distributions
# PROBABILITY DENSITY FUNCTION
# CUMULATIVE DENSITY FUNCTION
# UNIFORM DISTRIBUTION

randnum=np.random.uniform(1,100,100)



a=np.random.uniform(1,100,100000) # increase size to see normal distribution.
plt.hist(a)

# NORMAL DISTRIBUTION:
a=np.random.randn(100000)
plt.hist(a)
a.mean()

# Generating a dummy data frame:
d = pd.DataFrame({'A':np.random.randn(10),'B':2.5*np.random.randn(10) + 1.5})
plt.hist(d['A'])


column_list = data.columns.values.tolist()
a = len(column_list)
np.random.seed(1)
d = pd.DataFrame({'Column_Name': column_list,'A':np.random.randn(a),'B':2.5*np.random.randn(a) + 1.5})

# Index can also be passed as parameters
d=pd.DataFrame({'A':np.random.randn(10),'B':2.5*np.random.randn(10)+1.5},index=range(10,20))

#%% Grouping the data - aggregation, filtering and transformation.
d.drop(d.columns[[0]],axis = 1, inplace = True)

data_group = data.groupby(['State','Area Code'])['Phone'].count()
data.info()


grouped = data.groupby('State')
grouped.groups

for names,groups in grouped:
    print(names)
    print(groups)
    
# selecting a single group:
grouped.get_group('CA')
ca_data = data[data['State'] == 'CA']


grouped = data.groupby(['State','Area Code'])
grouped.get_group(('CA',415)).shape
test = grouped.sum()
test = data.corr()
grouped.size()


data.shape

grouped.aggregate({'Day Mins': np.sum, 'Day Calls': np.sum, 'Day Charge': np.average})
grouped.aggregate([np.sum, np.mean, np.std]) # apply to all columns 
# using lambda:

grouped.aggregate({'Day Mins': np.sum, 'Day Calls': np.sum, 'Day Charge' :lambda x : np.sum(x)/1000})

grouped['Day Mins'].filter(lambda x: np.sum(x) > 1500)

#%% FIT ==> PREDICT ==> EVALUATE (accuracy 65 - 90% good)
# First choice: the supposted vector learning Machine model
# Then, gradent descent.

#%% TRANSFORMATION

zscore = lambda x: (x - x.mean()) / x.std()
grouped.transform(zscore)


f = lambda x: x.fillna(x.mean())
grouped.transform(f)

grouped.nth(1)

data = data.sort(['State','Area Code'])

#%% Splitting a dataset in training and testing dataset:

from sklearn.cross_validation import train_test_split
train, test = train_test_split(data, test_size = 0.2)


#%% Concatenating and appending data:

data =pd.concat([train,test],axis=0)
