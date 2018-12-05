# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:42:53 2018

@author: nguyentrang
Data Wrangling
"""
#%% Subsetting a dataset:
import pandas as pd

#fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 3/'
fullpath = '/Users/Sony/Desktop/GitHub/PredictiveAnalysis/Chapter 3/'

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

data.columns.values.tolist()

# Selecting rows:
data[0:50]
data[25:75]
data[:50]


CA_data = data[(data['State']== 'CA') & (data['Day Calls'] > 100)]
CA_data = data[(data['State']== 'CA') | (data['Day Calls'] > 100)]
CA_data.shape

# Selecting both rows and columns - see loc and iloc.
    
subdata = data[['Account Length','VMail Message','Day Calls']][:50] #first 50 records 
data.loc[:50]
data.ix[1:100,1:5]
data.ix[1:100,[0,1,2]]

# Creating a new column:
data['Total Mins'] = data['Day Mins'] + data['Night Mins'] + data['Eve Mins'] + data['Intl Mins'] 
data['Total Mins'].head()

#%% Generating random numbers and their usage
import numpy as np
np.random.randint(1,100)
np.random.random() # range 0 - 1
import random
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
import numpy as np
randnum=np.random.uniform(1,100,100)

import numpy as np
import matplotlib.pyplot as plt

a=np.random.uniform(1,100,100000) # increase size to see normal distribution.
plt.hist(a)

# NORMAL DISTRIBUTION:
a=np.random.randn(100)
plt.hist(a)
a.mean()

