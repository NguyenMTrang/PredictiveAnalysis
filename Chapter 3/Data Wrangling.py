# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:42:53 2018

@author: nguyentrang
Data Wrangling
"""
#%% Subsetting a dataset:
import pandas as pd

fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 3/'
filename = 'Customer Churn Model.txt'
data = pd.read_csv(fullpath + filename)

account_length = data['Account Length']
account_length.head()

subdata = data[['Account Length','VMail Message','Day Calls']]
subdata.head()

wanted_columns = ['Account Length','VMail Message','Day Calls']
subdata = data[wanted_columns]
subdata.head()

data.columns == wanted_columns