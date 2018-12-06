# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Import and evaluate the data set.
import pandas as pd
#fullpath = '/Users/Sony/Desktop/GitHub/PredictiveAnalysis/Chapter 2/'
fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 2/'
filename = 'titanic3.csv'
data = pd.read_csv(fullpath + filename)
data.to_csv('cleansed.csv',sep =',')

print(data.head(10))
print(data.shape)
data_columns = data.columns.values
data_stats = data.describe()
data.dtypes
data.info()

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

# Droping dublicates: duplicated(), drop_duplicates()
data.drop_duplicates()
# Alter the columns with index: set_Index()

#%% Creating dummy variables.
dummy_sex = pd.get_dummies(data['sex'])
data.drop('sex',axis = 1) # data.drop('age',1,inplace = true) drop columns without reassign data.
data.drop(data.columns[[0,1]],axis = 1)
data.drop(['sex','body'],axis = 1)
data.drop(data.columns[[0]],axis = 1, inplace = True)


data_dummy = data.drop('sex',axis = 1)
data_dummy.join(dummy_sex)
data['Sex'] = data['Sex'].map({'male':0,'female':1})
#%% Visualizing a dataset by basic plotting

import pandas as pd
import matplotlib.pyplot as plt
#fullpath = '/Users/Sony/Desktop/GitHub/PredictiveAnalysis/Chapter 2'
fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 2/'
filename = 'Customer Churn Model.txt'
data_customer = pd.read_csv(fullpath + filename)

# Scatter and line plots
data_customer.plot(kind = 'scatter', x = 'Day Mins', y = 'Day Charge')
data_customer.plot(kind = 'scatter', x = 'Night Mins', y = 'Night Charge')

figure,axis = plt.subplots(2,2,sharey = True, sharex = True)
data_customer.plot(kind = 'line', x = 'Day Mins', y = 'Day Charge',ax = axis[0][0])
data_customer.plot(kind = 'line', x = 'Night Mins', y = 'Night Charge',ax = axis[0][1])
data_customer.plot(kind = 'scatter', x = 'Day Calls', y = 'Day Charge',ax = axis[1][0])
data_customer.plot(kind = 'scatter', x = 'Night Calls', y = 'Night Charge',ax = axis[1][1])

plt.figure()
plt.plot(data_customer['Day Mins'],data_customer['Day Charge'],'r-',data_customer['Night Mins'],data_customer['Night Charge'],'b-')

plt.figure()
plt.hold(True)
plt.plot(data_customer['Day Mins'],data_customer['Day Charge'])
plt.plot(data_customer['Eve Mins'],data_customer['Eve Charge'])
plt.plot(data_customer['Night Mins'],data_customer['Night Charge'])
plt.plot(data_customer['Intl Mins'],data_customer['Intl Charge'])
plt.show()

data_customer.plot()

# Histograms plots
plt.hist([data_customer['Day Calls'],data_customer['Night Calls']],bins = 10,label = ['Day Calls','Night Calls'])
plt.xlabel('Number of Calls')
plt.ylabel('Frequency')
plt.title('Frequency of Day Calls') 
plt.legend(loc = 'upper right')
plt.show()

# Box plot and save figure.
# Boxplots are potent tools to spot outliers in a distribution. 
# Any value that is 1.5*IQR below the 1st quartile and is 1.5*IQR above the 1st quartile can be classified as an outlier.
figure_boxplot = plt.figure()
plt.boxplot([data_customer['Day Calls'],data_customer['Night Calls']])
plt.show()
figure_boxplot.savefig(fullpath + 'Scatter Plots.jpeg')