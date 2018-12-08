# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:26:06 2018
Model Building for LIMS Appointments Dataset
@author: nguyentrang
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import seaborn as sn
from sklearn import preprocessing


fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 2/'
filename = 'limsappointments.csv'
data = pd.read_csv(fullpath + filename)
data.columns


#%% Data Pre-processing 
wanted_columns = ['TaskType','AppointmentType','AppointmentStatus','Language','ForcedAppointment','SameDayAsApplied','Lobby',   'DayNumberOfWeek', 'DayNumberOfMonth','AppHour','Attempts', 'Reschedules',
       'NoShows']

wanted_columns_test = ['TaskTypeID','SubTaskTypeID','AppointmentStatus','LanguageID','ForcedAppointment','SameDayAsApplied','LobbyBureauID',   'DayNumberOfWeek', 'DayNumberOfMonth','AppHour','Attempts', 'Reschedules','NoShows']
data = data[wanted_columns_test]

# Delete if appointment is forced checking and map appointments 'Showed' = 1, 
#  'No Answer', 'No Show', 'Cancelled', 'Scheduled', 'Rescheduled' = 0
data = data[data.AppointmentStatus.isin(['No Show', 'Showed', 'No Answer'])]

data['AppointmentStatus'] = data['AppointmentStatus'].map(lambda x : 1 if  x == 'Showed' else 0)
data['ForcedAppointment'] = data['ForcedAppointment'].map({'N':0,'Y':1})
data['SameDayAsApplied'] = data['SameDayAsApplied'].map({'N':0,'Y':1})


data['Attempts'] = data['Attempts'].fillna(0)
data['Reschedules'] = data['Reschedules'].fillna(0)
data['NoShows'] = data['NoShows'].fillna(0)

data.info()
data['AppointmentStatus'].unique()

data = data.dropna()

data.isna()
data.info()

#%% Building model
X = data.drop('AppointmentStatus',axis =1)
y = data['AppointmentStatus']




test = data.groupby('AppointmentStatus').mean()



pd.crosstab(data.DayNumberOfWeek,data.AppointmentStatus).plot(kind = 'bar') #Y

pd.crosstab(data.SubTaskTypeID,data.AppointmentStatus).plot(kind = 'bar') #Y

pd.crosstab(data.TaskType,data.AppointmentStatus).plot(kind = 'bar') # Y

pd.crosstab(data.LanguageID,data.AppointmentStatus).plot(kind = 'bar')

pd.crosstab(data.ForcedAppointment,data.AppointmentStatus).plot(kind = 'bar') # Y

pd.crosstab(data.SameDayAsApplied,data.AppointmentStatus).plot(kind = 'bar')

pd.crosstab(data.NoShows,data.AppointmentStatus).plot(kind = 'bar') # Y

pd.crosstab(data.Attempts,data.AppointmentStatus).plot(kind = 'bar') # Y

pd.crosstab(data.Reschedules,data.AppointmentStatus).plot(kind = 'bar') # Y

pd.crosstab(data.AppHour,data.AppointmentStatus).plot(kind = 'bar') # Y

pd.crosstab(data.SpecialSkillID,data.AppointmentStatus).plot(kind = 'bar')


#%% Over-sampling useing SMOTE
https://anaconda.org/conda-forge/imbalanced-learn

from imblearn import under_sampling, over_sampling
X = data.loc[:, data.columns != 'AppointmentStatus']
y = data.loc[:, data.columns == 'AppointmentStatus']

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

#%% 
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1)

#model = SVC(kernel = 'linear', probability = True, random_state = 0)
#model = tree.DecisionTreeClassifier()
#model = tree.DecisionTreeClassifier(min_samples_split = 15) 
model = SVC(C = 1.0 ,kernel='rbf', gamma = 1)

#model = LogisticRegression()            
#model = GaussianNB()

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_predict)

corr_matrix = data.corr()


import matplotlib.pyplot as plt
conf = confusion_matrix(y_test, y_predict)
plt.imshow(conf, cmap='binary', interpolation='None')
plt.show()

y_predict == 1

plt.hist(y_train)
plt.hist(y_predict)



