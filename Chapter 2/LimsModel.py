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


fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 2/'
filename = 'limsappointments.csv'
data = pd.read_csv(fullpath + filename)
data.columns
wanted_columns = ['TaskTypeID','SubTaskTypeID','AppointmentStatus','TimeSlotID','LanguageID','ForcedAppointment','SpecialSkillID','SameDayAsApplied','LobbyBureauID',   'DayNumberOfWeek', 'DayNumberOfMonth', 'IsHoliday']
data = data[wanted_columns]

# Delete if appointment is forced checking and map appointments 'Showed' = 1, 
#  'No Answer', 'No Show', 'Cancelled', 'Scheduled', 'Rescheduled' = 0
data = data[data['AppointmentStatus'] != 'Force Checkin' ]
data['AppointmentStatus'] = data['AppointmentStatus'].map(lambda x : 1 if  x == 'Showed' else 0)
data['ForcedAppointment'] = data['ForcedAppointment'].map({'N':0,'Y':1})
data['SameDayAsApplied'] = data['SameDayAsApplied'].map({'N':0,'Y':1})

data.info()
data['AppointmentStatus'].unique()

data = data.dropna()

data.isna()
data.info()

X = data.drop('AppointmentStatus',axis =1)
y = data['AppointmentStatus']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1)

model = SVC(kernel = 'linear', probability = True, random_state = 0)
#model = tree.DecisionTreeClassifier()
#model = tree.DecisionTreeClassifier(min_samples_split = 15) 

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)


