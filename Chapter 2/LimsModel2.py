# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 11:07:56 2018

@author: nguyentrang
"""

#%% Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import seaborn as sn
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np


#%% Import data
fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 2/'
filename = 'limsappointments_morefeatures.csv'
data = pd.read_csv(fullpath + filename)

#%% Descriptive Summary
data.columns
describe_data = data.describe()
data.shape

#%% Data Preprocessing
# Check if any error values or NaNs in data
all_columns = ['CaseID', 'TaskID', 'CaseName', 'TaskStatusID', 'TaskStatus',
       'TaskTypeID', 'TaskType', 'SubTaskTypeID', 'AppointmentType',
       'AppointmentStatusID', 'AppointmentStatus', 'NotifierStatusID',
       'NotifierStatus', 'ApplicationDate', 'AppointmentDate',
       'AppointmentStartTime', 'AppointmentTime', 'TimeSlotID', 'LanguageID',
       'Language', 'NONCReasonID', 'ReasonOutsideTimeframe',
       'ForcedAppointment', 'SpecialSkillID', 'Culture', 'InterpreterOptionID',
       'InterpreterNeeded', 'CashAidProgram', 'AppointmentPrograms', 'IN/ES',
       'SameDayAsApplied', 'SameDayAsRescheduled', 'AssignedTo', 'CompletedBy',
       'LobbyBureauID', 'Lobby', 'AppointmentMonth', 'LocationCD', 'LobbyID',
       'DayNumberOfWeek', 'DayNumberOfMonth', 'DayNumberOfYear', 'IsHoliday',
       'MonthNumberOfYear', 'AppHour', 'Age', 'Sex', 'HomeZip', 'HomeZip4',
       'NumberofPastAppts', 'LastApptDate', 'ScheduledDate', 'PastNoShow',
       'Interval', 'Waittime']

print('Age:',sorted(data.Age.unique()))
print('Sex:',data.Sex.unique())
print('TaskStatusID:',data.TaskStatusID.unique())
print('TaskStatus:',data.TaskStatus.unique())
print('TaskType:',data.TaskType.unique())
print('SubTaskTypeID:',data.SubTaskTypeID.unique())
print('AppointmentType:',data.AppointmentType.unique())
print('AppointmentStatusID:',sorted(data.AppointmentStatusID.unique()))
print('AppointmentStatus:',data.AppointmentStatus.unique())
print('NotifierStatusID:',data.NotifierStatusID.unique())
print('NotifierStatus:',data.NotifierStatus.unique())
print('AppointmentStartTime:',data.AppointmentStartTime.unique())
print('AppointmentTime:',data.AppointmentTime.unique())
print('TimeSlotID:',data.TimeSlotID.unique())
print('Language:',sorted(data.Language.unique()))
print('NONCReasonID:',data.NONCReasonID.unique())
print('ReasonOutsideTimeframe:',data.ReasonOutsideTimeframe.unique())
print('ForcedAppointment:',data.ForcedAppointment.unique())
print('SpecialSkillID:',data.SpecialSkillID.unique())
print('Culture:',data.Culture.unique())
print('InterpreterOptionID:',data.InterpreterOptionID.unique())
print('InterpreterNeeded:',sorted(data.InterpreterNeeded.unique()))
print('SameDayAsApplied:',data.SameDayAsApplied.unique())
print('SameDayAsRescheduled:',data.SameDayAsRescheduled.unique())
print('Lobby:',data.Lobby.unique())
print('AppointmentMonth:',data.AppointmentMonth.unique())
print('HomeZip:',data.HomeZip.unique())
print('TimeSlotID:',data.NumberofPastAppts.unique())


# delete canceled, scheduled or rescheduled appointments.
no_show = data["AppointmentStatus"].value_counts()
data = data[data.AppointmentStatus.isin(['No Show', 'Showed', 'No Answer'])]
percent_no_show = no_show["No Show"]/ no_show.sum() * 100
print("Percentage of no show :", percent_no_show)
data["NoShow"] = data["AppointmentStatus"].map(lambda x : 0 if x == "Showed" else 1)


group_sex = data.groupby("Sex")["NoShow"].mean()
group_age = data.groupby("Age")["NoShow"].mean()
group_tasktype = data.groupby("TaskType")["NoShow"].mean()

group_appttype = data.groupby("AppointmentType")["NoShow"].apply(lambda x: x/x.sum())

total = data.NoShow.value_counts()

plt.plot(group_age)
plt.plot(group_sex)
plt.plot(group_appttype)
group_appttype = pd.crosstab(data.NoShow,data.AppointmentType).apply(lambda x : x/x.sum(),axis = 1)


# Remove outliers
data = data[(data.Age >= 15) & (data.Age <= 90)]

# Remove if TimeSlotID is null:

data = data[data["TimeSlotID"].notnull()]

# Convert to categorical and create dummy variables for sex, 
data.Sex = pd.Categorical(data.Sex)
sex_dummy = pd.get_dummies(data.Sex, prefix = 'Sex')
data = pd.concat([data, sex_dummy], axis = 1)

# Convert to categorical and create dummy variables for TaskType, 
data.TaskType = pd.Categorical(data.TaskType)
TaskType_dummy = pd.get_dummies(data.TaskType, prefix = 'TaskType')
data = pd.concat([data, TaskType_dummy], axis = 1)

# ApptHour:
ApptHour_df = data.groupby(['AppHour']).agg({'NoShow':'value_counts'})
ApptHour = ApptHour_df.groupby(level = 0).apply(lambda x: x/x.sum())
ApptHour.xs(1,level = 'NoShow').plot(kind = 'bar')
ApptHour.unstack().plot(kind = 'bar')

# Created Dummy variables for ApptHour:
data.AppHour = pd.Categorical(data.AppHour)
AppHour_dummy = pd.get_dummies(data.AppHour, prefix = 'AppHour')
data = pd.concat([data, AppHour_dummy], axis = 1)

# Remove appointments with task status is new, open, deleted.
data = data[data.TaskStatus == "COMPLETE"]


data.isnull()
pd.isnull(data['age'])
pd.notnull(data['age'])
pd.notnull(data['age']).sum()
# Delete unnessary data:

data['Interval'] = data['Interval'].fillna(0)
data['Waittime']= data['Waittime'].fillna(0)

data['Waittime'].fillna(data['age'].mean())

data = data[(data['Interval'] < 30000) & (data['Interval'] >= 0)]
data = data[(data['Waittime'] < 30000) & (data['Waittime'] >= 0)]

describe_data = data.describe()




