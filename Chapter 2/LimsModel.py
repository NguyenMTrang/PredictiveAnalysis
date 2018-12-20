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
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

fullpath = 'D:/GitHub/PredictiveAnalysis/Chapter 2/'
filename = 'limsappointments.csv'
data = pd.read_csv(fullpath + filename)
data.columns

# Machine Learning Map: Scikit-learn algorithm cheat sheet:
#https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html


# Split the data set into 3 sets of data:
# Apply different ML method to the traning data sets
# Check accuracy% on cross-validation data set
# Fit the test set
# Doing accuracy again.
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

data.to_csv('cleansed.csv',sep =',')

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
#https://anaconda.org/conda-forge/imbalanced-learn

from imblearn import under_sampling, over_sampling
X = data.loc[:, data.columns != 'AppointmentStatus']
y = data.loc[:, data.columns == 'AppointmentStatus']

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns


os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
#%% 
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

data_final_vars=data.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

col = ['TaskTypeID','SubTaskTypeID','ForcedAppointment','SameDayAsApplied','LobbyBureauID','Attempts', 'Reschedules','NoShows']
X=os_data_X[col]
y=os_data_y['y']

#%% Implementing the model:
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


#%% Logistic Regression Model Fitting:
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test_os, y_train, y_test_os = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)



#%% Preding the test result:
X_test =X_test[col]
y=os_data_y['y']


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


#%% Confusion Matrix:

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


#%% Compute precision:

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) #Precision/Recall


#%% ROC Curve:

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
#%%
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



#%%

# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(), 
                                                        X, 
                                                        y,
                                                        # Number of folds in cross-validation
                                                        cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        #n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 10))
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
plot_learning_curve(LogisticRegression(), "Learning Curve", X, y, cv=cv,train_sizes=np.linspace(.1, 1.0, 50))
plt.show()



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

