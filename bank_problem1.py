# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 02:05:58 2018

@author: Atharv Kulkarni
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
#Importing dataset
dataset=pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
X_test = X_test.iloc[:,1:].values
X_train = dataset.iloc[:, 1:-1].values
y_train = dataset.iloc[:, 17].values
X = np.concatenate((X_train,X_test))

temp1 = DataFrame(data=X)
#Filling missing values
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,[2,5]])
X[:,[2,5]]=imputer.transform(X[:,[2,5]])

for i in range(84190):
    if pd.isnull(X[:,3][i]):
        X[:,3][i]="5 years"
    
imputer = Imputer(missing_values = 'NaN',strategy='most_frequent')        
imputer = imputer.fit(X[:,[14,15]])
X[:,[14,15]] = imputer.transform(X[:,[14,15]])    
        
imputer = Imputer(missing_values = 'NaN',strategy='median')        
imputer = imputer.fit(X[:,[9]])
X[:,[9]] = imputer.transform(X[:,[9]])




#encoding categorial variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
cat_var=[1,3,4,6]
for i in cat_var:
    X[:,i]=LabelEncoder().fit_transform(X[:,i])

 
''' 
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
''' 

    

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)








X_train1 = X[0:84190,:]
X_test1 = X[84190:,:]

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train1, y_train)

y_pred = classifier.predict(X_test1)

sub = pd.read_csv("sample_submission.csv")

sub["Predicted"] = y_pred

sub.to_csv("submission_7.csv" , index = False)
















