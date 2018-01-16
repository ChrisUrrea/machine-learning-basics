# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')
#take all rows and  take all columns EXCEPT last (-1)
X =  dataset.iloc[:,:-1].values
#y is last value
y = dataset.iloc[:,-1].values

#taking care of missing data
#import imputer class from scki library
from sklearn.preprocessing import Imputer

#create an object of Imputer class
#axis = 0 for columns, axis = 1 for rows
imputer  = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

#replace missing data with mean values
imputer= imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#to hand categorical variables transform into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#create an instance of LabelEncoder class and save it
labelencoderX = LabelEncoder()
#fit our transform to all countries to an array of integers
X[:,0] = labelencoderX.fit_transform(X[:,0])
#now we have id integers for each country, but these numbers [0,1,2]
#are not quantitative representations, but rather categorical
#Problem! Use dummy variables vectors of [0,1] 
#transformer is a matrix of integers, denoting the values taken on by categorical (discrete) features. 
#The output will be a sparse matrix where each column corresponds to one possible value of one feature.
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#now do same for y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split

#make the split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

#Before the machine learning work, we must do Feature Scaling
#prevent one variable overpowering others
#algorithim convereges much faster when features are scaled
from sklearn.preprocessing import StandardScaler
#create instance for Standard 
sc_X = StandardScaler()

#we must fit AND transfrom for training set
#important to fit standard scaler to Xtrain first before Xtest, so both are under same scale
X_train = sc_X.fit_transform(X_train)
#for test set we only have to transform, we do not fit it as we want to test how well train data predicts test
X_test = sc_X.transform(X_test)

#Tradeoffs scale our dummy variables:
#pros: everything will be on the same scale, prediction will be much more accurate
#cons: lose the interpretation of which ovservatiosn beling to which country(category)

