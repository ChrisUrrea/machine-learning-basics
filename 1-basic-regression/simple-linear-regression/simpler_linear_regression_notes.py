"""
Created on Mon Jan 15 17:42:41 2018

@author: christianurrea
"""
#data preprocessing first
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#in simple linear regression, sklearn automatically scales our variables
#thus no need to apply our own scaling


#Fitting our simple linear regression model
#import our linear regressor class
from sklearn.linear_model import LinearRegression
#create an instance of Linear regression class
regressor = LinearRegression()
#"fit"- train - our linear regressor on our training set
regressor.fit(X_train, y_train)

#Predicting the Test Set results

#use our regressor on X_test set to find a vector of predicted y
y_pred = regressor.predict(X_test)

#Plotting and visualizing our training set results

#make the scatter plot of our training set
plt.scatter(X_train, y_train, color='blue')
#make the scatter plot of our training set and predicted regression results
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#make the scatter plot of our test set
plt.scatter(X_test, y_test, color='blue')
#make the scatter plot of our training set and predicted regression results
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()