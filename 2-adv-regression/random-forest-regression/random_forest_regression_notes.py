# Random Forest Regression (Ensemble model - bundle of models)

# a team of Decesion Tree Regressions formed on different subsets of the dataset
# ultimate prediction of Random Forest is simply
# the average of all Decision Tree Regression's of the subsets
#Makes it more resistant to sharp changes in X

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Create our Random Forest Regressor here
from sklearn.ensemble import RandomForestRegressor
# n_estimators - how many decision tree's in your forest
#play around with n to find best fit (team of trees)
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Fitting the Random Forest Regression Model to the dataset

# Predicting a new result
y_pred = regressor.predict(6.5)

# Random Forest - better to visualize higher resoltuion 
# to see each AVG step interval of our random forest
# Visualising the Regression results for higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Random forest has many more steps in our regression
# Each step a calculation of the average of all decision tree's step (team avg.)
# much more accurate, and resistant to new X examples or outliers

#Note: more decision trees not necessairly more steps,
#      as random forest converges to same set of avg's