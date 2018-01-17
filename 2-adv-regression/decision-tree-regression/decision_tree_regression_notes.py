# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

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

# Fitting the Regression Model to the dataset
# import your Decision Tree Regression class
from sklearn.tree import DecisionTreeRegressor
# create  your tree regressor from instance of DecisionTreeRegression class
regressor = DecisionTreeRegressor(random_state=0)

# Fitting the Decision Tree Regression Model to the dataset
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Decision Tree makes decision based on resolution picked,
# Since Decision Tree is a non-continous regression model!

# Visualising the Decision Tree Regression results for low resolution
# Unable to see visible intervals
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Decision Regression results for higher resolution
# We can now see the visibile step intervals
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Decision Tree model is best suited for hiher level models

# Based on information entroy and information gain, our range is split into 
# Intervals, seperated by vertical lines and from which the 
# horizontal line is the aerage value within the interval