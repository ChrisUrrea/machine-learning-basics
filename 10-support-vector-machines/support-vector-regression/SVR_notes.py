# Regression Template

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

#For SVR regression, library does not do feature scaling
#Thus, we must do it ourselves
# Feature Scaling
from sklearn.preprocessing import StandardScaler
#create a standard scaler instance for each Matrix
sc_X = StandardScaler()
sc_y = StandardScaler()
#transform tool feature scales X and y using 
X = sc_X.fit_transform(X)
#reshape  y as vector to pass onto transform
y = y.reshape(-1, 1)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
# Create your regressor here 
# Choose your kernel depending on dataset, rbf (Gaussian) is default
regressor = SVR(kernel = 'rbf')
# Fit your regressor to your dataset
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred)

# Visualising the Regression results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

