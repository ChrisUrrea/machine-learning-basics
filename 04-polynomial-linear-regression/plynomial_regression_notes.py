#Polynomial regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#we want buy convention X to be a matrix, hence the 1:2 notation
X = dataset.iloc[:, 1:2].values
#likewise we want u as a vector
y = dataset.iloc[:, -1].values

#when we have small dataset i.e. < 10, no need of
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting a Linear Regression to the dataset as a comparison reference
from sklearn.linear_model import LinearRegression
#create your linear regression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
#transforms our features X into a new matrix X poly, with X as different powers
#increase # of polynomial degrees, but becareful of overfitting, in this case
#we do not care as the range is constrained (junior to CEO).
poly = PolynomialFeatures(degree = 4)
#apply our poly features to X
#poly reg automatically adds a column of ones in case of 
X_poly = poly.fit_transform(X)
#use our polynomial matrix to create a polynomial regression model
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

#Visualing our linear regression
#plot our points
plt.scatter(X, y, color ='red')
#plot our line
plt.plot(X, lin_reg.predict(X), color = 'blue')
#label our graph
plt.title('Bluffing? (linear regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#Visualizing our polynomial regression
#smooth our grid to smooth our line (by 0.1 pieces)
x_grid = np.arange(min(X), max(X), 0.1)

#change our xgrid from vector to matrix, in order to pass on to plt.plot
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(X, y, color ='red')

#plot our line with respect to X and our poly_reg features 
#with respect to the transformation x
#we put 'poly.fit_transform(X)' instead of our saved X variable
# as it's much more dynamic to changes in X
plt.plot(x_grid, poly_reg.predict(poly.fit_transform(x_grid)), color = 'blue')
#label our graph
plt.title('Bluffing? (polynomial regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#predict a specific value from a matrix regression
#transform 6.5 from poly matrix into a vector that poly regression model can interpret
poly_reg.predict(poly.fit_transform(6.5))