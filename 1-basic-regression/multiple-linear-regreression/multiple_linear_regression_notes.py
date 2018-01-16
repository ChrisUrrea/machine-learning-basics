# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode our categorical variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_state = LabelEncoder()
#Encoding our state variable
X[:,3] = labelencoder_state.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#now cols (0,1,2) : {California, Florida, New York}


#Avoiding dummy variable trap
#take away one dummy variable to make sure
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# library will take care of feature scaling automatically for multuple linear regression
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#create a muliple linear regression model that optimizes profit(dependent variable)
from sklearn.linear_model import LinearRegression

#fit our multiple linear regression to our training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting test results with our trained model
y_pred = regressor.predict(X_test)

#Building our optimal model using backward elimination of variables

#use to compute p-values and calculate statistical significance of our variables
import statsmodels.formula.api as sm

#add our standard coefficient (b0) as a column of ones
# as integers to prevent data type errors
ones = np.ones((50,1)).astype(int)

#add our matrix X to our column of ones, so our coefficient is located in the first column
#axis=1 since we're adding a column. axis=0(default) if we were adding a new row
X = np.append(arr=ones, values = X, axis=1)

#create our optimal X matrix of our statistically signifcant features
X_opt = X[:, [0,1,2,3,4,5]]
#create a new regressor OLS for our optimal features X matrix
#using our sm library's OLS regressor train our X to our y
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()

#print out our statstic table of our regression model
#check if our features of our OLS regression are statisitically signifcant (checking p-values)
regressor_OLS.summary()

#delete insignificant variable 2, and retrain model
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()

#delete insignificant variable 1, and retrain model
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()

#delete insignificant variable 4, and retrain model
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()

