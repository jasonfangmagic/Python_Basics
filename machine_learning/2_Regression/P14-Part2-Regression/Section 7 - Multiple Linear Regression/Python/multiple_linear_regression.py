# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Avoiding the Dummy Variable Trap
X=X[:, 1:]
print(X)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#only keep 2 decimals
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


print(regressor.predict([[ 1, 0, 160000, 130000, 300000]]))

print(regressor.coef_)
print(regressor.intercept_)

# Building the optimal model using Backward Elimination
# import statsmodels.formula.api as sm
# X=np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
# X_opt=X[:, [0, 1, 2, 3, 4, 5]]
# regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit() regressor_OLS.summary() X_opt=X[:, [0, 1, 3, 4, 5]] regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit() regressor_OLS.summary() X_opt=X[:, [0, 3, 4, 5]]
# regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit() regressor_OLS.summary() X_opt=X[:, [0, 3, 5]]
# regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit() regressor_OLS.summary() X_opt=X[:, [0, 3]]
# regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit() regressor_OLS.summary()
