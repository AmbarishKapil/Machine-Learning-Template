# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:43:41 2020

@author: Ambarish
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1:].values

# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Feature scaling is handled by scikitlearn for linear regression

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)

# Plotting results
import matplotlib.pyplot as plt
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,Y_pred,color='blue')
plt.show()


