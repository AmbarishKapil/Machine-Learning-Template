# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:37:17 2020

@author: Ambarish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../../data/Position_Salaries.csv")
X = df.iloc[:, 1:2].values
Y = df.iloc[:, 2:].values

# Here we are not making train-test-split as the dataset is small

# We do not need feature scaling for decision trees

# Creating andTraing the model\
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X,Y)

# plotting the decision tree(this can only be done for when both x and y are 1D)
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.scatter(X,Y,color = 'r')
ax.plot(X_grid,regressor.predict(X_grid),color = 'b')
ax.set_title('Salary vs Experience level')
ax.set_xlabel('position level')
ax.set_ylabel('salary')
plt.show()