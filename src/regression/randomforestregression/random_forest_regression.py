# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:50:25 2020

@author: Ambarish Kapil
"""

import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:2].values
Y = df.iloc[:, 2:].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X,Y)

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.scatter(X,Y,color = 'b')
ax.plot(X_grid,regressor.predict(X_grid),color='r')
ax.set_title('Salary-Position Level plot')
ax.set_xlabel('level')
ax.set_ylabel('Salary')
plt.show()