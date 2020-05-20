# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_salaries.csv')
X = df.iloc[:,1:2].values
Y = df.iloc[:,2:].values

from sklearn.preprocessing import  PolynomialFeatures
poly_feature = PolynomialFeatures(degree = 2)
X_poly = poly_feature.fit_transform(X)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly,Y)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.scatter(X,Y,color = 'r')
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
ax.plot(X_grid,lin_reg.predict(poly_feature.fit_transform(X_grid)))
ax.set_title('level-salary curve')
ax.set_xlabel('level')
ax.set_ylabel('salary')
plt.show()