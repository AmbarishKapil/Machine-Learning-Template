# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../../data/Position_Salaries.csv")
X = df.iloc[:,1:2].values
Y = df.iloc[:,2:].values

# We won't make the train test split for this dataset

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Training svr
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

# Plotting
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_title('svr plot')
ax.set_xlabel('no. of years')
ax.set_ylabel('salary per annum')
ax.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(Y),color = 'r')
ax.plot(sc_X.inverse_transform(X),sc_Y.inverse_transform(regressor.predict(X)))