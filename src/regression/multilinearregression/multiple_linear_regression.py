# Importing Librsries
import pandas as pd
import numpy as np

# Importing Dataset
dataset = pd.read_csv("../../../data/50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1:].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Dummy Variable trap is handled by the library

# Train-Test Split
from  sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)

# Freature Scaling is handled by the library

'''
# Backward Elimamination
import statsmodels.formula.api as sm
# We append a column of ones in the beginning of the matrix of features as
# the library doesnot take care of that
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

SL = 0.05

X_opt = X[:, [0,1,2,3,4,5]]
is_optimal = True

while True:
    regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
    for pi, pv in enumerate(regressor_OLS.pvalues):
        if pv > SL:
            is_optimal = False
            X_opt = np.delete(X_opt, pi, axis = 1)
            break
    if is_optimal:
        break
    is_optimal = True
'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
